"""Uplift model implementations.

The models here are intentionally simple and explicit. They are designed for
learning, auditing, and policy evaluation rather than maximum leaderboard score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from uplift_marketing.features import FeatureSpec, build_preprocessor


ArrayLike = np.ndarray | pd.Series


def make_base_classifier(random_state: int = 42) -> ClassifierMixin:
    """Create a stable baseline probabilistic classifier."""

    return LogisticRegression(max_iter=1_000, class_weight="balanced", random_state=random_state)


def make_black_box_classifier(random_state: int = 42) -> ClassifierMixin:
    """Create a stronger non-linear comparison model."""

    return RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=50,
        max_depth=8,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )


def _positive_class_probability(model: BaseEstimator, x: pd.DataFrame) -> np.ndarray:
    """Return positive-class probabilities from a fitted classifier."""

    if not hasattr(model, "predict_proba"):
        raise TypeError("The model must expose predict_proba")

    probabilities = model.predict_proba(x)  # type: ignore[attr-defined]
    if probabilities.shape[1] == 1:
        # Degenerate training fold with one class. Return the only probability.
        return probabilities[:, 0]
    return probabilities[:, 1]


@dataclass
class SLearner:
    """S-learner uplift model.

    The S-learner trains one outcome model using both customer features and the
    treatment indicator. Uplift is estimated by scoring each customer twice:
    once as treated and once as control.
    """

    spec: FeatureSpec
    base_model: ClassifierMixin | None = None
    random_state: int = 42
    treatment_col: str = "treatment"

    def __post_init__(self) -> None:
        if self.base_model is None:
            self.base_model = make_base_classifier(self.random_state)
        self.pipeline_: Pipeline | None = None

    def fit(self, x: pd.DataFrame, treatment: ArrayLike, y: ArrayLike) -> "SLearner":
        """Fit the S-learner."""

        if len(x) != len(treatment) or len(x) != len(y):
            raise ValueError("x, treatment, and y must have the same length")

        training_frame = x.copy()
        training_frame[self.treatment_col] = np.asarray(treatment).astype(int)

        spec_with_treatment = FeatureSpec(
            numerical=[*self.spec.numerical, self.treatment_col],
            categorical=self.spec.categorical,
            excluded=self.spec.excluded,
        )
        self.pipeline_ = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(spec_with_treatment)),
                ("model", clone(self.base_model)),
            ]
        )
        self.pipeline_.fit(training_frame, np.asarray(y).astype(int))
        return self

    def predict_potential_outcomes(self, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return predicted probabilities under control and treatment."""

        if self.pipeline_ is None:
            raise RuntimeError("Model is not fitted")

        control = x.copy()
        control[self.treatment_col] = 0
        treated = x.copy()
        treated[self.treatment_col] = 1

        p_control = _positive_class_probability(self.pipeline_, control)
        p_treated = _positive_class_probability(self.pipeline_, treated)
        return p_control, p_treated

    def predict_uplift(self, x: pd.DataFrame) -> np.ndarray:
        """Predict individual uplift scores."""

        p_control, p_treated = self.predict_potential_outcomes(x)
        return p_treated - p_control


@dataclass
class TLearner:
    """T-learner uplift model.

    The T-learner trains separate outcome models for treated and control users.
    Uplift is estimated as the difference between the two predicted outcomes.
    """

    spec: FeatureSpec
    base_model: ClassifierMixin | None = None
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.base_model is None:
            self.base_model = make_base_classifier(self.random_state)
        self.treated_pipeline_: Pipeline | None = None
        self.control_pipeline_: Pipeline | None = None

    def fit(self, x: pd.DataFrame, treatment: ArrayLike, y: ArrayLike) -> "TLearner":
        """Fit separate treated and control outcome models."""

        if len(x) != len(treatment) or len(x) != len(y):
            raise ValueError("x, treatment, and y must have the same length")

        treatment_array = np.asarray(treatment).astype(int)
        y_array = np.asarray(y).astype(int)
        if set(np.unique(treatment_array)) != {0, 1}:
            raise ValueError("Both treatment arms 0 and 1 are required")

        treated_mask = treatment_array == 1
        control_mask = treatment_array == 0

        self.treated_pipeline_ = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(self.spec)),
                ("model", clone(self.base_model)),
            ]
        )
        self.control_pipeline_ = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(self.spec)),
                ("model", clone(self.base_model)),
            ]
        )

        self.treated_pipeline_.fit(x.loc[treated_mask], y_array[treated_mask])
        self.control_pipeline_.fit(x.loc[control_mask], y_array[control_mask])
        return self

    def predict_potential_outcomes(self, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return predicted probabilities under control and treatment."""

        if self.treated_pipeline_ is None or self.control_pipeline_ is None:
            raise RuntimeError("Model is not fitted")

        p_control = _positive_class_probability(self.control_pipeline_, x)
        p_treated = _positive_class_probability(self.treated_pipeline_, x)
        return p_control, p_treated

    def predict_uplift(self, x: pd.DataFrame) -> np.ndarray:
        """Predict individual uplift scores."""

        p_control, p_treated = self.predict_potential_outcomes(x)
        return p_treated - p_control
