"""Uplift model implementations.

The models here are intentionally simple and explicit. They are designed for
learning, auditing, and policy evaluation rather than maximum leaderboard score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
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


def make_uplift_regressor(random_state: int = 42) -> RegressorMixin:
    """Create the default regressor used by the X-learner second stage.

    The X-learner regresses imputed individual treatment effects onto features.
    Ridge is used as a stable, interpretable default.
    """

    return Ridge(alpha=1.0, random_state=random_state)


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


@dataclass
class XLearner:
    """X-learner uplift model (Künzel et al., 2019).

    The X-learner refines the T-learner by regressing imputed individual
    treatment effects on features. The estimator proceeds in three stages:

    1. Fit two outcome models, one on treated and one on control, exactly as a
       T-learner would.
    2. Build pseudo-effects on each arm by combining observed outcomes with
       counterfactual predictions from the opposite-arm model.
    3. Fit one regressor per arm to map features to those pseudo-effects, then
       blend the two regressors using a propensity weight ``g``.

    For randomized data the propensity is constant and ``propensity`` defaults
    to 0.5. For non-randomized data, pass an estimated propensity per row when
    calling ``predict_uplift``.
    """

    spec: FeatureSpec
    base_model: ClassifierMixin | None = None
    effect_model: RegressorMixin | None = None
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.base_model is None:
            self.base_model = make_base_classifier(self.random_state)
        if self.effect_model is None:
            self.effect_model = make_uplift_regressor(self.random_state)
        self.treated_outcome_pipeline_: Pipeline | None = None
        self.control_outcome_pipeline_: Pipeline | None = None
        self.treated_effect_pipeline_: Pipeline | None = None
        self.control_effect_pipeline_: Pipeline | None = None

    def fit(self, x: pd.DataFrame, treatment: ArrayLike, y: ArrayLike) -> "XLearner":
        """Fit the X-learner on randomized treatment/control data."""

        if len(x) != len(treatment) or len(x) != len(y):
            raise ValueError("x, treatment, and y must have the same length")

        treatment_array = np.asarray(treatment).astype(int)
        y_array = np.asarray(y).astype(int)
        if set(np.unique(treatment_array)) != {0, 1}:
            raise ValueError("Both treatment arms 0 and 1 are required")

        treated_mask = treatment_array == 1
        control_mask = treatment_array == 0

        # Stage 1: outcome models per arm.
        self.treated_outcome_pipeline_ = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(self.spec)),
                ("model", clone(self.base_model)),
            ]
        )
        self.control_outcome_pipeline_ = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(self.spec)),
                ("model", clone(self.base_model)),
            ]
        )
        self.treated_outcome_pipeline_.fit(x.loc[treated_mask], y_array[treated_mask])
        self.control_outcome_pipeline_.fit(x.loc[control_mask], y_array[control_mask])

        # Stage 2: imputed treatment effects on each arm.
        # On treated rows, the counterfactual is the control-arm prediction.
        # On control rows, the counterfactual is the treated-arm prediction.
        x_treated = x.loc[treated_mask]
        x_control = x.loc[control_mask]
        mu0_on_treated = _positive_class_probability(self.control_outcome_pipeline_, x_treated)
        mu1_on_control = _positive_class_probability(self.treated_outcome_pipeline_, x_control)

        d_treated = y_array[treated_mask] - mu0_on_treated
        d_control = mu1_on_control - y_array[control_mask]

        # Stage 3: regress pseudo-effects on features.
        self.treated_effect_pipeline_ = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(self.spec)),
                ("model", clone(self.effect_model)),
            ]
        )
        self.control_effect_pipeline_ = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(self.spec)),
                ("model", clone(self.effect_model)),
            ]
        )
        self.treated_effect_pipeline_.fit(x_treated, d_treated)
        self.control_effect_pipeline_.fit(x_control, d_control)
        return self

    def predict_potential_outcomes(self, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return Stage 1 outcome probabilities under control and treatment.

        These are the T-learner-style predictions used internally; the X-learner
        improves the *uplift* estimate via the effect models, not the absolute
        outcome predictions.
        """

        if self.treated_outcome_pipeline_ is None or self.control_outcome_pipeline_ is None:
            raise RuntimeError("Model is not fitted")

        p_control = _positive_class_probability(self.control_outcome_pipeline_, x)
        p_treated = _positive_class_probability(self.treated_outcome_pipeline_, x)
        return p_control, p_treated

    def predict_uplift(
        self,
        x: pd.DataFrame,
        *,
        propensity: float | np.ndarray = 0.5,
    ) -> np.ndarray:
        """Predict individual uplift scores using the blended effect models.

        Parameters
        ----------
        x:
            Feature frame.
        propensity:
            Per-row probability of being treated. For randomized campaigns this
            is constant (the experimental treatment share). The default of 0.5
            matches the synthetic generator and a balanced A/B test.
        """

        if self.treated_effect_pipeline_ is None or self.control_effect_pipeline_ is None:
            raise RuntimeError("Model is not fitted")

        tau_treated = self.treated_effect_pipeline_.predict(x)
        tau_control = self.control_effect_pipeline_.predict(x)

        g = np.asarray(propensity, dtype=float)
        if g.ndim == 0:
            g = np.full(len(x), float(g))
        if g.shape != (len(x),):
            raise ValueError("propensity must be a scalar or have one value per row")
        if np.any((g < 0) | (g > 1)):
            raise ValueError("propensity must be in [0, 1]")

        # Künzel et al. weight: blend the two effect models by propensity.
        # Weighting puts more mass on the arm with more samples relative to the
        # population, which is the arm whose pseudo-effect is more reliable.
        return g * tau_control + (1.0 - g) * tau_treated
