"""Uplift model implementations.

The models here are intentionally simple and explicit. They are designed for
learning, auditing, and policy evaluation rather than maximum leaderboard score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from uplift_marketing.features import (
    FeatureSpec,
    _validated_binary_indicator,
    build_preprocessor,
)

ArrayLike: TypeAlias = NDArray[Any] | pd.Series


class UpliftMetaLearner(Protocol):
    """Structural interface every meta-learner in this module exposes."""

    def fit(self, x: pd.DataFrame, treatment: ArrayLike, y: ArrayLike) -> UpliftMetaLearner: ...
    def predict_uplift(self, x: pd.DataFrame) -> NDArray[Any]: ...


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


def _positive_class_probability(model: BaseEstimator, x: pd.DataFrame) -> NDArray[Any]:
    """Return ``P(Y=1)`` using a fitted classifier's class-column mapping.

    Scikit-learn orders the columns of ``predict_proba`` according to
    ``classes_``. A model trained on only negative outcomes exposes one
    column of ones for class 0; those are *not* positive-class probabilities.
    """

    predict_proba = getattr(model, "predict_proba", None)
    if not callable(predict_proba):
        raise TypeError("The model must expose predict_proba")

    probabilities = np.asarray(predict_proba(x), dtype=float)
    if probabilities.ndim != 2:
        raise ValueError("predict_proba must return a two-dimensional array")
    if probabilities.shape[0] != len(x):
        raise ValueError("predict_proba must return one probability row per sample")

    raw_classes = getattr(model, "classes_", None)
    if raw_classes is None:
        raise ValueError("Fitted classifier must expose classes_ to identify P(Y=1)")
    classes = np.asarray(raw_classes)
    if classes.ndim != 1 or len(classes) != probabilities.shape[1]:
        raise ValueError("classes_ must match the probability columns")
    if not 1 <= len(classes) <= 2 or set(classes.tolist()) - {0, 1}:
        raise ValueError("Classifier classes_ must contain binary outcome labels 0/1")
    if len(set(classes.tolist())) != len(classes):
        raise ValueError("Classifier classes_ must contain unique labels")

    positive_columns = np.flatnonzero(classes == 1)
    if len(positive_columns) == 0:
        # A single-class negative training fold has P(Y=1) = 0 everywhere.
        return np.zeros(len(x), dtype=float)
    return np.asarray(probabilities[:, int(positive_columns[0])], dtype=float)


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

    def fit(self, x: pd.DataFrame, treatment: ArrayLike, y: ArrayLike) -> SLearner:
        """Fit the S-learner on fully observed binary experiment labels."""

        if len(x) != len(treatment) or len(x) != len(y):
            raise ValueError("x, treatment, and y must have the same length")

        treatment_array = _validated_binary_indicator(
            treatment, name="treatment", require_both=True
        )
        y_array = _validated_binary_indicator(y, name="outcome")
        training_frame = x.copy()
        training_frame[self.treatment_col] = treatment_array

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
        self.pipeline_.fit(training_frame, y_array)
        return self

    def predict_potential_outcomes(self, x: pd.DataFrame) -> tuple[NDArray[Any], NDArray[Any]]:
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

    def predict_uplift(self, x: pd.DataFrame) -> NDArray[Any]:
        """Predict individual uplift scores."""

        p_control, p_treated = self.predict_potential_outcomes(x)
        return np.asarray(p_treated - p_control, dtype=float)


def make_uplift_regressor(random_state: int = 42) -> RegressorMixin:
    """Create the default regressor used by the X-learner second stage.

    The X-learner regresses imputed individual treatment effects onto features.
    Ridge is used as a stable, interpretable default.
    """

    return Ridge(alpha=1.0, random_state=random_state)


def make_calibrated_classifier(
    base_estimator: ClassifierMixin | None = None,
    *,
    method: str = "isotonic",
    cv: int = 5,
    random_state: int = 42,
) -> ClassifierMixin:
    """Wrap a classifier so its `predict_proba` is well-calibrated.

    Probabilities from a vanilla logistic regression on rare-event data
    (e.g. Hillstrom's ~1 percent conversion rate) can be miscalibrated by
    several times their true level, which inflates the absolute uplift
    estimates produced by S-, T-, and X-learners. Wrapping the base
    classifier in scikit-learn's :class:`CalibratedClassifierCV` fits an
    isotonic (or sigmoid) calibrator on cross-validated out-of-fold
    predictions and is the simplest fix.

    Parameters
    ----------
    base_estimator:
        Classifier to wrap. Defaults to the same logistic baseline used
        elsewhere in the project.
    method:
        Calibration method, ``"isotonic"`` (non-parametric, recommended for
        rare-event data with enough samples) or ``"sigmoid"`` (Platt scaling).
    cv:
        Number of folds used by the calibrator's internal cross-validation.
    random_state:
        Seed used to construct the default base estimator.
    """

    if base_estimator is None:
        base_estimator = make_base_classifier(random_state)
    return CalibratedClassifierCV(estimator=base_estimator, method=method, cv=cv)


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

    def fit(self, x: pd.DataFrame, treatment: ArrayLike, y: ArrayLike) -> TLearner:
        """Fit separate treated and control outcome models."""

        if len(x) != len(treatment) or len(x) != len(y):
            raise ValueError("x, treatment, and y must have the same length")

        treatment_array = _validated_binary_indicator(
            treatment, name="treatment", require_both=True
        )
        y_array = _validated_binary_indicator(y, name="outcome")

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

    def predict_potential_outcomes(self, x: pd.DataFrame) -> tuple[NDArray[Any], NDArray[Any]]:
        """Return predicted probabilities under control and treatment."""

        if self.treated_pipeline_ is None or self.control_pipeline_ is None:
            raise RuntimeError("Model is not fitted")

        p_control = _positive_class_probability(self.control_pipeline_, x)
        p_treated = _positive_class_probability(self.treated_pipeline_, x)
        return p_control, p_treated

    def predict_uplift(self, x: pd.DataFrame) -> NDArray[Any]:
        """Predict individual uplift scores."""

        p_control, p_treated = self.predict_potential_outcomes(x)
        return np.asarray(p_treated - p_control, dtype=float)


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

    def fit(self, x: pd.DataFrame, treatment: ArrayLike, y: ArrayLike) -> XLearner:
        """Fit the X-learner on randomized treatment/control data."""

        if len(x) != len(treatment) or len(x) != len(y):
            raise ValueError("x, treatment, and y must have the same length")

        treatment_array = _validated_binary_indicator(
            treatment, name="treatment", require_both=True
        )
        y_array = _validated_binary_indicator(y, name="outcome")

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

    def predict_potential_outcomes(self, x: pd.DataFrame) -> tuple[NDArray[Any], NDArray[Any]]:
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
        propensity: float | NDArray[Any] = 0.5,
    ) -> NDArray[Any]:
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

        tau_treated = np.asarray(self.treated_effect_pipeline_.predict(x), dtype=float)
        tau_control = np.asarray(self.control_effect_pipeline_.predict(x), dtype=float)

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
        return np.asarray(g * tau_control + (1.0 - g) * tau_treated, dtype=float)


def cross_fit_uplift_oof(
    model_factory: Callable[[], UpliftMetaLearner],
    x: pd.DataFrame,
    treatment: ArrayLike,
    y: ArrayLike,
    *,
    n_splits: int = 5,
    random_state: int = 42,
) -> NDArray[Any]:
    """Out-of-fold uplift predictions via stratified K-fold cross-fitting.

    Partitions the data into K folds stratified on the joint
    treatment-outcome label, fits a fresh meta-learner on the K-1 training
    folds, and scores the held-out fold with that fitted model. The result
    is one uplift prediction per row that was *not* used to fit the model
    that produced it.

    This is the standard remedy for the optimistic AUQ inflation that
    happens when a T-learner (or X-learner) is evaluated on the same rows
    that fit its arm-specific outcome models. AUQ on cross-fit predictions
    is what should be reported when comparing meta-learners.

    Parameters
    ----------
    model_factory:
        Zero-argument callable that returns a fresh, unfitted meta-learner.
        For example: ``lambda: TLearner(spec=spec, random_state=42)``.
    x:
        Feature frame.
    treatment:
        Binary treatment indicator (0/1).
    y:
        Binary outcome (0/1).
    n_splits:
        Number of stratified folds.
    random_state:
        Seed for the fold assignment.
    """

    if not callable(model_factory):
        raise TypeError("model_factory must be a zero-argument callable")
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")

    x_idx = x.reset_index(drop=True)
    treatment_arr = _validated_binary_indicator(treatment, name="treatment", require_both=True)
    y_arr = _validated_binary_indicator(y, name="outcome")
    if len(x_idx) != len(treatment_arr) or len(x_idx) != len(y_arr):
        raise ValueError("x, treatment, and y must have the same length")
    if np.bincount(treatment_arr, minlength=2).min() < n_splits:
        raise ValueError("Each treatment arm must contain at least n_splits observations")

    stratify = treatment_arr.astype(str).astype(object) + "_" + y_arr.astype(str).astype(object)

    oof = np.empty(len(x_idx), dtype=float)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for tr_idx, te_idx in skf.split(np.zeros(len(x_idx)), stratify):
        model = model_factory()
        model.fit(
            x_idx.iloc[tr_idx],
            treatment_arr[tr_idx],
            y_arr[tr_idx],
        )
        oof[te_idx] = np.asarray(model.predict_uplift(x_idx.iloc[te_idx]))
    return oof


@dataclass
class DRLearner:
    """Doubly robust uplift estimator with cross-fitting.

    The DR-learner combines an outcome model per arm, a propensity model,
    and an Augmented Inverse Propensity Weighting (AIPW) pseudo-outcome:

    .. math::
        \\psi_i = \\hat\\mu_1(X_i) - \\hat\\mu_0(X_i)
                + \\frac{T_i}{\\hat e(X_i)} (Y_i - \\hat\\mu_1(X_i))
                - \\frac{1 - T_i}{1 - \\hat e(X_i)} (Y_i - \\hat\\mu_0(X_i))

    A second-stage regressor maps :math:`X` to :math:`\\hat\\tau(x)` by
    fitting on those pseudo-outcomes. The estimator is *doubly robust*:
    consistent for the conditional treatment effect when *either* the
    outcome model or the propensity model is well-specified.

    Cross-fitting is enabled by default (``n_splits=5``) so the
    pseudo-outcome for row :math:`i` is built from models that were trained
    on folds *not* containing :math:`i`. This is what makes the resulting
    estimator efficient under the standard DML conditions.
    """

    spec: FeatureSpec
    base_model: ClassifierMixin | None = None
    propensity_model: ClassifierMixin | None = None
    effect_model: RegressorMixin | None = None
    n_splits: int = 5
    propensity_clip: tuple[float, float] = (0.05, 0.95)
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.base_model is None:
            self.base_model = make_base_classifier(self.random_state)
        if self.propensity_model is None:
            self.propensity_model = make_base_classifier(self.random_state)
        if self.effect_model is None:
            self.effect_model = make_uplift_regressor(self.random_state)
        if self.n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        clip_lo, clip_hi = self.propensity_clip
        if not 0 < clip_lo < clip_hi < 1:
            raise ValueError("propensity_clip must satisfy 0 < lo < hi < 1")
        self.effect_pipeline_: Pipeline | None = None
        self.cross_fit_pseudo_outcomes_: NDArray[Any] | None = None

    def fit(self, x: pd.DataFrame, treatment: ArrayLike, y: ArrayLike) -> DRLearner:
        """Fit the DR-learner via cross-fitted pseudo-outcomes."""

        x_local = x.reset_index(drop=True)
        treatment_arr = _validated_binary_indicator(treatment, name="treatment", require_both=True)
        y_arr = _validated_binary_indicator(y, name="outcome")
        if len(x_local) != len(treatment_arr) or len(x_local) != len(y_arr):
            raise ValueError("x, treatment, and y must have the same length")
        if np.bincount(treatment_arr, minlength=2).min() < self.n_splits:
            raise ValueError("Each treatment arm must contain at least n_splits observations")

        n = len(x_local)
        pseudo = np.zeros(n, dtype=float)
        clip_lo, clip_hi = self.propensity_clip
        stratify = (
            treatment_arr.astype(str).astype(object)
            + "_"
            + y_arr.astype(str).astype(object)
        )
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        for tr_idx, te_idx in skf.split(np.zeros(n), stratify):
            x_tr = x_local.iloc[tr_idx]
            x_te = x_local.iloc[te_idx]
            t_tr = treatment_arr[tr_idx]
            y_tr = y_arr[tr_idx]
            t_te = treatment_arr[te_idx].astype(float)
            y_te = y_arr[te_idx].astype(float)

            mu0 = Pipeline(
                steps=[
                    ("preprocessor", build_preprocessor(self.spec)),
                    ("model", clone(self.base_model)),
                ]
            )
            mu1 = Pipeline(
                steps=[
                    ("preprocessor", build_preprocessor(self.spec)),
                    ("model", clone(self.base_model)),
                ]
            )
            propensity = Pipeline(
                steps=[
                    ("preprocessor", build_preprocessor(self.spec)),
                    ("model", clone(self.propensity_model)),
                ]
            )

            ctrl_mask = t_tr == 0
            trtd_mask = t_tr == 1
            mu0.fit(x_tr.iloc[ctrl_mask], y_tr[ctrl_mask])
            mu1.fit(x_tr.iloc[trtd_mask], y_tr[trtd_mask])
            propensity.fit(x_tr, t_tr)

            mu0_te = _positive_class_probability(mu0, x_te)
            mu1_te = _positive_class_probability(mu1, x_te)
            e_te = np.clip(
                _positive_class_probability(propensity, x_te),
                clip_lo,
                clip_hi,
            )

            pseudo[te_idx] = (
                (mu1_te - mu0_te)
                + t_te * (y_te - mu1_te) / e_te
                - (1.0 - t_te) * (y_te - mu0_te) / (1.0 - e_te)
            )

        self.cross_fit_pseudo_outcomes_ = pseudo
        self.effect_pipeline_ = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(self.spec)),
                ("model", clone(self.effect_model)),
            ]
        )
        self.effect_pipeline_.fit(x_local, pseudo)
        return self

    def predict_uplift(self, x: pd.DataFrame) -> NDArray[Any]:
        """Predict uplift via the second-stage regressor."""

        if self.effect_pipeline_ is None:
            raise RuntimeError("Model is not fitted")
        return np.asarray(self.effect_pipeline_.predict(x))
