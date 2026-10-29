from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import train_test_split

from uplift_marketing.features import (
    FeatureSpec,
    infer_feature_spec,
    split_features_treatment_outcome,
)
from uplift_marketing.models import (
    DRLearner,
    SLearner,
    TLearner,
    XLearner,
    _positive_class_probability,
    cross_fit_uplift_oof,
    make_calibrated_classifier,
)


def test_s_learner_predicts_uplift(synthetic_data) -> None:  # type: ignore[no-untyped-def]
    spec = infer_feature_spec(synthetic_data, extra_excluded=["customer_id", "visit"])
    x, treatment, y = split_features_treatment_outcome(synthetic_data, spec)
    x_train, x_test, t_train, _, y_train, _ = train_test_split(
        x, treatment, y, test_size=0.25, random_state=1, stratify=treatment
    )
    model = SLearner(spec=spec, random_state=1).fit(x_train, t_train, y_train)
    uplift = model.predict_uplift(x_test)
    assert len(uplift) == len(x_test)


def test_t_learner_predicts_uplift(synthetic_data) -> None:  # type: ignore[no-untyped-def]
    spec = infer_feature_spec(synthetic_data, extra_excluded=["customer_id", "visit"])
    x, treatment, y = split_features_treatment_outcome(synthetic_data, spec)
    x_train, x_test, t_train, _, y_train, _ = train_test_split(
        x, treatment, y, test_size=0.25, random_state=1, stratify=treatment
    )
    model = TLearner(spec=spec, random_state=1).fit(x_train, t_train, y_train)
    uplift = model.predict_uplift(x_test)
    assert len(uplift) == len(x_test)


def test_x_learner_predicts_uplift(synthetic_data) -> None:  # type: ignore[no-untyped-def]
    spec = infer_feature_spec(synthetic_data, extra_excluded=["customer_id", "visit"])
    x, treatment, y = split_features_treatment_outcome(synthetic_data, spec)
    x_train, x_test, t_train, _, y_train, _ = train_test_split(
        x, treatment, y, test_size=0.25, random_state=1, stratify=treatment
    )
    model = XLearner(spec=spec, random_state=1).fit(x_train, t_train, y_train)
    uplift = model.predict_uplift(x_test)
    assert len(uplift) == len(x_test)
    # On a randomized sample, the default is the measured training allocation.
    training_share = float(np.mean(t_train))
    uplift_explicit = model.predict_uplift(
        x_test, propensity=np.full(len(x_test), training_share)
    )
    assert np.allclose(uplift, uplift_explicit)


class _ArmSizeEffectRegressor(RegressorMixin, BaseEstimator):
    """Produce distinguishable X-learner effects based on each arm's sample size."""

    def fit(self, x: NDArray[Any], y: NDArray[Any]) -> _ArmSizeEffectRegressor:
        """Use the arm's training share as a controlled prediction."""
        self.effect_ = len(x) / 100.0
        return self

    def predict(self, x: NDArray[Any]) -> NDArray[np.float64]:
        """Return one controlled effect prediction per customer."""
        return np.full(len(x), self.effect_, dtype=float)


def test_x_learner_uses_observed_randomization_rate_by_default() -> None:
    """A 20/80 experiment must not blend arm-specific effect models 50/50."""
    n_control, n_treated = 80, 20
    x = pd.DataFrame({"signal": np.arange(n_control + n_treated, dtype=float)})
    treatment = np.r_[np.zeros(n_control, dtype=int), np.ones(n_treated, dtype=int)]
    outcome = np.zeros(len(x), dtype=int)
    spec = FeatureSpec(numerical=["signal"], categorical=[], excluded=[])
    model = XLearner(
        spec=spec,
        base_model=DummyClassifier(),
        effect_model=_ArmSizeEffectRegressor(),
    ).fit(x, treatment, outcome)

    # tau_control = 0.8, tau_treated = 0.2. The documented blending formula
    # weights tau_control by treatment propensity (0.2), yielding 0.32.
    assert model.treatment_share_ == pytest.approx(0.2)
    np.testing.assert_allclose(model.predict_uplift(x), 0.32)
    np.testing.assert_allclose(model.predict_uplift(x, propensity=0.5), 0.5)
    np.testing.assert_allclose(model.predict_uplift(x, propensity=np.full(len(x), 0.8)), 0.68)


def test_make_calibrated_classifier_returns_calibrated_wrapper() -> None:
    cal = make_calibrated_classifier(method="isotonic", cv=3, random_state=1)
    assert isinstance(cal, CalibratedClassifierCV)
    assert cal.method == "isotonic"
    assert cal.cv == 3


def test_cross_fit_uplift_oof_shape_and_fold_count(synthetic_data) -> None:  # type: ignore[no-untyped-def]
    spec = infer_feature_spec(synthetic_data, extra_excluded=["customer_id", "visit"])
    x, treatment, y = split_features_treatment_outcome(synthetic_data, spec)
    oof = cross_fit_uplift_oof(
        lambda: TLearner(spec=spec, random_state=1),
        x, treatment, y,
        n_splits=3,
        random_state=1,
    )
    assert oof.shape == (len(x),)
    assert np.isfinite(oof).all()
    # OOF predictions should not all be identical (model produces variation).
    assert oof.std() > 0


def test_cross_fit_uplift_oof_validates_inputs(synthetic_data) -> None:  # type: ignore[no-untyped-def]
    spec = infer_feature_spec(synthetic_data, extra_excluded=["customer_id", "visit"])
    x, treatment, y = split_features_treatment_outcome(synthetic_data, spec)
    with pytest.raises(TypeError, match="callable"):
        cross_fit_uplift_oof("not_a_callable", x, treatment, y)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_splits"):
        cross_fit_uplift_oof(lambda: TLearner(spec=spec), x, treatment, y, n_splits=1)


def test_dr_learner_predicts_uplift(synthetic_data) -> None:  # type: ignore[no-untyped-def]
    spec = infer_feature_spec(synthetic_data, extra_excluded=["customer_id", "visit"])
    x, treatment, y = split_features_treatment_outcome(synthetic_data, spec)
    x_train, x_test, t_train, _, y_train, _ = train_test_split(
        x, treatment, y, test_size=0.25, random_state=1, stratify=treatment
    )
    model = DRLearner(spec=spec, n_splits=3, random_state=1).fit(x_train, t_train, y_train)
    uplift = model.predict_uplift(x_test)
    assert len(uplift) == len(x_test)
    assert np.isfinite(uplift).all()
    # Pseudo-outcomes are populated during fit.
    assert model.cross_fit_pseudo_outcomes_ is not None
    assert model.cross_fit_pseudo_outcomes_.shape == (len(x_train),)


def test_dr_learner_rejects_invalid_propensity_clip() -> None:
    from uplift_marketing.features import FeatureSpec

    spec = FeatureSpec(numerical=["x"], categorical=[], excluded=[])
    with pytest.raises(ValueError, match="propensity_clip"):
        DRLearner(spec=spec, propensity_clip=(0.5, 0.4))


class _ZeroOutcomeProbabilityClassifier(ClassifierMixin, BaseEstimator):
    """Deliberately misspecified outcome nuisance model for DR regression tests."""

    def fit(
        self, x: NDArray[Any], y: NDArray[Any]
    ) -> _ZeroOutcomeProbabilityClassifier:
        """Expose both binary outcome classes regardless of training labels."""
        self.classes_ = np.array([0, 1], dtype=np.int64)
        return self

    def predict_proba(self, x: NDArray[Any]) -> NDArray[np.float64]:
        """Incorrectly predict that conversion is impossible for every customer."""
        return np.tile(np.array([1.0, 0.0]), (len(x), 1))


@pytest.mark.parametrize("n_treated", [10, 30, 50, 70, 90])
def test_dr_learner_propensity_preserves_unbalanced_randomization(n_treated: int) -> None:
    """DR correction must recover the effect when outcome nuisances are wrong.

    Different proportions of customers are randomized to treatment. Their
    observed outcome is exactly the treatment indicator, so the effect is
    always 1.0. With intentionally incorrect zero-valued outcome models,
    identification depends entirely on an unweighted propensity near the
    actual treatment share. Class-balanced logistic regression instead
    predicts about 50%, biasing all imbalanced scenarios.
    """
    n_control = 100 - n_treated
    x = pd.DataFrame({"signal": np.zeros(n_control + n_treated)})
    treatment = np.r_[np.zeros(n_control, dtype=int), np.ones(n_treated, dtype=int)]
    outcome = treatment.copy()
    spec = FeatureSpec(numerical=["signal"], categorical=[], excluded=[])

    learner = DRLearner(
        spec=spec,
        base_model=_ZeroOutcomeProbabilityClassifier(),
        effect_model=DummyRegressor(strategy="mean"),
        n_splits=5,
        random_state=17,
    ).fit(x, treatment, outcome)

    assert learner.cross_fit_pseudo_outcomes_ is not None
    assert np.mean(learner.cross_fit_pseudo_outcomes_) == pytest.approx(1.0, abs=0.02)
    np.testing.assert_allclose(learner.predict_uplift(x), 1.0, atol=0.02)


def _constant_outcome_by_arm() -> tuple[pd.DataFrame, NDArray[np.int64], NDArray[np.int64]]:
    """Create balanced treatment groups with one outcome class per arm."""
    x = pd.DataFrame({"signal": np.linspace(-1.0, 1.0, 24)})
    treatment = (np.arange(len(x)) % 2).astype(np.int64)
    outcome = treatment.copy()  # All controls fail; all treated users convert.
    return x, treatment, outcome


def test_s_learner_respects_single_negative_outcome_class() -> None:
    x, treatment, _ = _constant_outcome_by_arm()
    y = np.zeros(len(x), dtype=np.int64)
    spec = FeatureSpec(numerical=["signal"], categorical=[], excluded=[])

    model = SLearner(spec=spec, base_model=DummyClassifier()).fit(x, treatment, y)
    p_control, p_treated = model.predict_potential_outcomes(x)

    # A one-column class-0 probability must represent P(Y=1)=0, not 1.
    assert np.allclose(p_control, 0.0)
    assert np.allclose(p_treated, 0.0)


def test_t_learner_handles_single_class_per_treatment_arm() -> None:
    x, treatment, y = _constant_outcome_by_arm()
    spec = FeatureSpec(numerical=["signal"], categorical=[], excluded=[])

    model = TLearner(spec=spec, base_model=DummyClassifier()).fit(x, treatment, y)
    p_control, p_treated = model.predict_potential_outcomes(x)

    assert np.allclose(p_control, 0.0)
    assert np.allclose(p_treated, 1.0)
    assert np.allclose(model.predict_uplift(x), 1.0)


def test_x_learner_handles_single_class_per_treatment_arm() -> None:
    x, treatment, y = _constant_outcome_by_arm()
    spec = FeatureSpec(numerical=["signal"], categorical=[], excluded=[])

    model = XLearner(spec=spec, base_model=DummyClassifier()).fit(x, treatment, y)
    p_control, p_treated = model.predict_potential_outcomes(x)

    assert np.allclose(p_control, 0.0)
    assert np.allclose(p_treated, 1.0)
    assert np.allclose(model.predict_uplift(x), 1.0)


def test_dr_learner_pseudo_outcomes_handle_single_class_arms() -> None:
    x, treatment, y = _constant_outcome_by_arm()
    spec = FeatureSpec(numerical=["signal"], categorical=[], excluded=[])

    model = DRLearner(
        spec=spec,
        base_model=DummyClassifier(),
        propensity_model=DummyClassifier(),
        n_splits=3,
        random_state=1,
    ).fit(x, treatment, y)

    assert model.cross_fit_pseudo_outcomes_ is not None
    # Both arm-specific outcomes are deterministic; AIPW's residuals are zero.
    assert np.allclose(model.cross_fit_pseudo_outcomes_, 1.0)
    assert np.allclose(model.predict_uplift(x), 1.0)


class _ReversedBinaryClassifier(BaseEstimator):
    """Minimal fitted classifier whose positive class is the first column."""

    classes_: NDArray[np.int64] = np.array([1, 0], dtype=np.int64)

    def predict_proba(self, x: pd.DataFrame) -> NDArray[np.float64]:
        """Return class-ordered probabilities for each sample."""
        return np.tile(np.array([0.8, 0.2]), (len(x), 1))


def test_positive_probability_respects_class_column_order() -> None:
    x = pd.DataFrame({"signal": [0.0, 1.0]})

    positive_probability = _positive_class_probability(_ReversedBinaryClassifier(), x)

    assert np.allclose(positive_probability, 0.8)


@pytest.mark.parametrize("learner_type", [SLearner, TLearner, XLearner, DRLearner])
@pytest.mark.parametrize(
    ("label", "invalid_value"),
    [
        ("treatment", 0.4),
        ("treatment", np.nan),
        ("treatment", np.inf),
        ("outcome", 0.4),
        ("outcome", np.nan),
    ],
)
def test_meta_learners_validate_binary_labels_before_cast(
    learner_type: type[SLearner] | type[TLearner] | type[XLearner] | type[DRLearner],
    label: str,
    invalid_value: float,
) -> None:
    """Direct model users must not silently truncate observed experiment labels."""
    x, treatment, outcome = _constant_outcome_by_arm()
    bad_treatment = treatment.astype(float)
    bad_outcome = outcome.astype(float)
    if label == "treatment":
        bad_treatment[0] = invalid_value
    else:
        bad_outcome[0] = invalid_value

    spec = FeatureSpec(numerical=["signal"], categorical=[], excluded=[])
    model = learner_type(spec=spec, base_model=DummyClassifier())
    with pytest.raises(ValueError, match=f"{label}.*binary 0/1"):
        model.fit(x, bad_treatment, bad_outcome)


@pytest.mark.parametrize("learner_type", [SLearner, TLearner, XLearner, DRLearner])
def test_meta_learners_require_both_treatment_arms(
    learner_type: type[SLearner] | type[TLearner] | type[XLearner] | type[DRLearner],
) -> None:
    """Without controls, modelled counterfactuals have no empirical support."""
    x, _, outcome = _constant_outcome_by_arm()
    spec = FeatureSpec(numerical=["signal"], categorical=[], excluded=[])
    model = learner_type(spec=spec, base_model=DummyClassifier())
    with pytest.raises(ValueError, match="Both treatment arms"):
        model.fit(x, np.ones(len(x), dtype=int), outcome)


@pytest.mark.parametrize("learner_type", [SLearner, TLearner, XLearner, DRLearner])
def test_meta_learners_accept_boolean_experimental_labels(
    learner_type: type[SLearner] | type[TLearner] | type[XLearner] | type[DRLearner],
) -> None:
    """Bool-labelled experiments remain supported through direct fit calls."""
    x, treatment, outcome = _constant_outcome_by_arm()
    spec = FeatureSpec(numerical=["signal"], categorical=[], excluded=[])
    model = learner_type(spec=spec, base_model=DummyClassifier(), random_state=1)
    model.fit(x, treatment.astype(bool), outcome.astype(bool))
    assert np.isfinite(model.predict_uplift(x)).all()


@pytest.mark.parametrize(
    ("label", "invalid_value"),
    [("treatment", 0.4), ("treatment", np.nan), ("outcome", 0.4), ("outcome", np.inf)],
)
def test_cross_fit_rejects_invalid_binary_labels(
    label: str, invalid_value: float
) -> None:
    """OOF metrics must not be computed from altered treatment or outcome labels."""
    x, treatment, outcome = _constant_outcome_by_arm()
    bad_treatment = treatment.astype(float)
    bad_outcome = outcome.astype(float)
    if label == "treatment":
        bad_treatment[0] = invalid_value
    else:
        bad_outcome[0] = invalid_value
    spec = FeatureSpec(numerical=["signal"], categorical=[], excluded=[])

    with pytest.raises(ValueError, match=f"{label}.*binary 0/1"):
        cross_fit_uplift_oof(
            lambda: TLearner(spec=spec, base_model=DummyClassifier()),
            x,
            bad_treatment,
            bad_outcome,
            n_splits=3,
        )


def test_cross_fit_rejects_insufficient_treatment_arm_count() -> None:
    """A fold count larger than an arm cannot support honest OOF comparison."""
    x, _, outcome = _constant_outcome_by_arm()
    treatment = np.r_[np.zeros(22, dtype=int), np.ones(2, dtype=int)]
    spec = FeatureSpec(numerical=["signal"], categorical=[], excluded=[])

    with pytest.raises(ValueError, match="at least n_splits"):
        cross_fit_uplift_oof(
            lambda: TLearner(spec=spec, base_model=DummyClassifier()),
            x,
            treatment,
            outcome,
            n_splits=3,
        )


def test_cross_fit_accepts_boolean_labels() -> None:
    """OOF predictions work on Boolean treatment and outcome vectors."""
    x, treatment, outcome = _constant_outcome_by_arm()
    spec = FeatureSpec(numerical=["signal"], categorical=[], excluded=[])
    uplift = cross_fit_uplift_oof(
        lambda: TLearner(spec=spec, base_model=DummyClassifier()),
        x,
        treatment.astype(bool),
        outcome.astype(bool),
        n_splits=3,
    )
    assert uplift.shape == (len(x),)
    assert np.isfinite(uplift).all()


class _FixedProbabilityClassifier(BaseEstimator):
    """Minimal classifier with explicitly controlled probability columns."""

    classes_: NDArray[np.int64] = np.array([0, 1], dtype=np.int64)

    def __init__(self, probabilities: NDArray[np.float64]) -> None:
        self.probabilities = probabilities

    def predict_proba(self, x: pd.DataFrame) -> NDArray[np.float64]:
        """Repeat the specified two-column probability row for each customer."""
        return np.tile(self.probabilities, (len(x), 1))


@pytest.mark.parametrize(
    ("probabilities", "message"),
    [
        (np.array([np.nan, 1.0]), "finite"),
        (np.array([0.0, np.inf]), "finite"),
        (np.array([-0.1, 1.1]), r"\[0, 1\]"),
        (np.array([0.6, 0.6]), "sum to one"),
        (np.array([0.0, 0.0]), "sum to one"),
    ],
)
def test_positive_class_probability_rejects_invalid_distributions(
    probabilities: NDArray[np.float64], message: str
) -> None:
    """Invalid nuisance probabilities must not enter treatment-effect estimates."""
    x = pd.DataFrame({"signal": [0.0, 1.0]})
    with pytest.raises(ValueError, match=message):
        _positive_class_probability(_FixedProbabilityClassifier(probabilities), x)


def test_positive_class_probability_accepts_normalized_distribution() -> None:
    x = pd.DataFrame({"signal": [0.0, 1.0]})
    predicted = _positive_class_probability(
        _FixedProbabilityClassifier(np.array([0.4, 0.6])), x
    )
    np.testing.assert_allclose(predicted, [0.6, 0.6])


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -0.1, 1.1])
@pytest.mark.parametrize("as_vector", [False, True])
def test_x_learner_rejects_invalid_propensity_weights(
    invalid: float, as_vector: bool
) -> None:
    """A missing or impossible propensity cannot be used as a blending weight."""
    x, treatment, outcome = _constant_outcome_by_arm()
    spec = FeatureSpec(numerical=["signal"], categorical=[], excluded=[])
    model = XLearner(spec=spec, base_model=DummyClassifier()).fit(x, treatment, outcome)
    propensity: float | NDArray[np.float64]
    if as_vector:
        propensity = np.full(len(x), 0.5)
        propensity[0] = invalid
    else:
        propensity = invalid

    with pytest.raises(ValueError, match=r"propensity.*finite|propensity.*\[0, 1\]"):
        model.predict_uplift(x, propensity=propensity)


def _effect_predictions(kind: str, n_rows: int) -> NDArray[Any]:
    """Build malformed or extrapolated effect predictions for boundary tests."""

    if kind == "scalar":
        return np.asarray(0.3)
    if kind == "column":
        return np.full((n_rows, 1), 0.3)
    if kind == "short":
        return np.full(n_rows - 1, 0.3)
    if kind == "nan":
        return np.full(n_rows, np.nan)
    if kind == "infinite":
        return np.full(n_rows, np.inf)
    if kind == "complex":
        return np.full(n_rows, 0.3 + 0.1j)
    if kind == "nonnumeric":
        return np.full(n_rows, "not-an-effect")
    if kind == "extrapolated":
        return np.full(n_rows, 1.5)
    raise ValueError(f"Unknown prediction shape: {kind}")


class _ControlledUpliftModel:
    """OOF meta-learner whose prediction contract can be deliberately broken."""

    def __init__(self, kind: str) -> None:
        self.kind = kind

    def fit(
        self,
        x: pd.DataFrame,
        treatment: NDArray[Any] | pd.Series,
        y: NDArray[Any] | pd.Series,
    ) -> _ControlledUpliftModel:
        """Stand in for fitting without changing the controlled prediction."""
        return self

    def predict_uplift(self, x: pd.DataFrame) -> NDArray[Any]:
        """Return the requested malformed or valid uplift predictions."""
        return _effect_predictions(self.kind, len(x))


class _ControlledEffectRegressor(RegressorMixin, BaseEstimator):
    """Cloneable effect-stage regressor exposing the same boundary cases."""

    def __init__(self, kind: str) -> None:
        self.kind = kind

    def fit(self, x: NDArray[Any], y: NDArray[Any]) -> _ControlledEffectRegressor:
        """Mark the regression stage as fitted for the sklearn pipeline."""
        self.fitted_ = True
        return self

    def predict(self, x: NDArray[Any]) -> NDArray[Any]:
        """Return a deliberately controlled second-stage effect vector."""
        return _effect_predictions(self.kind, len(x))


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("scalar", "one prediction per row"),
        ("column", "one prediction per row"),
        ("short", "one prediction per row"),
        ("nan", "finite"),
        ("infinite", "finite"),
        ("complex", "real numeric"),
        ("nonnumeric", "real numeric"),
    ],
)
def test_cross_fit_rejects_malformed_held_out_effects(kind: str, message: str) -> None:
    """A scalar score must never silently broadcast across a held-out fold."""
    x, treatment, outcome = _constant_outcome_by_arm()

    with pytest.raises(ValueError, match=message):
        cross_fit_uplift_oof(
            lambda: _ControlledUpliftModel(kind),
            x,
            treatment,
            outcome,
            n_splits=3,
        )


@pytest.mark.parametrize("learner_type", [XLearner, DRLearner])
@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("scalar", "one prediction per row"),
        ("column", "one prediction per row"),
        ("short", "one prediction per row"),
        ("nan", "finite"),
        ("infinite", "finite"),
        ("complex", "real numeric"),
        ("nonnumeric", "real numeric"),
    ],
)
def test_effect_learners_reject_malformed_regression_scores(
    learner_type: type[XLearner] | type[DRLearner], kind: str, message: str
) -> None:
    """Policy-facing uplift estimates must be finite and aligned with each row."""
    x, treatment, outcome = _constant_outcome_by_arm()
    spec = FeatureSpec(numerical=["signal"], categorical=[], excluded=[])
    learner = learner_type(
        spec=spec,
        base_model=DummyClassifier(),
        effect_model=_ControlledEffectRegressor(kind),
        random_state=1,
    ).fit(x, treatment, outcome)

    with pytest.raises(ValueError, match=message):
        learner.predict_uplift(x)


@pytest.mark.parametrize("learner_type", [XLearner, DRLearner])
def test_effect_learners_do_not_silently_clip_extrapolated_scores(
    learner_type: type[XLearner] | type[DRLearner],
) -> None:
    """Second-stage effect regression is unconstrained, so retain its estimates."""
    x, treatment, outcome = _constant_outcome_by_arm()
    spec = FeatureSpec(numerical=["signal"], categorical=[], excluded=[])
    learner = learner_type(
        spec=spec,
        base_model=DummyClassifier(),
        effect_model=_ControlledEffectRegressor("extrapolated"),
        random_state=1,
    ).fit(x, treatment, outcome)

    np.testing.assert_allclose(learner.predict_uplift(x), 1.5)
