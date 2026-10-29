from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
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
    # Default propensity 0.5 should match an explicit per-row propensity vector.
    uplift_explicit = model.predict_uplift(x_test, propensity=np.full(len(x_test), 0.5))
    assert np.allclose(uplift, uplift_explicit)


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
