from __future__ import annotations

import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

from uplift_marketing.features import infer_feature_spec, split_features_treatment_outcome
from uplift_marketing.models import (
    DRLearner,
    SLearner,
    TLearner,
    XLearner,
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
