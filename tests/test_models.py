from __future__ import annotations

from sklearn.model_selection import train_test_split

from uplift_marketing.features import infer_feature_spec, split_features_treatment_outcome
from uplift_marketing.models import SLearner, TLearner, XLearner


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
    import numpy as np

    uplift_explicit = model.predict_uplift(x_test, propensity=np.full(len(x_test), 0.5))
    assert np.allclose(uplift, uplift_explicit)
