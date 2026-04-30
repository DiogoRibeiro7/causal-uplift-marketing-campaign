from __future__ import annotations

import pytest

from uplift_marketing.features import (
    build_preprocessor,
    honest_train_test_split,
    infer_feature_spec,
    split_features_treatment_outcome,
)


def test_infer_feature_spec_excludes_treatment_and_outcome(synthetic_data) -> None:  # type: ignore[no-untyped-def]
    spec = infer_feature_spec(synthetic_data, extra_excluded=["customer_id", "visit"])
    assert "treatment" not in spec.feature_columns
    assert "conversion" not in spec.feature_columns
    assert "customer_id" not in spec.feature_columns
    assert spec.feature_columns


def test_build_preprocessor_fits(synthetic_data) -> None:  # type: ignore[no-untyped-def]
    spec = infer_feature_spec(synthetic_data, extra_excluded=["customer_id", "visit"])
    x, _, _ = split_features_treatment_outcome(synthetic_data, spec)
    transformer = build_preprocessor(spec)
    matrix = transformer.fit_transform(x)
    assert matrix.shape[0] == len(synthetic_data)


def test_honest_train_test_split_preserves_arm_and_outcome_balance(  # type: ignore[no-untyped-def]
    synthetic_data,
) -> None:
    spec = infer_feature_spec(synthetic_data, extra_excluded=["customer_id", "visit"])
    x, treatment, y = split_features_treatment_outcome(synthetic_data, spec)
    x_tr, x_te, t_tr, t_te, y_tr, y_te = honest_train_test_split(
        x, treatment, y,
        treatment=treatment,
        outcome=y,
        test_size=0.30,
        random_state=0,
    )
    assert len(x_tr) + len(x_te) == len(x)
    # Both arms must appear on each side of the split.
    assert set(t_tr.unique()) == {0, 1}
    assert set(t_te.unique()) == {0, 1}
    # Treated share preserved within ~1pp on a 500-row fixture.
    assert abs(t_tr.mean() - t_te.mean()) < 0.02


def test_honest_train_test_split_rejects_non_binary_treatment(synthetic_data) -> None:  # type: ignore[no-untyped-def]
    bad = synthetic_data["treatment"].copy()
    bad.iloc[0] = 2
    with pytest.raises(ValueError, match="binary"):
        honest_train_test_split(synthetic_data, treatment=bad, test_size=0.3, random_state=0)
