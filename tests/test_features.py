from __future__ import annotations

from uplift_marketing.features import build_preprocessor, infer_feature_spec, split_features_treatment_outcome


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
