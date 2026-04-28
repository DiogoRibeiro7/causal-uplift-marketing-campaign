from __future__ import annotations

from uplift_marketing.evaluation import area_under_qini, policy_value_at_fraction, qini_curve


def test_qini_curve_has_expected_columns(synthetic_data) -> None:  # type: ignore[no-untyped-def]
    curve = qini_curve(
        outcome=synthetic_data["conversion"].to_numpy(),
        treatment=synthetic_data["treatment"].to_numpy(),
        uplift_score=synthetic_data["discount_affinity"].to_numpy(),
    )
    assert {"population_fraction", "incremental_gain"}.issubset(curve.columns)
    assert len(curve) == len(synthetic_data)


def test_area_under_qini_returns_float(synthetic_data) -> None:  # type: ignore[no-untyped-def]
    curve = qini_curve(
        outcome=synthetic_data["conversion"].to_numpy(),
        treatment=synthetic_data["treatment"].to_numpy(),
        uplift_score=synthetic_data["discount_affinity"].to_numpy(),
    )
    value = area_under_qini(curve)
    assert isinstance(value, float)


def test_policy_value_at_fraction_returns_summary(synthetic_data) -> None:  # type: ignore[no-untyped-def]
    data = synthetic_data.assign(uplift_score=synthetic_data["discount_affinity"])
    summary = policy_value_at_fraction(data, fraction=0.2)
    assert summary["fraction"] == 0.2
    assert summary["selected_n"] > 0
