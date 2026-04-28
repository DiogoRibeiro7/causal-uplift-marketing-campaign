from __future__ import annotations

from uplift_marketing.treatment import naive_average_treatment_effect, outcome_by_treatment, treatment_balance_table


def test_treatment_balance_table_counts_rows(synthetic_data) -> None:  # type: ignore[no-untyped-def]
    table = treatment_balance_table(synthetic_data)
    assert int(table["n"].sum()) == len(synthetic_data)


def test_outcome_by_treatment_has_two_arms(synthetic_data) -> None:  # type: ignore[no-untyped-def]
    summary = outcome_by_treatment(synthetic_data)
    assert set(summary["treatment"]) == {0, 1}
    assert summary["outcome_rate"].between(0, 1).all()


def test_naive_average_treatment_effect_returns_float(synthetic_data) -> None:  # type: ignore[no-untyped-def]
    ate = naive_average_treatment_effect(synthetic_data)
    assert isinstance(ate, float)
