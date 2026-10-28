from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from uplift_marketing.evaluation import (
    area_under_qini,
    area_under_uplift,
    bootstrap_decile_uplift,
    bootstrap_policy_value,
    normalized_qini,
    policy_value_at_fraction,
    qini_curve,
    uplift_curve,
)


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


def test_uplift_curve_terminates_at_total_gain(synthetic_data) -> None:  # type: ignore[no-untyped-def]
    curve = uplift_curve(
        outcome=synthetic_data["conversion"].to_numpy(),
        treatment=synthetic_data["treatment"].to_numpy(),
        uplift_score=synthetic_data["discount_affinity"].to_numpy(),
    )
    treated_rate = synthetic_data.loc[synthetic_data.treatment == 1, "conversion"].mean()
    control_rate = synthetic_data.loc[synthetic_data.treatment == 0, "conversion"].mean()
    # Full-population incremental conversions adjust for the treatment share.
    expected_end = float(len(synthetic_data) * (treated_rate - control_rate))
    assert curve["incremental_gain"].iloc[-1] == pytest.approx(expected_end, rel=1e-9)
    auuc = area_under_uplift(curve)
    assert isinstance(auuc, float)


def test_uplift_curve_does_not_confuse_arm_imbalance_with_effect() -> None:
    # Both arms have a 100% conversion rate, despite a 2:1 allocation.
    treatment = np.r_[np.ones(200, dtype=int), np.zeros(100, dtype=int)]
    outcome = np.ones(300, dtype=int)
    scores = np.linspace(1, 0, 300)

    curve = uplift_curve(outcome=outcome, treatment=treatment, uplift_score=scores)

    assert curve["incremental_gain"].iloc[-1] == pytest.approx(0.0)
    assert curve["n_treated_total"].iloc[-1] == 200
    assert curve["n_control_total"].iloc[-1] == 100


def test_uplift_curve_uses_conversion_rates_not_raw_counts() -> None:
    treatment = np.r_[np.ones(4, dtype=int), np.zeros(2, dtype=int)]
    outcome = np.array([1, 1, 1, 0, 1, 0], dtype=int)
    scores = np.arange(6, dtype=float)

    curve = uplift_curve(outcome=outcome, treatment=treatment, uplift_score=scores)

    # N * (treated conversion rate - control conversion rate) = 6 * (0.75 - 0.5).
    assert curve["incremental_gain"].iloc[-1] == pytest.approx(1.5)


def test_normalized_qini_bounded_below_oracle(synthetic_data) -> None:  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(0)
    nq = normalized_qini(
        outcome=synthetic_data["conversion"].to_numpy(),
        treatment=synthetic_data["treatment"].to_numpy(),
        uplift_score=rng.uniform(size=len(synthetic_data)),
    )
    assert {"model_auc", "oracle_auc", "random_auc", "normalized"} <= set(nq)
    # An oracle ranks at or above any other ranking; the normalized score
    # against random should be near 0 for a random ranker and at most 1
    # for the oracle.
    assert nq["oracle_auc"] >= nq["model_auc"] - 1e-9
    assert nq["normalized"] <= 1.0 + 1e-9


def test_bootstrap_policy_value_excludes_zero_for_clear_signal(synthetic_data) -> None:  # type: ignore[no-untyped-def]
    # Build a perfect uplift score so the top-fraction CI clearly excludes zero.
    data = synthetic_data.assign(
        uplift_score=(synthetic_data["treatment"] * synthetic_data["conversion"]).astype(float)
        - ((1 - synthetic_data["treatment"]) * synthetic_data["conversion"]).astype(float)
    )
    summary = bootstrap_policy_value(
        data, fraction=0.2, n_boot=200, random_state=0,
    )
    assert summary["fraction"] == 0.2
    assert summary["uplift_ci_low"] > 0
    # When economics are supplied, profit fields appear.
    summary_profit = bootstrap_policy_value(
        data,
        fraction=0.2,
        n_boot=200,
        random_state=0,
        cost_per_contact=0.5,
        value_per_conversion=10.0,
    )
    assert "profit_mean" in summary_profit
    assert "profit_ci_low" in summary_profit


def test_bootstrap_decile_uplift_returns_one_row_per_decile(synthetic_data) -> None:  # type: ignore[no-untyped-def]
    data = synthetic_data.assign(uplift_score=synthetic_data["discount_affinity"])
    table = bootstrap_decile_uplift(data, n_deciles=5, n_boot=100, random_state=0)
    assert list(table["decile"]) == [1, 2, 3, 4, 5]
    assert {"uplift_mean", "ci_low", "ci_high"}.issubset(table.columns)
    # CI bounds should bracket the mean for any populated row.
    populated = table.dropna()
    assert (populated["ci_low"] <= populated["uplift_mean"]).all()
    assert (populated["uplift_mean"] <= populated["ci_high"]).all()


def test_bootstrap_policy_value_rejects_invalid_fraction(synthetic_data) -> None:  # type: ignore[no-untyped-def]
    data = synthetic_data.assign(uplift_score=synthetic_data["discount_affinity"])
    with pytest.raises(ValueError, match="fraction"):
        bootstrap_policy_value(data, fraction=0.0)


def test_bootstrap_policy_value_skips_resamples_without_both_arms() -> None:
    # Small selected groups frequently lose an arm under resampling.
    data = pd.DataFrame(
        {
            "uplift_score": [0.99, 0.90, 0.70, 0.65, 0.30, 0.10],
            "treatment": [1, 0, 1, 0, 1, 0],
            "conversion": [1, 0, 0, 0, 1, 0],
        }
    )
    result = bootstrap_policy_value(
        data,
        fraction=0.5,
        n_boot=200,
        random_state=5,
        cost_per_contact=0.5,
        value_per_conversion=10.0,
    )

    assert 2 <= result["n_valid_boot"] < result["n_boot"]
    assert result["n_valid_boot"] + result["n_invalid_boot"] == result["n_boot"]
    for metric in (
        "uplift_mean",
        "uplift_ci_low",
        "uplift_ci_high",
        "profit_mean",
        "profit_ci_low",
        "profit_ci_high",
    ):
        assert np.isfinite(result[metric])
    # Each bootstrap sample contacts three customers at this fraction.
    assert result["profit_mean"] == pytest.approx(result["uplift_mean"] * 3 * 10.0 - 1.5)


@pytest.mark.parametrize(
    ("cost_per_contact", "value_per_conversion"),
    [(0.5, None), (None, 10.0)],
)
def test_bootstrap_policy_value_requires_complete_economics(
    synthetic_data: pd.DataFrame,
    cost_per_contact: float | None,
    value_per_conversion: float | None,
) -> None:
    data = synthetic_data.assign(uplift_score=synthetic_data["discount_affinity"])

    with pytest.raises(ValueError, match="must be provided together"):
        bootstrap_policy_value(
            data,
            cost_per_contact=cost_per_contact,
            value_per_conversion=value_per_conversion,
            n_boot=10,
        )


@pytest.mark.parametrize(
    ("cost_per_contact", "value_per_conversion", "message"),
    [
        (-1.0, 10.0, "cost_per_contact"),
        (np.inf, 10.0, "cost_per_contact"),
        (0.5, 0.0, "value_per_conversion"),
        (0.5, np.nan, "value_per_conversion"),
    ],
)
def test_bootstrap_policy_value_rejects_invalid_economics(
    synthetic_data: pd.DataFrame,
    cost_per_contact: float,
    value_per_conversion: float,
    message: str,
) -> None:
    data = synthetic_data.assign(uplift_score=synthetic_data["discount_affinity"])

    with pytest.raises(ValueError, match=message):
        bootstrap_policy_value(
            data,
            cost_per_contact=cost_per_contact,
            value_per_conversion=value_per_conversion,
            n_boot=10,
        )


def test_bootstrap_policy_value_rejects_unidentified_top_segment() -> None:
    # The overall experiment has both arms, but only treated users are in top-k.
    data = pd.DataFrame(
        {
            "uplift_score": [0.9, 0.8, 0.7, 0.6],
            "treatment": [1, 1, 0, 0],
            "conversion": [1, 0, 1, 0],
        }
    )

    with pytest.raises(ValueError, match="requires both treatment arms"):
        bootstrap_policy_value(data, fraction=0.5, n_boot=20)


def test_bootstrap_policy_value_rejects_insufficient_valid_resamples() -> None:
    # With two customers, seed 4 produces two single-arm resamples.
    data = pd.DataFrame(
        {
            "uplift_score": [0.9, 0.1],
            "treatment": [1, 0],
            "conversion": [1, 0],
        }
    )

    with pytest.raises(ValueError, match="fewer than two bootstrap resamples"):
        bootstrap_policy_value(data, fraction=1.0, n_boot=2, random_state=4)


def test_bootstrap_policy_value_rejects_empty_data() -> None:
    data = pd.DataFrame(columns=["uplift_score", "treatment", "conversion"])

    with pytest.raises(ValueError, match="data must not be empty"):
        bootstrap_policy_value(data, n_boot=20)
