from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

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


def test_curve_areas_include_zero_contact_origin() -> None:
    # Trapezoids must cover [0, 1], including the first [0, 0.5] segment.
    curve = pd.DataFrame(
        {"population_fraction": [0.5, 1.0], "incremental_gain": [2.0, 4.0]}
    )

    assert area_under_qini(curve) == pytest.approx(2.0)
    assert area_under_uplift(curve) == pytest.approx(2.0)


def test_tied_scores_have_row_order_invariant_qini_and_uplift_curves() -> None:
    outcome = np.array([1, 0, 0, 1, 1, 0], dtype=int)
    treatment = np.array([1, 1, 0, 0, 1, 0], dtype=int)
    equal_scores = np.zeros(len(outcome), dtype=float)

    for metric in (qini_curve, uplift_curve):
        original = metric(outcome, treatment, equal_scores)
        reversed_rows = metric(outcome[::-1], treatment[::-1], equal_scores)

        np.testing.assert_allclose(original["incremental_gain"], reversed_rows["incremental_gain"])
        # A constant ranking cannot distinguish customers: the neutral curve
        # connects the no-contact point to the observed full-sample effect.
        expected = np.linspace(0.0, float(original["incremental_gain"].iloc[-1]), 7)[1:]
        np.testing.assert_allclose(original["incremental_gain"], expected)


def test_constant_scores_have_zero_normalized_qini() -> None:
    outcome = np.array([1, 0, 0, 1, 1, 0], dtype=int)
    treatment = np.array([1, 1, 0, 0, 1, 0], dtype=int)

    result = normalized_qini(outcome, treatment, np.zeros(len(outcome)))

    assert np.isfinite(result["normalized"])
    assert result["model_auc"] == pytest.approx(result["random_auc"])
    assert result["normalized"] == pytest.approx(0.0)


def test_qini_reference_is_not_a_strict_upper_bound() -> None:
    # With severe arm imbalance, prefix-dependent scaling means the
    # signed-outcome reference need not maximise Qini area.
    treatment = np.array([1, 0, 0, 0, 0, 0], dtype=int)
    outcome = np.array([0, 1, 1, 0, 0, 0], dtype=int)
    scores = np.array([1.0, 6.0, 5.0, 4.0, 3.0, 2.0])

    result = normalized_qini(outcome, treatment, scores)

    assert result["model_auc"] > result["oracle_auc"]
    assert result["normalized"] > 1.0  # Documented, never clip silently.


@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_qini_curves_reject_nonfinite_scores(invalid: float) -> None:
    outcome = np.array([1, 0, 1, 0], dtype=int)
    treatment = np.array([1, 0, 0, 1], dtype=int)
    scores = np.array([0.5, invalid, 0.2, 0.1])

    for metric in (qini_curve, uplift_curve):
        with pytest.raises(ValueError, match="finite"):
            metric(outcome, treatment, scores)


@pytest.mark.parametrize(
    "invalid_treatment",
    [
        np.array([0.4, 1.8, 0.0, 1.0]),
        np.array([0.0, 1.0, np.nan, 0.0]),
        np.array([0.0, 1.0, np.inf, 0.0]),
        np.array([-1.0, 1.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 2.0, 0.0]),
    ],
)
def test_qini_metrics_reject_nonbinary_treatment(
    invalid_treatment: NDArray[np.float64],
) -> None:
    """Fractional or non-finite assignments must not be truncated to 0/1."""
    outcome = np.array([1, 0, 1, 0], dtype=int)
    scores = np.array([0.9, 0.7, 0.6, 0.2])

    for metric in (qini_curve, uplift_curve, normalized_qini):
        with pytest.raises(ValueError, match="treatment must contain only binary 0/1"):
            metric(outcome, invalid_treatment, scores)


def test_qini_metrics_require_one_dimensional_inputs() -> None:
    """Metrics should reject matrices instead of accidentally broadcasting them."""
    outcome = np.array([1, 0, 1, 0], dtype=int)
    treatment = np.array([1, 0, 1, 0], dtype=int)
    scores = np.array([0.9, 0.7, 0.6, 0.2])

    for metric in (qini_curve, uplift_curve, normalized_qini):
        for malformed in (
            (outcome.reshape(-1, 1), treatment, scores),
            (outcome, treatment.reshape(-1, 1), scores),
            (outcome, treatment, scores.reshape(-1, 1)),
        ):
            with pytest.raises(ValueError, match="one-dimensional"):
                metric(*malformed)


def test_qini_metrics_accept_binary_booleans_and_continuous_outcomes() -> None:
    """Input checks preserve valid binary flags and finite monetary outcomes."""
    treatment = np.array([True, False, True, False])
    outcome = np.array([2.5, 0.0, 1.25, 0.5])
    scores = np.array([0.9, 0.7, 0.6, 0.2])

    assert np.isfinite(qini_curve(outcome, treatment, scores)["incremental_gain"]).all()
    assert np.isfinite(uplift_curve(outcome, treatment, scores)["incremental_gain"]).all()
    assert np.isfinite(normalized_qini(outcome, treatment, scores)["model_auc"])


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


def test_normalized_qini_reports_reference_adjusted_score(synthetic_data) -> None:  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(0)
    nq = normalized_qini(
        outcome=synthetic_data["conversion"].to_numpy(),
        treatment=synthetic_data["treatment"].to_numpy(),
        uplift_score=rng.uniform(size=len(synthetic_data)),
    )
    assert {"model_auc", "oracle_auc", "random_auc", "normalized"} <= set(nq)
    assert nq["oracle_auc"] > nq["random_auc"]
    assert nq["normalized"] == pytest.approx(
        (nq["model_auc"] - nq["random_auc"])
        / (nq["oracle_auc"] - nq["random_auc"])
    )


def test_curve_area_accepts_explicit_zero_origin() -> None:
    curve = pd.DataFrame(
        {"population_fraction": [0.0, 0.5, 1.0], "incremental_gain": [0.0, 2.0, 4.0]}
    )
    assert area_under_qini(curve) == pytest.approx(2.0)
    assert area_under_uplift(curve) == pytest.approx(2.0)


def test_curve_area_rejects_missing_or_invalid_points() -> None:
    with pytest.raises(ValueError, match="finite"):
        area_under_qini(pd.DataFrame(columns=["population_fraction", "incremental_gain"]))
    with pytest.raises(ValueError, match="strictly increasing"):
        area_under_uplift(
            pd.DataFrame(
                {"population_fraction": [0.5, 0.2], "incremental_gain": [1.0, 2.0]}
            )
        )


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
