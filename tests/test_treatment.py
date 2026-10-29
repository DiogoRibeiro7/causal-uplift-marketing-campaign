from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from uplift_marketing.treatment import (
    RandomizationReport,
    naive_average_treatment_effect,
    outcome_by_treatment,
    treatment_balance_table,
    validate_randomization,
)


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


def test_validate_randomization_passes_for_balanced_data(synthetic_data) -> None:  # type: ignore[no-untyped-def]
    report = validate_randomization(synthetic_data, extra_excluded=["customer_id", "visit"])
    assert isinstance(report, RandomizationReport)
    # The synthetic generator splits 50/50, so the SRM test should be very mild.
    assert report.sample_ratio_pvalue > 0.1
    assert report.sample_ratio_chi2 >= 0
    # Synthetic covariates are independent of treatment, so |SMD| should be small
    # at a 500-row fixture.
    assert (report.balance["smd"].abs() < 0.30).all()
    assert report.null_share.empty


def test_validate_randomization_flags_sample_ratio_mismatch() -> None:
    # Build a deliberately imbalanced dataset: 90% treated.
    rng = np.random.default_rng(0)
    n = 1000
    treatment = (rng.uniform(size=n) < 0.9).astype(int)
    conversion = rng.binomial(1, 0.1, size=n)
    feature = rng.normal(size=n)
    data = pd.DataFrame(
        {"treatment": treatment, "conversion": conversion, "feature": feature}
    )
    report = validate_randomization(data, expected_treated_share=0.5)
    assert report.sample_ratio_pvalue < 1e-6
    assert report.observed_treated_share > 0.85


def test_validate_randomization_flags_imbalanced_covariate() -> None:
    rng = np.random.default_rng(1)
    n = 2000
    treatment = (rng.uniform(size=n) < 0.5).astype(int)
    # Bake an imbalance into a feature: treated rows are shifted by 1 sigma.
    feature = rng.normal(size=n) + treatment.astype(float)
    data = pd.DataFrame(
        {
            "treatment": treatment,
            "conversion": rng.binomial(1, 0.1, size=n),
            "feature": feature,
        }
    )
    report = validate_randomization(data, smd_threshold=0.10)
    flagged = set(report.imbalanced_features["feature"])
    assert "feature" in flagged


def test_validate_randomization_rejects_non_binary_treatment() -> None:
    data = pd.DataFrame(
        {"treatment": [0, 1, 2], "conversion": [0, 0, 1], "feature": [1.0, 2.0, 3.0]}
    )
    with pytest.raises(ValueError, match="binary 0/1"):
        validate_randomization(data)


def test_validate_randomization_rejects_mixed_nonbinary_labels() -> None:
    """Invalid mixed-type labels should raise a descriptive validation error."""
    data = pd.DataFrame({"treatment": [0, "treated"], "conversion": [0, 1]})

    with pytest.raises(ValueError, match="binary 0/1"):
        validate_randomization(data)


def test_validate_randomization_rejects_missing_assignments() -> None:
    """Missing treatments must never be silently counted as controls in SRM."""
    data = pd.DataFrame(
        {
            "treatment": [0.0, 1.0, np.nan, 1.0],
            "conversion": [0, 1, 0, 0],
            "feature": [0.2, 0.3, 0.4, 0.5],
        }
    )

    with pytest.raises(ValueError, match="missing treatment assignments"):
        validate_randomization(data)


@pytest.mark.parametrize("arm", [0, 1])
def test_validate_randomization_requires_both_arms(arm: int) -> None:
    """Balance diagnostics require treated and control observations."""
    data = pd.DataFrame(
        {"treatment": [arm] * 6, "conversion": [0, 1] * 3, "feature": range(6)}
    )

    with pytest.raises(ValueError, match="Both treatment arms"):
        validate_randomization(data)


def test_validate_randomization_rejects_empty_data() -> None:
    """An empty experiment cannot support an SRM test."""
    data = pd.DataFrame(columns=["treatment", "conversion"])

    with pytest.raises(ValueError, match="must not be empty"):
        validate_randomization(data)


@pytest.mark.parametrize("threshold", [-0.1, np.nan, np.inf])
def test_validate_randomization_rejects_invalid_smd_threshold(threshold: float) -> None:
    """Invalid thresholds must not mask covariate imbalance."""
    data = pd.DataFrame({"treatment": [0, 1], "conversion": [0, 1], "feature": [1, 2]})

    with pytest.raises(ValueError, match="smd_threshold"):
        validate_randomization(data, smd_threshold=threshold)


def test_validate_randomization_supports_no_covariates() -> None:
    """The SRM diagnostic still applies when no feature columns are present."""
    data = pd.DataFrame(
        {"treatment": [0, 1, 0, 1], "conversion": [0, 0, 1, 1]}
    )

    report = validate_randomization(data)

    assert report.observed_treated_share == pytest.approx(0.5)
    assert report.sample_ratio_chi2 == pytest.approx(0.0)
    assert report.balance.empty
    assert report.imbalanced_features.empty
    assert list(report.balance.columns) == [
        "feature",
        "mean_treated",
        "mean_control",
        "std_treated",
        "std_control",
        "smd",
    ]
