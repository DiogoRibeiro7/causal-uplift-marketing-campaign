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
