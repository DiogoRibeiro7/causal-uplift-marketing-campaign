"""Treatment/control diagnostics for randomized campaign data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def treatment_balance_table(
    data: pd.DataFrame, *, treatment_col: str = "treatment"
) -> pd.DataFrame:
    """Return treatment/control counts and proportions."""

    if treatment_col not in data:
        raise ValueError(f"Treatment column not found: {treatment_col}")

    counts = data[treatment_col].value_counts(dropna=False).rename("n").to_frame()
    counts["share"] = counts["n"] / counts["n"].sum()
    counts.index.name = treatment_col
    return counts.reset_index()


def outcome_by_treatment(
    data: pd.DataFrame,
    *,
    treatment_col: str = "treatment",
    outcome_col: str = "conversion",
) -> pd.DataFrame:
    """Calculate observed outcome rates by treatment arm."""

    missing = [column for column in [treatment_col, outcome_col] if column not in data]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    summary = (
        data.groupby(treatment_col, observed=True)[outcome_col]
        .agg(n="size", outcome_rate="mean", positives="sum")
        .reset_index()
    )
    return summary


def naive_average_treatment_effect(
    data: pd.DataFrame,
    *,
    treatment_col: str = "treatment",
    outcome_col: str = "conversion",
) -> float:
    """Estimate the naive difference in outcome rates between treated and control.

    This is appropriate only under randomized treatment assignment and no major
    data quality problems. It is provided as a simple diagnostic, not as a full
    causal proof.
    """

    rates = outcome_by_treatment(data, treatment_col=treatment_col, outcome_col=outcome_col)
    if set(rates[treatment_col]) != {0, 1}:
        raise ValueError("Both treatment arms 0 and 1 must be present")

    treated_rate = float(rates.loc[rates[treatment_col] == 1, "outcome_rate"].iloc[0])
    control_rate = float(rates.loc[rates[treatment_col] == 0, "outcome_rate"].iloc[0])
    return treated_rate - control_rate


@dataclass(frozen=True)
class RandomizationReport:
    """Diagnostic summary of a randomized treatment/control assignment.

    Attributes
    ----------
    expected_treated_share:
        Share of treated rows the test compares against.
    observed_treated_share:
        Empirical share of treated rows.
    sample_ratio_chi2:
        Chi-square statistic for the sample-ratio mismatch test against
        `expected_treated_share`.
    sample_ratio_pvalue:
        Two-sided p-value of the sample-ratio test.
    balance:
        Per-feature standardized mean difference between treated and control,
        with the means and (sample) standard deviations used to compute it.
    imbalanced_features:
        Subset of `balance` whose ``|smd|`` exceeds `smd_threshold`.
    null_share:
        Per-column share of missing values, sorted descending. Empty if no
        column has nulls.
    """

    expected_treated_share: float
    observed_treated_share: float
    sample_ratio_chi2: float
    sample_ratio_pvalue: float
    balance: pd.DataFrame
    imbalanced_features: pd.DataFrame
    null_share: pd.Series
    smd_threshold: float


def validate_randomization(
    data: pd.DataFrame,
    *,
    treatment_col: str = "treatment",
    outcome_col: str = "conversion",
    expected_treated_share: float = 0.5,
    smd_threshold: float = 0.10,
    extra_excluded: list[str] | None = None,
) -> RandomizationReport:
    """Run standard randomization checks on a treatment/control dataset.

    Combines three diagnostics that should agree under proper randomization:

    1. **Sample ratio mismatch.** Chi-square against the expected share of
       treated rows. A small p-value indicates a logging or assignment bug
       rather than a real effect.
    2. **Covariate balance.** Standardized mean difference (SMD) per numeric
       feature; categorical features are one-hot expanded so each level gets
       its own SMD. Conventionally ``|SMD| > 0.10`` flags imbalance.
    3. **Missingness.** Share of nulls per column, since downstream feature
       inference assumes complete cases by default.
    """

    if treatment_col not in data:
        raise ValueError(f"Treatment column not found: {treatment_col}")
    if not 0 < expected_treated_share < 1:
        raise ValueError("expected_treated_share must be in (0, 1)")

    treatment = data[treatment_col]
    if set(treatment.dropna().unique()) - {0, 1}:
        raise ValueError(
            f"Treatment column {treatment_col!r} must be binary 0/1; "
            f"found values {sorted(treatment.dropna().unique())}"
        )

    n_total = len(data)
    n_treated = int(treatment.sum())
    observed_share = n_treated / n_total if n_total else float("nan")

    expected_counts = np.array(
        [n_total * (1 - expected_treated_share), n_total * expected_treated_share],
        dtype=float,
    )
    observed_counts = np.array([n_total - n_treated, n_treated], dtype=float)
    # Pearson chi-square; safer than scipy here so the package keeps zero
    # extra runtime deps.
    chi2 = float(np.sum((observed_counts - expected_counts) ** 2 / expected_counts))
    # Survival of chi-square with 1 d.o.f. closed form: erfc(sqrt(chi2/2)).
    from math import erfc, sqrt

    pvalue = float(erfc(sqrt(chi2 / 2.0)))

    excluded = {treatment_col, outcome_col, *(extra_excluded or [])}
    feature_columns = [c for c in data.columns if c not in excluded]
    feature_frame = data[feature_columns]
    numeric = feature_frame.select_dtypes(include="number")
    non_numeric = feature_frame.drop(columns=numeric.columns)
    if not non_numeric.empty:
        non_numeric = pd.get_dummies(non_numeric, drop_first=False).astype(float)
    expanded = pd.concat([numeric.astype(float), non_numeric], axis=1)

    treated_mask = treatment.eq(1)
    control_mask = treatment.eq(0)
    rows = []
    for column in expanded.columns:
        treated_values = expanded.loc[treated_mask, column]
        control_values = expanded.loc[control_mask, column]
        mt = float(treated_values.mean())
        mc = float(control_values.mean())
        vt = float(treated_values.var(ddof=1)) if len(treated_values) > 1 else 0.0
        vc = float(control_values.var(ddof=1)) if len(control_values) > 1 else 0.0
        pooled = float(np.sqrt((vt + vc) / 2.0))
        smd = (mt - mc) / pooled if pooled > 0 else float("nan")
        rows.append(
            {
                "feature": column,
                "mean_treated": mt,
                "mean_control": mc,
                "std_treated": float(np.sqrt(vt)),
                "std_control": float(np.sqrt(vc)),
                "smd": smd,
            }
        )
    balance = pd.DataFrame(rows).sort_values(
        "smd", key=lambda series: series.abs(), ascending=False
    )
    imbalanced = balance[balance["smd"].abs() > smd_threshold].copy()

    null_share = data.isna().mean()
    null_share = null_share[null_share > 0].sort_values(ascending=False)

    return RandomizationReport(
        expected_treated_share=float(expected_treated_share),
        observed_treated_share=float(observed_share),
        sample_ratio_chi2=chi2,
        sample_ratio_pvalue=pvalue,
        balance=balance.reset_index(drop=True),
        imbalanced_features=imbalanced.reset_index(drop=True),
        null_share=null_share,
        smd_threshold=float(smd_threshold),
    )
