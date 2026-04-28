"""Treatment/control diagnostics for randomized campaign data."""

from __future__ import annotations

import pandas as pd


def treatment_balance_table(data: pd.DataFrame, *, treatment_col: str = "treatment") -> pd.DataFrame:
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
