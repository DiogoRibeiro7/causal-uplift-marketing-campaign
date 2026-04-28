"""Uplift scoring, bins, and segment utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def attach_uplift_scores(
    data: pd.DataFrame,
    *,
    p_control: np.ndarray,
    p_treated: np.ndarray,
    uplift_col: str = "uplift_score",
) -> pd.DataFrame:
    """Return data with potential outcome probabilities and uplift scores."""

    if len(data) != len(p_control) or len(data) != len(p_treated):
        raise ValueError("data, p_control, and p_treated must have the same length")

    scored = data.copy()
    scored["p_control"] = np.asarray(p_control, dtype=float)
    scored["p_treated"] = np.asarray(p_treated, dtype=float)
    scored[uplift_col] = scored["p_treated"] - scored["p_control"]
    return scored


def make_uplift_deciles(data: pd.DataFrame, *, uplift_col: str = "uplift_score") -> pd.Series:
    """Create descending uplift deciles where decile 1 has highest uplift."""

    if uplift_col not in data:
        raise ValueError(f"Uplift column not found: {uplift_col}")

    ranks = data[uplift_col].rank(method="first", ascending=False)
    deciles = pd.qcut(ranks, q=10, labels=False, duplicates="drop") + 1
    return deciles.astype(int)


def uplift_by_group(
    data: pd.DataFrame,
    *,
    group_col: str,
    treatment_col: str = "treatment",
    outcome_col: str = "conversion",
) -> pd.DataFrame:
    """Estimate observed uplift by group using treated-control rate difference."""

    missing = [column for column in [group_col, treatment_col, outcome_col] if column not in data]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    rows: list[dict[str, float | int | str]] = []
    for group_value, group_data in data.groupby(group_col, observed=True):
        treated = group_data[group_data[treatment_col] == 1]
        control = group_data[group_data[treatment_col] == 0]
        treated_rate = float(treated[outcome_col].mean()) if len(treated) else np.nan
        control_rate = float(control[outcome_col].mean()) if len(control) else np.nan
        rows.append(
            {
                group_col: group_value,
                "n": len(group_data),
                "treated_n": len(treated),
                "control_n": len(control),
                "treated_rate": treated_rate,
                "control_rate": control_rate,
                "observed_uplift": treated_rate - control_rate,
            }
        )
    return pd.DataFrame(rows).sort_values(group_col)
