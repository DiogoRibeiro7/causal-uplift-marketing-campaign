"""Evaluation metrics for uplift modelling and campaign policy ranking."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_auc_score


def response_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Return ROC-AUC for a standard response model."""

    if len(np.unique(y_true)) < 2:
        raise ValueError("ROC-AUC requires both positive and negative labels")
    return float(roc_auc_score(y_true, y_score))


def qini_curve(
    outcome: np.ndarray,
    treatment: np.ndarray,
    uplift_score: np.ndarray,
) -> pd.DataFrame:
    """Compute a simple Qini-style cumulative uplift curve.

    Customers are sorted by predicted uplift from high to low. At each prefix,
    the curve estimates incremental conversions as:

    treated_conversions - control_conversions * treated_count / control_count

    This implementation is intentionally transparent and suitable for teaching.
    """

    outcome = np.asarray(outcome).astype(float)
    treatment = np.asarray(treatment).astype(int)
    uplift_score = np.asarray(uplift_score).astype(float)

    if not (len(outcome) == len(treatment) == len(uplift_score)):
        raise ValueError("outcome, treatment, and uplift_score must have the same length")
    if set(np.unique(treatment)) != {0, 1}:
        raise ValueError("Both treatment arms 0 and 1 are required")

    order = np.argsort(-uplift_score)
    y_sorted = outcome[order]
    t_sorted = treatment[order]

    treated_cum = np.cumsum(t_sorted == 1)
    control_cum = np.cumsum(t_sorted == 0)
    treated_positive_cum = np.cumsum((t_sorted == 1) * y_sorted)
    control_positive_cum = np.cumsum((t_sorted == 0) * y_sorted)

    # Avoid division by zero at very early prefixes.
    safe_control = np.maximum(control_cum, 1)
    incremental_gain = treated_positive_cum - control_positive_cum * treated_cum / safe_control

    population_fraction = np.arange(1, len(outcome) + 1) / len(outcome)
    return pd.DataFrame(
        {
            "population_fraction": population_fraction,
            "treated_count": treated_cum,
            "control_count": control_cum,
            "incremental_gain": incremental_gain,
        }
    )


def area_under_qini(curve: pd.DataFrame) -> float:
    """Approximate the area under a Qini-style curve."""

    required = {"population_fraction", "incremental_gain"}
    missing = required - set(curve.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    return float(auc(curve["population_fraction"], curve["incremental_gain"]))


def policy_value_at_fraction(
    data: pd.DataFrame,
    *,
    uplift_col: str = "uplift_score",
    treatment_col: str = "treatment",
    outcome_col: str = "conversion",
    fraction: float = 0.2,
) -> dict[str, float]:
    """Estimate observed campaign value for the top uplift fraction.

    The result is a descriptive policy diagnostic based on observed outcomes in
    treated and control users inside the selected segment.
    """

    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in the interval (0, 1]")

    missing = [column for column in [uplift_col, treatment_col, outcome_col] if column not in data]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    n_selected = max(1, int(np.ceil(len(data) * fraction)))
    selected = data.sort_values(uplift_col, ascending=False).head(n_selected)
    treated = selected[selected[treatment_col] == 1]
    control = selected[selected[treatment_col] == 0]

    treated_rate = float(treated[outcome_col].mean()) if len(treated) else np.nan
    control_rate = float(control[outcome_col].mean()) if len(control) else np.nan

    return {
        "fraction": fraction,
        "selected_n": float(len(selected)),
        "treated_rate": treated_rate,
        "control_rate": control_rate,
        "observed_uplift": treated_rate - control_rate,
    }
