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


def subgroup_uplift_with_ci(
    data: pd.DataFrame,
    *,
    group_col: str,
    treatment_col: str = "treatment",
    outcome_col: str = "conversion",
    confidence: float = 0.95,
    min_arm_size: int = 30,
) -> pd.DataFrame:
    """Subgroup-level observed uplift with a normal-approximation CI.

    For each unique value of ``group_col``, returns the within-subgroup
    treated-control conversion-rate difference and a confidence interval
    based on the standard error

    .. math::
        \\text{SE} = \\sqrt{p_t(1 - p_t) / n_t + p_c(1 - p_c) / n_c}.

    The CI is set to ``NaN`` for subgroups whose treated or control arm has
    fewer than ``min_arm_size`` rows, since the normal approximation is
    unreliable there.

    This is the per-segment counterpart to the marginal CIs in
    :func:`uplift_marketing.evaluation.bootstrap_policy_value`. It is the
    natural lens for fairness checks and business-segment review: where is
    the campaign actually moving customers, and where is the signal too thin
    to act on?
    """

    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    missing = [c for c in (group_col, treatment_col, outcome_col) if c not in data]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    from math import sqrt
    from scipy.stats import norm

    z = float(norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    rows: list[dict[str, float | int | str]] = []
    for group_value, group_data in data.groupby(group_col, observed=True):
        treated = group_data[group_data[treatment_col] == 1]
        control = group_data[group_data[treatment_col] == 0]
        n_t = len(treated)
        n_c = len(control)
        if n_t == 0 or n_c == 0:
            rows.append(
                {
                    group_col: group_value,
                    "n": len(group_data),
                    "treated_n": n_t,
                    "control_n": n_c,
                    "observed_uplift": float("nan"),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                }
            )
            continue
        p_t = float(treated[outcome_col].mean())
        p_c = float(control[outcome_col].mean())
        diff = p_t - p_c
        if n_t < min_arm_size or n_c < min_arm_size:
            ci_low = ci_high = float("nan")
        else:
            se = sqrt(p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c)
            ci_low = diff - z * se
            ci_high = diff + z * se
        rows.append(
            {
                group_col: group_value,
                "n": len(group_data),
                "treated_n": n_t,
                "control_n": n_c,
                "treated_rate": p_t,
                "control_rate": p_c,
                "observed_uplift": diff,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("observed_uplift", ascending=False)
        .reset_index(drop=True)
    )
