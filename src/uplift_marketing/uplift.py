"""Uplift scoring, bins, and segment utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def _validated_probability_vector(
    values: NDArray[Any], *, name: str, expected_length: int
) -> NDArray[np.float64]:
    """Validate one row-aligned vector of potential-outcome probabilities.

    The scoring boundary may also receive predictions from user-supplied
    estimators, so it cannot assume the upstream model checked their range.
    """

    try:
        probabilities = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric probabilities") from exc

    if probabilities.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if len(probabilities) != expected_length:
        raise ValueError(f"{name} must have the same length as data")
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{name} probabilities must be finite")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError(f"{name} probabilities must be in [0, 1]")

    return probabilities


def attach_uplift_scores(
    data: pd.DataFrame,
    *,
    p_control: NDArray[Any],
    p_treated: NDArray[Any],
    uplift_col: str = "uplift_score",
) -> pd.DataFrame:
    """Return data with validated potential-outcome probabilities and uplift.

    Both probability inputs must be finite, one-dimensional, in [0, 1], and
    aligned by row position with ``data``. Uplift is the treated probability
    minus the control probability; negative effects remain negative.
    """

    if uplift_col in {"p_control", "p_treated"}:
        raise ValueError("uplift_col must differ from p_control and p_treated")

    control_prob = _validated_probability_vector(
        p_control, name="p_control", expected_length=len(data)
    )
    treated_prob = _validated_probability_vector(
        p_treated, name="p_treated", expected_length=len(data)
    )

    scored = data.copy()
    scored["p_control"] = control_prob
    scored["p_treated"] = treated_prob
    scored[uplift_col] = scored["p_treated"] - scored["p_control"]
    return scored


def make_uplift_deciles(
    data: pd.DataFrame, *, uplift_col: str = "uplift_score", n_deciles: int = 10
) -> pd.Series:
    """Create descending score-quantile groups without splitting tied scores.

    The highest predicted scores receive group 1. When many customers share
    the same score, fewer than ``n_deciles`` groups may be occupied; assigning
    them artificial separate deciles would imply ranking power that the model
    does not have. Labels for occupied groups remain consecutive from 1.
    """

    if uplift_col not in data:
        raise ValueError(f"Uplift column not found: {uplift_col}")
    if not isinstance(n_deciles, (int, np.integer)) or n_deciles < 2:
        raise ValueError("n_deciles must be >= 2")
    if data.empty:
        raise ValueError("data must not be empty")

    scores = data[uplift_col].to_numpy(dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("uplift scores must contain only finite values")

    # Quantiles of the score itself keep equal predictions in one bin.
    # Sparse score distributions may skip quantile edges, so dense ranking
    # renumbers only the occupied groups starting with the highest scores.
    numeric_scores = pd.Series(scores, index=data.index, name=uplift_col)
    bins = pd.qcut(numeric_scores, q=n_deciles, labels=False, duplicates="drop")
    if bins.isna().all():
        # A constant model has one indistinguishable group, not N fake bins.
        return pd.Series(1, index=data.index, name=uplift_col, dtype=int)
    return bins.rank(method="dense", ascending=False).astype(int)


def _validated_group_data(
    data: pd.DataFrame,
    *,
    group_col: str,
    treatment_col: str,
    outcome_col: str,
    binary_outcome: bool,
) -> pd.DataFrame:
    """Validate all observations before computing subgroup treatment gaps.

    Descriptive group means support finite continuous outcomes. Confidence
    intervals using Bernoulli variance additionally require binary outcomes.
    Single-arm groups remain valid inputs and yield unidentified estimates.
    """
    required = (group_col, treatment_col, outcome_col)
    missing = [column for column in required if column not in data]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if data.empty:
        raise ValueError("data must not be empty")
    if not data.columns.is_unique:
        raise ValueError("data columns must be unique")
    if data[group_col].isna().any():
        raise ValueError(f"group column {group_col!r} must not contain missing values")

    try:
        treatment = data[treatment_col].to_numpy(dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("treatment must contain only binary 0/1 values") from exc
    if not np.isfinite(treatment).all() or not np.isin(treatment, (0.0, 1.0)).all():
        raise ValueError("treatment must contain only binary 0/1 values")

    try:
        outcomes = data[outcome_col].to_numpy(dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("outcome must contain finite numeric values") from exc
    if not np.isfinite(outcomes).all():
        raise ValueError("outcome must contain only finite values")
    if binary_outcome and not np.isin(outcomes, (0.0, 1.0)).all():
        raise ValueError("outcome must contain only binary 0/1 values for Bernoulli CIs")

    # Normalize numeric-string or boolean assignments so group comparisons
    # retain all treatment/control observations instead of silently omitting them.
    validated = data.copy()
    validated[treatment_col] = np.asarray(treatment, dtype=np.int64)
    validated[outcome_col] = outcomes
    return validated


def uplift_by_group(
    data: pd.DataFrame,
    *,
    group_col: str,
    treatment_col: str = "treatment",
    outcome_col: str = "conversion",
) -> pd.DataFrame:
    """Estimate per-group treated-control means for finite numeric outcomes."""

    data = _validated_group_data(
        data,
        group_col=group_col,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        binary_outcome=False,
    )

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
    unreliable there. This Bernoulli-based interval requires a binary 0/1
    outcome. For continuous outcomes, use :func:`uplift_by_group` for the
    descriptive estimate and a suitable continuous-outcome interval method.

    This is the per-segment counterpart to the marginal CIs in
    :func:`uplift_marketing.evaluation.bootstrap_policy_value`. It is the
    natural lens for fairness checks and business-segment review: where is
    the campaign actually moving customers, and where is the signal too thin
    to act on?
    """

    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    if (
        isinstance(min_arm_size, (bool, np.bool_))
        or not isinstance(min_arm_size, (int, np.integer))
        or min_arm_size < 1
    ):
        raise ValueError("min_arm_size must be a positive integer")
    data = _validated_group_data(
        data,
        group_col=group_col,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        binary_outcome=True,
    )

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
