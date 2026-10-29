"""Uplift scoring, bins, and segment utilities."""

from __future__ import annotations

from math import sqrt
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


def _wilson_score_bounds(proportion: float, n: int, z: float) -> tuple[float, float]:
    """Wilson score limits for a binomial proportion at a supplied z-score.

    The score limits remain informative when the observed proportion is 0 or
    1, unlike the corresponding plug-in Wald standard error.
    """
    z_squared = z * z
    denominator = 1.0 + z_squared / n
    center = (proportion + z_squared / (2.0 * n)) / denominator
    radius = (z / denominator) * sqrt(
        proportion * (1.0 - proportion) / n + z_squared / (4.0 * n * n)
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _newcombe_difference_bounds(
    p_t: float, p_c: float, n_t: int, n_c: int, z: float
) -> tuple[float, float]:
    """Newcombe's Wilson-based interval for two independent proportions.

    Combine each arm's lower and upper Wilson score limits using the
    square-root-of-squared-distances method (Newcombe, 1998, method 10).
    """
    treated_low, treated_high = _wilson_score_bounds(p_t, n_t, z)
    control_low, control_high = _wilson_score_bounds(p_c, n_c, z)
    difference = p_t - p_c
    lower = difference - sqrt((p_t - treated_low) ** 2 + (control_high - p_c) ** 2)
    upper = difference + sqrt((treated_high - p_t) ** 2 + (p_c - control_low) ** 2)
    return max(-1.0, lower), min(1.0, upper)


def subgroup_uplift_with_ci(
    data: pd.DataFrame,
    *,
    group_col: str,
    treatment_col: str = "treatment",
    outcome_col: str = "conversion",
    confidence: float = 0.95,
    min_arm_size: int = 30,
) -> pd.DataFrame:
    """Subgroup-level observed uplift with a Newcombe-Wilson confidence interval.

    For each unique value of ``group_col``, returns the within-subgroup
    treated-control conversion-rate difference. The interval combines Wilson
    score limits for the two independent binomial proportions (Newcombe, 1998,
    method 10). It is bounded by [-1, 1] and does not collapse to zero width
    when conversion rates happen to be 0 or 1.

    The CI is set to ``NaN`` for subgroups whose treated or control arm has
    fewer than ``min_arm_size`` rows, retaining the reporting threshold for
    thin segments. This binomial interval requires a binary 0/1 outcome.
    For continuous outcomes, use :func:`uplift_by_group` for the descriptive
    estimate and a suitable continuous-outcome interval method.

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
            ci_low, ci_high = _newcombe_difference_bounds(p_t, p_c, n_t, n_c, z)
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
