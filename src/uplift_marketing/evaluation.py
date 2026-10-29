"""Evaluation metrics for uplift modelling and campaign policy ranking."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import auc, roc_auc_score


def _validate_curve_inputs(
    outcome: NDArray[Any], treatment: NDArray[Any], uplift_score: NDArray[Any]
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
    """Validate aligned one-dimensional arrays before coercing treatment labels.

    Outcomes may be continuous, but treatment must be a fully observed 0/1
    indicator. Casting treatment to integers before checking its values would
    silently turn fractional assignments into seemingly valid experiment arms.
    """
    raw_outcome = np.asarray(outcome)
    raw_treatment = np.asarray(treatment)
    raw_scores = np.asarray(uplift_score)

    if any(array.ndim != 1 for array in (raw_outcome, raw_treatment, raw_scores)):
        raise ValueError("outcome, treatment, and uplift_score must be one-dimensional")
    if not (len(raw_outcome) == len(raw_treatment) == len(raw_scores)):
        raise ValueError("outcome, treatment, and uplift_score must have the same length")

    try:
        numeric_treatment = np.asarray(raw_treatment, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("treatment must contain only binary 0/1 values") from exc
    if (
        not np.isfinite(numeric_treatment).all()
        or not np.isin(numeric_treatment, (0.0, 1.0)).all()
    ):
        raise ValueError("treatment must contain only binary 0/1 values")
    if set(np.unique(numeric_treatment)) != {0.0, 1.0}:
        raise ValueError("Both treatment arms 0 and 1 are required")

    numeric_outcome = np.asarray(raw_outcome, dtype=float)
    numeric_scores = np.asarray(raw_scores, dtype=float)
    if not np.isfinite(numeric_outcome).all() or not np.isfinite(numeric_scores).all():
        raise ValueError("outcome and uplift_score must contain only finite values")

    return (
        np.asarray(numeric_outcome, dtype=np.float64),
        np.asarray(numeric_treatment, dtype=np.int64),
        np.asarray(numeric_scores, dtype=np.float64),
    )


def _interpolate_tied_gains(
    sorted_scores: NDArray[Any], cumulative_gain: NDArray[Any]
) -> NDArray[Any]:
    """Interpolate gains within tied-score blocks, preserving their endpoints.

    Customers with identical predicted scores cannot be ranked relative to
    each other. Interpolating between distinct score thresholds prevents the
    original row order from creating artificial ranking performance.
    """
    n = len(sorted_scores)
    boundaries = np.r_[np.flatnonzero(np.diff(sorted_scores) != 0) + 1, n]
    return np.asarray(
        np.interp(
            np.arange(1, n + 1),
            np.r_[0, boundaries],
            np.r_[0.0, cumulative_gain[boundaries - 1]],
        ),
        dtype=float,
    )


def _area_from_zero(curve: pd.DataFrame) -> float:
    """Integrate cumulative gains from the no-contact point (0, 0)."""
    required = {"population_fraction", "incremental_gain"}
    missing = required - set(curve.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    x = curve["population_fraction"].to_numpy(dtype=float)
    y = curve["incremental_gain"].to_numpy(dtype=float)
    if not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("curve must contain finite population fractions and gains")
    if x[0] < 0 or np.any(np.diff(x) <= 0):
        raise ValueError("population_fraction must be strictly increasing from zero")
    if x[0] == 0.0:
        if y[0] != 0.0:
            raise ValueError("the no-contact point must have zero gain")
    else:
        # Existing curve producers start at 1/N, so include the first
        # trapezoid instead of integrating from that first customer onward.
        x = np.r_[0.0, x]
        y = np.r_[0.0, y]
    return float(auc(x, y))


def response_auc(y_true: NDArray[Any], y_score: NDArray[Any]) -> float:
    """Return ROC-AUC for a standard response model."""

    if len(np.unique(y_true)) < 2:
        raise ValueError("ROC-AUC requires both positive and negative labels")
    return float(roc_auc_score(y_true, y_score))


def qini_curve(
    outcome: NDArray[Any],
    treatment: NDArray[Any],
    uplift_score: NDArray[Any],
) -> pd.DataFrame:
    """Compute a simple Qini-style cumulative uplift curve.

    Customers are sorted by predicted uplift from high to low. At each prefix,
    the curve estimates incremental conversions as:

    treated_conversions - control_conversions * treated_count / control_count

    Gains within equal-score blocks are interpolated between their threshold
    endpoints: a model cannot choose an order inside a tie. The integer arm
    counts still describe one stable row order; interpolated gains within a
    tie block should not be recomputed from those intermediate counts.
    """

    outcome, treatment, uplift_score = _validate_curve_inputs(outcome, treatment, uplift_score)

    order = np.argsort(-uplift_score, kind="mergesort")
    y_sorted = outcome[order]
    t_sorted = treatment[order]
    scores_sorted = uplift_score[order]

    treated_cum = np.cumsum(t_sorted == 1)
    control_cum = np.cumsum(t_sorted == 0)
    treated_positive_cum = np.cumsum((t_sorted == 1) * y_sorted)
    control_positive_cum = np.cumsum((t_sorted == 0) * y_sorted)

    # Avoid division by zero at very early prefixes.
    safe_control = np.maximum(control_cum, 1)
    incremental_gain = treated_positive_cum - control_positive_cum * treated_cum / safe_control
    incremental_gain = _interpolate_tied_gains(scores_sorted, incremental_gain)

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
    """Area under a Qini-style curve, including the no-contact origin."""
    return _area_from_zero(curve)


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
    treated and control users inside the selected segment. Customers tied at
    the contact-budget cutoff receive equal fractional selection weights,
    avoiding arbitrary decisions based on dataframe row order. The weights
    sum to ``ceil(N * fraction)``; rates use weighted treated/control means.
    """

    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in the interval (0, 1]")

    missing = [column for column in [uplift_col, treatment_col, outcome_col] if column not in data]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if data.empty:
        raise ValueError("data must not be empty")

    n_selected = max(1, int(np.ceil(len(data) * fraction)))
    scores = data[uplift_col].to_numpy(dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("uplift scores must be finite")

    # The cut-off score identifies the last customer the budget can cover.
    # When that score is tied, uniformly share the remaining budget across
    # the entire tie block rather than using the input row order as a ranking.
    cutoff = float(np.partition(scores, len(scores) - n_selected)[len(scores) - n_selected])
    above_cutoff = scores > cutoff
    at_cutoff = scores == cutoff
    remaining = n_selected - int(np.count_nonzero(above_cutoff))
    tie_weight = remaining / int(np.count_nonzero(at_cutoff))
    weights = np.where(above_cutoff, 1.0, np.where(at_cutoff, tie_weight, 0.0))

    selected = weights > 0
    treatment = data[treatment_col].to_numpy()
    outcomes = data[outcome_col].to_numpy(dtype=float)
    treated = selected & (treatment == 1)
    control = selected & (treatment == 0)

    treated_rate = (
        float(np.average(outcomes[treated], weights=weights[treated]))
        if treated.any()
        else np.nan
    )
    control_rate = (
        float(np.average(outcomes[control], weights=weights[control]))
        if control.any()
        else np.nan
    )

    return {
        "fraction": fraction,
        "selected_n": float(n_selected),
        "treated_rate": treated_rate,
        "control_rate": control_rate,
        "observed_uplift": treated_rate - control_rate,
    }


def uplift_curve(
    outcome: NDArray[Any],
    treatment: NDArray[Any],
    uplift_score: NDArray[Any],
) -> pd.DataFrame:
    """Compute a cumulative, treatment-share-adjusted uplift curve.

    Customers are ranked by predicted uplift. For each top-k prefix, let
    ``T_pos(k)`` and ``C_pos(k)`` be its treated and control conversions,
    ``N_T`` and ``N_C`` the *full-sample* treatment-arm sizes, and ``N``
    the full sample size. The cumulative gain is

    ``N * (T_pos(k) / N_T - C_pos(k) / N_C)``.

    This inverse-treatment-share weighting corrects for unequal treatment
    allocation. At the full population, the gain is ``N`` times the
    observed treated-minus-control conversion-rate difference, measured
    in incremental conversions. It assumes random treatment assignment;
    observational data require appropriate propensity adjustment. Gains for
    tied scores are interpolated to avoid artificial within-tie ranking.
    """

    outcome, treatment, uplift_score = _validate_curve_inputs(outcome, treatment, uplift_score)

    order = np.argsort(-uplift_score, kind="mergesort")
    y_sorted = outcome[order]
    t_sorted = treatment[order]
    scores_sorted = uplift_score[order]

    treated_cum = np.cumsum(t_sorted == 1)
    control_cum = np.cumsum(t_sorted == 0)
    treated_positive_cum = np.cumsum((t_sorted == 1) * y_sorted)
    control_positive_cum = np.cumsum((t_sorted == 0) * y_sorted)

    n_t = max(int(treated_cum[-1]), 1)
    n_c = max(int(control_cum[-1]), 1)

    # Keep the *global* arm denominators fixed across all ranked prefixes.
    # Subtracting raw arm conversions would mistake allocation imbalance
    # for a treatment effect even when both arms convert at the same rate.
    n_total = len(outcome)
    incremental_gain = n_total * (
        treated_positive_cum / n_t - control_positive_cum / n_c
    )
    incremental_gain = _interpolate_tied_gains(scores_sorted, incremental_gain)
    population_fraction = np.arange(1, len(outcome) + 1) / len(outcome)

    return pd.DataFrame(
        {
            "population_fraction": population_fraction,
            "treated_count": treated_cum,
            "control_count": control_cum,
            "incremental_gain": incremental_gain,
            "n_treated_total": n_t,
            "n_control_total": n_c,
        }
    )


def area_under_uplift(curve: pd.DataFrame) -> float:
    """Area under cumulative uplift, including the no-contact origin."""
    return _area_from_zero(curve)


def normalized_qini(
    outcome: NDArray[Any],
    treatment: NDArray[Any],
    uplift_score: NDArray[Any],
) -> dict[str, float]:
    """Return a baseline-adjusted Qini score against an in-sample reference.

    Raw Qini AUC scales with population size and conversion rate, which makes
    cross-dataset comparisons hard. The reference ranking uses the realised
    signed contribution of each treated/control outcome. It is an optimistic
    *in-sample* benchmark, not a mathematical upper bound for the
    prefix-dependent Qini formula. Scores can be negative or exceed 1.0;
    a nondiscriminating constant-score model has value zero when a positive
    reference denominator exists, and random rankings have expected score
    close to zero across repeated samples.

    The dictionary returned contains:

    - ``model_auc`` — area under the Qini curve produced by ``uplift_score``;
    - ``oracle_auc`` — area under an optimistic signed-outcome ranking
      (an in-sample reference, not a guaranteed upper bound);
    - ``random_auc`` — the diagonal, no-skill baseline area. The finite-sample
      expectation for a prefix-ratio Qini curve need not equal this exactly;
    - ``normalized`` — ``(model_auc - random_auc) / (oracle_auc - random_auc)``;
      this is what to quote when comparing models across datasets.
    """

    outcome, treatment, uplift_score = _validate_curve_inputs(outcome, treatment, uplift_score)

    model_curve = qini_curve(outcome, treatment, uplift_score)
    model_auc = area_under_qini(model_curve)

    # Signed-outcome reference; sorting by additive, full-sample arm weights
    # is optimistic but does not necessarily maximise prefix-ratio Qini.
    n_t = max(int((treatment == 1).sum()), 1)
    n_c = max(int((treatment == 0).sum()), 1)
    oracle_score = np.where(
        treatment == 1,
        outcome / n_t,
        -outcome / n_c,
    )
    oracle_curve = qini_curve(outcome, treatment, oracle_score)
    oracle_auc = area_under_qini(oracle_curve)

    # The no-skill *reference* is the diagonal between (0, 0) and the final
    # gain. Its area is half the endpoint; it is not an exact finite-sample
    # random-permutation expectation for the nonlinear prefix ratio.
    final_gain = float(model_curve["incremental_gain"].iloc[-1])
    random_auc = 0.5 * final_gain

    denom = oracle_auc - random_auc
    normalized = (model_auc - random_auc) / denom if denom > 0 else float("nan")
    return {
        "model_auc": float(model_auc),
        "oracle_auc": float(oracle_auc),
        "random_auc": float(random_auc),
        "normalized": float(normalized),
    }


def bootstrap_policy_value(
    data: pd.DataFrame,
    *,
    fraction: float = 0.2,
    n_boot: int = 500,
    confidence: float = 0.95,
    uplift_col: str = "uplift_score",
    treatment_col: str = "treatment",
    outcome_col: str = "conversion",
    cost_per_contact: float | None = None,
    value_per_conversion: float | None = None,
    random_state: int = 42,
) -> dict[str, float]:
    """Bootstrap CIs for the top-k policy's observed uplift and net profit.

    Resamples the input dataframe with replacement ``n_boot`` times. On
    each resample, applies the same top-``fraction`` selection rule and
    records the within-segment treated-control gap. If ``cost_per_contact``
    and ``value_per_conversion`` are both provided, also reports the bootstrap
    distribution of realised net profit for the same selection. Resamples
    without both arms or finite outcome rates in the selected segment are
    excluded from the interval; ``n_valid_boot`` and ``n_invalid_boot`` make
    that limitation explicit.
    At least two valid resamples are required.
    """

    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in the interval (0, 1]")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    if n_boot < 2:
        raise ValueError("n_boot must be >= 2")
    if data.empty:
        raise ValueError("data must not be empty")
    if (cost_per_contact is None) != (value_per_conversion is None):
        raise ValueError("cost_per_contact and value_per_conversion must be provided together")
    if cost_per_contact is not None and (
        not np.isfinite(cost_per_contact) or cost_per_contact < 0
    ):
        raise ValueError("cost_per_contact must be finite and non-negative")
    if value_per_conversion is not None and (
        not np.isfinite(value_per_conversion) or value_per_conversion <= 0
    ):
        raise ValueError("value_per_conversion must be finite and positive")

    missing = [c for c in (uplift_col, treatment_col, outcome_col) if c not in data]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    original_policy = policy_value_at_fraction(
        data,
        uplift_col=uplift_col,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        fraction=fraction,
    )
    if not np.isfinite(original_policy["observed_uplift"]):
        raise ValueError(
            "top-fraction selection requires both treatment arms and finite outcome rates"
        )

    rng = np.random.default_rng(random_state)
    boot_uplift = np.empty(n_boot, dtype=float)
    # Allocate profit samples only when both economics inputs are available.
    boot_profit = np.empty(n_boot, dtype=float) if cost_per_contact is not None else None

    n = len(data)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = data.iloc[idx]
        result = policy_value_at_fraction(
            sample,
            uplift_col=uplift_col,
            treatment_col=treatment_col,
            outcome_col=outcome_col,
            fraction=fraction,
        )
        boot_uplift[i] = result["observed_uplift"]
        if boot_profit is not None:
            n_sel = result["selected_n"]
            assert cost_per_contact is not None and value_per_conversion is not None
            boot_profit[i] = (
                result["observed_uplift"] * n_sel * value_per_conversion
                - n_sel * cost_per_contact
            )

    # Percentiles on NaN-containing arrays are NaN even when most draws are
    # usable. Preserve the number of discarded draws so interval reliability
    # is visible rather than silently reported as a fully supported CI.
    valid_mask = np.isfinite(boot_uplift)
    n_valid = int(valid_mask.sum())
    if n_valid < 2:
        raise ValueError(
            "fewer than two bootstrap resamples have both treatment arms and "
            "finite outcome rates in the selected segment; increase the sample "
            "size or contact fraction"
        )
    valid_uplift = boot_uplift[valid_mask]
    alpha = (1.0 - confidence) / 2.0
    ci_low_u, ci_high_u = np.quantile(valid_uplift, [alpha, 1.0 - alpha])
    summary: dict[str, float] = {
        "fraction": fraction,
        "n_boot": float(n_boot),
        "n_valid_boot": float(n_valid),
        "n_invalid_boot": float(n_boot - n_valid),
        "uplift_mean": float(np.mean(valid_uplift)),
        "uplift_ci_low": float(ci_low_u),
        "uplift_ci_high": float(ci_high_u),
    }
    if boot_profit is not None:
        valid_profit = boot_profit[valid_mask]
        ci_low_p, ci_high_p = np.quantile(valid_profit, [alpha, 1.0 - alpha])
        summary.update(
            profit_mean=float(np.mean(valid_profit)),
            profit_ci_low=float(ci_low_p),
            profit_ci_high=float(ci_high_p),
        )
    return summary


def bootstrap_decile_uplift(
    data: pd.DataFrame,
    *,
    n_deciles: int = 10,
    n_boot: int = 500,
    confidence: float = 0.95,
    uplift_col: str = "uplift_score",
    treatment_col: str = "treatment",
    outcome_col: str = "conversion",
    random_state: int = 42,
) -> pd.DataFrame:
    """Bootstrap CI for observed uplift inside each uplift decile.

    For each bootstrap resample, customers are re-binned into deciles by
    their uplift score (so a customer can drift between deciles across
    samples — this is part of the stability question). The observed
    treated-control gap inside each decile is recorded; CIs come from
    quantiles across the ``n_boot`` samples.
    """

    if n_deciles < 2:
        raise ValueError("n_deciles must be >= 2")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    if n_boot < 2:
        raise ValueError("n_boot must be >= 2")

    missing = [c for c in (uplift_col, treatment_col, outcome_col) if c not in data]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    rng = np.random.default_rng(random_state)
    n = len(data)
    boot = np.full((n_boot, n_deciles), np.nan, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = data.iloc[idx].reset_index(drop=True)
        ranks = sample[uplift_col].rank(method="first", ascending=False)
        deciles = pd.qcut(ranks, q=n_deciles, labels=False, duplicates="drop") + 1
        sample = sample.assign(_decile=deciles.astype(int))
        for d in range(1, n_deciles + 1):
            chunk = sample[sample["_decile"] == d]
            treated = chunk[chunk[treatment_col] == 1]
            control = chunk[chunk[treatment_col] == 0]
            if len(treated) == 0 or len(control) == 0:
                continue
            boot[i, d - 1] = (
                treated[outcome_col].mean() - control[outcome_col].mean()
            )

    alpha = (1.0 - confidence) / 2.0
    rows = []
    for d in range(n_deciles):
        column = boot[:, d]
        valid = column[~np.isnan(column)]
        if len(valid) < 2:
            rows.append(
                {"decile": d + 1, "uplift_mean": np.nan, "ci_low": np.nan, "ci_high": np.nan}
            )
            continue
        ci_low, ci_high = np.quantile(valid, [alpha, 1.0 - alpha])
        rows.append(
            {
                "decile": d + 1,
                "uplift_mean": float(np.mean(valid)),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
            }
        )
    return pd.DataFrame(rows)
