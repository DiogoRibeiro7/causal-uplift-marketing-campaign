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


def uplift_curve(
    outcome: np.ndarray,
    treatment: np.ndarray,
    uplift_score: np.ndarray,
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
    observational data require appropriate propensity adjustment.
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

    n_t = max(int(treated_cum[-1]), 1)
    n_c = max(int(control_cum[-1]), 1)

    # Keep the *global* arm denominators fixed across all ranked prefixes.
    # Subtracting raw arm conversions would mistake allocation imbalance
    # for a treatment effect even when both arms convert at the same rate.
    n_total = len(outcome)
    incremental_gain = n_total * (
        treated_positive_cum / n_t - control_positive_cum / n_c
    )
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
    """Approximate the area under an AUUC-style cumulative uplift curve."""

    required = {"population_fraction", "incremental_gain"}
    missing = required - set(curve.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    return float(auc(curve["population_fraction"], curve["incremental_gain"]))


def normalized_qini(
    outcome: np.ndarray,
    treatment: np.ndarray,
    uplift_score: np.ndarray,
) -> dict[str, float]:
    """Return the area under the Qini curve normalized to ``[0, 1]``.

    Raw Qini AUC scales with population size and conversion rate, which makes
    cross-dataset comparisons hard. Normalization divides by the *oracle*
    AUC — the AUC achieved by ranking customers by their realised treated /
    control conversions, i.e. by the best possible ranking on the same
    sample. The score is bounded above by 1.0 (oracle) and equal to 0
    when the model ranks no better than random.

    The dictionary returned contains:

    - ``model_auc`` — area under the Qini curve produced by ``uplift_score``;
    - ``oracle_auc`` — area under the Qini curve when sorting by the
      observed treated-control rate difference (an in-sample oracle);
    - ``random_auc`` — expected AUC under uniform random ranking;
    - ``normalized`` — ``(model_auc - random_auc) / (oracle_auc - random_auc)``;
      this is what to quote when comparing models across datasets.
    """

    outcome = np.asarray(outcome).astype(float)
    treatment = np.asarray(treatment).astype(int)
    uplift_score = np.asarray(uplift_score).astype(float)

    model_curve = qini_curve(outcome, treatment, uplift_score)
    model_auc = area_under_qini(model_curve)

    # Oracle: an in-sample upper bound on Qini achievable by any ranking.
    # A customer's realised contribution to the Qini sum depends on
    # treatment, outcome, and the running ratio of treated to control. The
    # cleanest in-sample oracle is to sort by the customer's signed
    # contribution to treated_positive minus control_positive after
    # rebalancing — equivalent to sorting by ``y * (2t - 1)`` (treated
    # positives first, control positives last) for a balanced experiment.
    # We score by an unbiased per-row contribution and re-run qini_curve
    # on those scores.
    n_t = max(int((treatment == 1).sum()), 1)
    n_c = max(int((treatment == 0).sum()), 1)
    oracle_score = np.where(
        treatment == 1,
        outcome / n_t,
        -outcome / n_c,
    )
    oracle_curve = qini_curve(outcome, treatment, oracle_score)
    oracle_auc = area_under_qini(oracle_curve)

    # Under uniform random ranking, the curve is the diagonal that goes from
    # (0, 0) to (1, total_qini_gain). Its area is half the endpoint value.
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
    and ``value_per_conversion`` are provided, also reports the bootstrap
    distribution of realised net profit for the same selection.
    """

    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in the interval (0, 1]")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    if n_boot < 2:
        raise ValueError("n_boot must be >= 2")

    missing = [c for c in (uplift_col, treatment_col, outcome_col) if c not in data]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    rng = np.random.default_rng(random_state)
    boot_uplift = np.empty(n_boot, dtype=float)
    boot_profit = np.empty(n_boot, dtype=float) if cost_per_contact is not None else None

    n = len(data)
    have_economics = cost_per_contact is not None and value_per_conversion is not None
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
        if have_economics:
            n_sel = result["selected_n"]
            assert boot_profit is not None
            boot_profit[i] = (
                result["observed_uplift"] * n_sel * value_per_conversion
                - n_sel * cost_per_contact
            )

    alpha = (1.0 - confidence) / 2.0
    ci_low_u, ci_high_u = np.quantile(boot_uplift, [alpha, 1.0 - alpha])
    summary: dict[str, float] = {
        "fraction": fraction,
        "n_boot": float(n_boot),
        "uplift_mean": float(np.nanmean(boot_uplift)),
        "uplift_ci_low": float(ci_low_u),
        "uplift_ci_high": float(ci_high_u),
    }
    if boot_profit is not None:
        ci_low_p, ci_high_p = np.quantile(boot_profit, [alpha, 1.0 - alpha])
        summary.update(
            profit_mean=float(np.nanmean(boot_profit)),
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
