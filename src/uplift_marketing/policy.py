"""Campaign policy segmentation and recommended actions."""

from __future__ import annotations

import numpy as np
import pandas as pd

ACTION_MAP = {
    "persuadable": "Target with campaign. These customers appear responsive to treatment.",
    "sure_thing": "Avoid aggressive targeting. They may convert without paid intervention.",
    "lost_cause": "Do not spend aggressively. Test cheaper education or broad nurture only.",
    "do_not_disturb": "Avoid treatment. The estimated effect is negative or harmful.",
    "review": "Keep under review. The estimated effect is uncertain or moderate.",
}

CAUTION_MAP = {
    "persuadable": "Validate incrementality in holdout tests before scaling spend.",
    "sure_thing": "Do not confuse high conversion probability with high uplift.",
    "lost_cause": "Low estimated response does not mean low customer value.",
    "do_not_disturb": "Investigate whether treatment fatigue or poor targeting is present.",
    "review": "Use business context and experimentation before automating decisions.",
}


def assign_campaign_segments(
    scored: pd.DataFrame,
    *,
    uplift_col: str = "uplift_score",
    p_control_col: str = "p_control",
    p_treated_col: str = "p_treated",
    high_uplift_threshold: float | None = None,
    low_probability_threshold: float = 0.05,
    high_probability_threshold: float = 0.40,
) -> pd.Series:
    """Assign campaign decision segments from predicted potential outcomes.

    Parameters
    ----------
    scored:
        DataFrame containing potential outcome probabilities and uplift scores.
    high_uplift_threshold:
        Threshold for the persuadable segment. If omitted, the 75th percentile
        of positive uplift scores is used.
    low_probability_threshold:
        Maximum treated probability for the lost-cause segment.
    high_probability_threshold:
        Minimum control probability for the sure-thing segment.

    Notes
    -----
    A customer is persuadable only with strictly positive predicted uplift.
    Negative uplift always takes precedence over the other classifications.
    Invalid probabilities and non-finite predictions are rejected before
    assigning any campaign action.
    """

    required = {uplift_col, p_control_col, p_treated_col}
    missing = required - set(scored.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    uplift = scored[uplift_col].astype(float)
    p_control = scored[p_control_col].astype(float)
    p_treated = scored[p_treated_col].astype(float)

    if not (
        np.isfinite(low_probability_threshold)
        and np.isfinite(high_probability_threshold)
        and 0 <= low_probability_threshold <= high_probability_threshold <= 1
    ):
        raise ValueError("probability thresholds must satisfy 0 <= low <= high <= 1")
    if high_uplift_threshold is not None and (
        not np.isfinite(high_uplift_threshold) or high_uplift_threshold < 0
    ):
        raise ValueError("high_uplift_threshold must be finite and non-negative")

    if not all(np.isfinite(values.to_numpy()).all() for values in (uplift, p_control, p_treated)):
        raise ValueError("uplift scores and predicted probabilities must be finite")
    if ((p_control < 0) | (p_control > 1) | (p_treated < 0) | (p_treated > 1)).any():
        raise ValueError("predicted probabilities must be in [0, 1]")

    positive_uplift = uplift[uplift > 0]
    if high_uplift_threshold is None:
        high_uplift_threshold = (
            float(positive_uplift.quantile(0.75)) if len(positive_uplift) else 0.0
        )

    segment = np.full(len(scored), "review", dtype=object)
    nonnegative = uplift >= 0
    segment[nonnegative & (p_treated <= low_probability_threshold) & (uplift <= 0.02)] = (
        "lost_cause"
    )
    segment[nonnegative & (p_control >= high_probability_threshold) & (uplift <= 0.02)] = (
        "sure_thing"
    )
    # An implicit threshold of zero does not make zero-uplift users persuadable.
    segment[(uplift > 0) & (uplift >= high_uplift_threshold)] = "persuadable"
    # Negative uplift is never overridden by high response probabilities.
    segment[uplift < 0] = "do_not_disturb"

    return pd.Series(segment, index=scored.index, name="campaign_segment")


def campaign_action_table(
    scored: pd.DataFrame, *, segment_col: str = "campaign_segment"
) -> pd.DataFrame:
    """Summarize segment sizes, uplift scores, and recommended actions."""

    if segment_col not in scored:
        raise ValueError(f"Segment column not found: {segment_col}")
    if "uplift_score" not in scored:
        raise ValueError("scored must contain an uplift_score column")

    summary = (
        scored.groupby(segment_col, observed=True)
        .agg(
            n=(segment_col, "size"),
            share=(segment_col, lambda values: len(values) / len(scored)),
            mean_uplift=("uplift_score", "mean"),
            median_uplift=("uplift_score", "median"),
            mean_p_control=("p_control", "mean"),
            mean_p_treated=("p_treated", "mean"),
        )
        .reset_index()
    )
    summary["recommended_action"] = summary[segment_col].map(ACTION_MAP)
    summary["caution"] = summary[segment_col].map(CAUTION_MAP)
    return summary.sort_values("mean_uplift", ascending=False)
