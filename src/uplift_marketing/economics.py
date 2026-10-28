"""Campaign economics, budget optimisation, and scenario analysis.

This module wraps the cost / value assumptions a campaign owner brings to
the uplift workflow into a single :class:`CampaignEconomics` value object,
and exposes a :func:`optimize_budget` helper that returns the recommended
top-fraction together with the customer list it implies. A
:func:`scenario_table` helper sweeps over a grid of policies and economics
scenarios for sensitivity analysis.

Everything here operates on the project's standard scored DataFrame —
the output of :func:`uplift_marketing.uplift.attach_uplift_scores` — so it
composes with the rest of the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass(frozen=True)
class CampaignEconomics:
    """Per-contact cost and per-conversion value assumptions.

    Both quantities are expressed in the same currency (typically EUR or
    USD). The class is intentionally minimal — richer cost models (channel
    differences, fatigue penalties, contribution margin) belong in a
    project-specific subclass that overrides :meth:`net_profit`.
    """

    cost_per_contact: float
    value_per_conversion: float
    currency: str = "EUR"

    def __post_init__(self) -> None:
        if not np.isfinite(self.cost_per_contact) or self.cost_per_contact < 0:
            raise ValueError("cost_per_contact must be finite and non-negative")
        if not np.isfinite(self.value_per_conversion) or self.value_per_conversion <= 0:
            raise ValueError("value_per_conversion must be finite and positive")

    @property
    def break_even_uplift(self) -> float:
        """Per-customer uplift at which a contact pays for itself."""

        return self.cost_per_contact / self.value_per_conversion

    def net_profit(
        self,
        incremental_conversions: float | NDArray[Any],
        n_contacts: float | NDArray[Any],
    ) -> float | NDArray[Any]:
        """Net profit given expected incremental conversions and contact count."""

        revenue = np.asarray(incremental_conversions, dtype=float) * self.value_per_conversion
        cost = np.asarray(n_contacts, dtype=float) * self.cost_per_contact
        result = revenue - cost
        if np.ndim(result) == 0:
            return float(result)
        return result


@dataclass(frozen=True)
class BudgetRecommendation:
    """Result of :func:`optimize_budget`.

    Attributes
    ----------
    fraction:
        Optimal top-k fraction selected by the chosen score.
    n_selected:
        Number of customers in the recommended selection.
    threshold_score:
        The score value at the cut-off, or ``None`` when no customers should
        be contacted. At ties, ``targets`` gives the exact selection.
    predicted_incremental_conversions:
        Sum of the uplift score over the selected customers (the model's
        belief about the campaign's lift). Use the calibration discussion
        in notebook 06 before quoting this to stakeholders.
    predicted_revenue / predicted_cost / predicted_net_profit:
        Economics applied to the prediction above.
    targets:
        The selected rows from the scored DataFrame, sorted by the score
        descending. Includes whatever columns the input carried — typically
        ``customer_id``, the uplift score, and any segment labels.
    profit_curve:
        Per-fraction predicted profit table; the row achieving the maximum
        is the one the recommendation is built from.
    """

    fraction: float
    n_selected: int
    threshold_score: float | None
    predicted_incremental_conversions: float
    predicted_revenue: float
    predicted_cost: float
    predicted_net_profit: float
    targets: pd.DataFrame
    profit_curve: pd.DataFrame


def _expected_profit_at_count(
    sorted_scores: NDArray[Any],
    cumulative_uplift: NDArray[Any],
    n_selected: int,
    fraction: float,
    economics: CampaignEconomics,
) -> dict[str, float]:
    """Summarise one contact count using already-sorted cumulative uplift."""

    predicted_inc = float(cumulative_uplift[n_selected - 1]) if n_selected else 0.0
    revenue = predicted_inc * economics.value_per_conversion
    cost = n_selected * economics.cost_per_contact
    return {
        "fraction": fraction,
        "n_selected": n_selected,
        "predicted_incremental_conversions": predicted_inc,
        "revenue": revenue,
        "cost": cost,
        "net_profit": revenue - cost,
        "threshold_score": float(sorted_scores[n_selected - 1]) if n_selected else float("nan"),
    }


def optimize_budget(
    scored: pd.DataFrame,
    economics: CampaignEconomics,
    *,
    score_col: str = "uplift_score",
    uplift_col: str = "uplift_score",
    fractions: NDArray[Any] | None = None,
) -> BudgetRecommendation:
    """Pick the top-fraction that maximises predicted net profit.

    By default, evaluates every possible top-k contact count using cumulative
    uplift and returns the global maximum, including the no-contact baseline.
    The ``profit_curve`` retains the compact plotting grid plus the exact
    optimum, so dashboard exports remain small. Providing ``fractions``
    explicitly limits the search to that grid and zero contacts. A no-contact
    recommendation has an empty ``targets`` DataFrame and
    ``threshold_score=None``.

    Parameters
    ----------
    scored:
        DataFrame produced by
        :func:`uplift_marketing.uplift.attach_uplift_scores`, containing the
        feature columns and at least ``score_col`` and ``uplift_col``.
    economics:
        Cost and value assumptions.
    score_col:
        Column used to rank customers for selection. Defaults to the
        ``uplift_score`` produced upstream; pass a different name to compare
        a response-targeted policy against an uplift-targeted one with the
        same economics.
    uplift_col:
        Column whose sum over the selected rows is taken as the predicted
        incremental conversions. Defaults to ``uplift_score``.
    fractions:
        Optional non-empty grid of contact fractions in ``(0, 1]``.
        If omitted, all top-k counts are eligible and a 50-point summary
        grid is used only for plotting. If provided, only the specified
        fractions are eligible. The zero-contact baseline is always included.
    """

    missing = [c for c in (score_col, uplift_col) if c not in scored.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    exact_search = fractions is None
    if exact_search:
        fractions = np.round(np.linspace(0.02, 1.0, 50), 4)
    fractions = np.asarray(fractions, dtype=float)
    if (
        fractions.ndim != 1
        or fractions.size == 0
        or not np.all(np.isfinite(fractions))
        or np.any((fractions <= 0) | (fractions > 1))
    ):
        raise ValueError("fractions must be a non-empty 1D grid in (0, 1]")

    # A stable sort keeps ties deterministic; one cumulative sum avoids
    # re-sorting and re-summing the customer base at each candidate budget.
    ranked = scored.sort_values(score_col, ascending=False, kind="mergesort")
    sorted_scores = ranked[score_col].to_numpy(dtype=float)
    sorted_uplift = ranked[uplift_col].to_numpy(dtype=float)
    if not np.isfinite(sorted_scores).all() or not np.isfinite(sorted_uplift).all():
        raise ValueError("score and uplift columns must contain only finite values")

    n_total = len(ranked)
    cumulative_uplift = np.cumsum(sorted_uplift)
    rows = [_expected_profit_at_count(sorted_scores, cumulative_uplift, 0, 0.0, economics)]
    for fraction in fractions:
        n_selected = min(n_total, max(1, int(np.ceil(n_total * fraction))))
        rows.append(
            _expected_profit_at_count(
                sorted_scores, cumulative_uplift, n_selected, float(fraction), economics
            )
        )

    best = max(rows, key=lambda row: row["net_profit"])
    if exact_search and n_total:
        # This global optimum need not lie on the 50-point plotting grid.
        cumulative_profit = (
            cumulative_uplift * economics.value_per_conversion
            - np.arange(1, n_total + 1) * economics.cost_per_contact
        )
        exact_count = int(np.argmax(np.r_[0.0, cumulative_profit]))
        best = _expected_profit_at_count(
            sorted_scores,
            cumulative_uplift,
            exact_count,
            exact_count / n_total,
            economics,
        )
        if exact_count > 0 and not any(
            row["n_selected"] == exact_count and row["fraction"] == best["fraction"]
            for row in rows
        ):
            rows.append(best)
            rows.sort(key=lambda row: row["fraction"])

    profit_curve = pd.DataFrame(rows)
    n_selected = int(best["n_selected"])
    targets = ranked.head(n_selected).copy()
    return BudgetRecommendation(
        fraction=float(best["fraction"]),
        n_selected=n_selected,
        threshold_score=float(best["threshold_score"]) if n_selected else None,
        predicted_incremental_conversions=float(best["predicted_incremental_conversions"]),
        predicted_revenue=float(best["revenue"]),
        predicted_cost=float(best["cost"]),
        predicted_net_profit=float(best["net_profit"]),
        targets=targets,
        profit_curve=profit_curve,
    )


def scenario_table(
    scored: pd.DataFrame,
    economics_scenarios: dict[str, CampaignEconomics],
    *,
    policies: dict[str, str] | None = None,
    uplift_col: str = "uplift_score",
    fractions: NDArray[Any] | None = None,
) -> pd.DataFrame:
    """Sweep policy x economics combinations and report each optimum.

    Parameters
    ----------
    scored:
        Scored DataFrame.
    economics_scenarios:
        Mapping from a label (e.g. ``"baseline"`` / ``"low_value"``) to a
        :class:`CampaignEconomics` instance.
    policies:
        Mapping from policy label to the column to sort by. Defaults to a
        single ``{"uplift_targeted": uplift_col}`` policy.
    uplift_col:
        Column summed to produce predicted incremental conversions.
    fractions:
        Optional grid restricting eligible budgets. Without a grid, every
        possible top-k prefix is eligible; zero contacts are always evaluated.

    Returns
    -------
    pandas.DataFrame
        One row per (policy, scenario) pair with the optimum fraction,
        selected count, predicted profit, and break-even uplift.
    """

    if not economics_scenarios:
        raise ValueError("economics_scenarios must not be empty")
    if policies is None:
        policies = {"uplift_targeted": uplift_col}

    rows: list[dict[str, float | int | str]] = []
    for scenario_name, economics in economics_scenarios.items():
        for policy_name, score_col in policies.items():
            recommendation = optimize_budget(
                scored,
                economics,
                score_col=score_col,
                uplift_col=uplift_col,
                fractions=fractions,
            )
            rows.append(
                {
                    "policy": policy_name,
                    "scenario": scenario_name,
                    "cost_per_contact": economics.cost_per_contact,
                    "value_per_conversion": economics.value_per_conversion,
                    "break_even_uplift": economics.break_even_uplift,
                    "optimal_fraction": recommendation.fraction,
                    "n_selected": recommendation.n_selected,
                    "predicted_incremental": round(
                        recommendation.predicted_incremental_conversions, 2
                    ),
                    "predicted_net_profit": round(recommendation.predicted_net_profit, 2),
                }
            )
    return pd.DataFrame(rows)
