"""Programmatic figure generation for the campaign policy report.

The notebooks already produce figures inline via ``save_figure``; this
module is the headless counterpart — a small set of canonical figures
that the pipeline can render without a Jupyter kernel. Each ``fig_*``
function returns a :class:`matplotlib.figure.Figure` and is independent
of any global state, so it composes cleanly with both the dashboard
exporter and ad-hoc analyses. :func:`export_campaign_figures` wraps the
whole set and writes PNGs under ``output_dir``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from uplift_marketing.evaluation import qini_curve, uplift_curve
from uplift_marketing.uplift import make_uplift_deciles, uplift_by_group

if TYPE_CHECKING:
    from uplift_marketing.economics import BudgetRecommendation, CampaignEconomics


PALETTE = {
    "primary": "#3b6ea8",
    "secondary": "#b04a4a",
    "neutral": "#7d8a9b",
    "accent": "#e0a82e",
}


_PLOT_STYLE: dict[str, object] = {
    "figure.figsize": (8.0, 4.5),
    "figure.dpi": 110,
    "savefig.dpi": 140,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "axes.titlesize": 12,
    "axes.titleweight": "semibold",
    "axes.labelsize": 10,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
}


def _styled() -> plt.rc_context:
    """Apply the project's matplotlib style without mutating globals."""

    return plt.rc_context(_PLOT_STYLE)


def fig_qini_curve(
    scored: pd.DataFrame,
    *,
    treatment_col: str = "treatment",
    outcome_col: str = "conversion",
    uplift_col: str = "uplift_score",
    title: str | None = None,
) -> plt.Figure:
    """Qini curve: cumulative incremental conversions vs targeted fraction.

    The diagonal is the random-targeting baseline; the model curve sitting
    above it is what "the model has signal" looks like. ``area_under_qini``
    integrates the difference between the two.
    """

    qini = qini_curve(
        outcome=scored[outcome_col].to_numpy(),
        treatment=scored[treatment_col].to_numpy(),
        uplift_score=scored[uplift_col].to_numpy(),
    )
    fraction = qini["population_fraction"].to_numpy()
    incremental = qini["incremental_gain"].to_numpy()
    final_gain = float(incremental[-1]) if len(incremental) else 0.0
    random_baseline = fraction * final_gain

    with _styled():
        fig, ax = plt.subplots()
        ax.plot(fraction, incremental, color=PALETTE["primary"], label="Model")
        ax.plot(
            fraction,
            random_baseline,
            color=PALETTE["neutral"],
            linestyle="--",
            label="Random",
        )
        ax.set_xlabel("Fraction targeted")
        ax.set_ylabel("Cumulative incremental conversions")
        ax.set_title(title or "Qini curve — model vs random targeting")
        ax.legend()
    return fig


def fig_profit_curve(
    recommendation: BudgetRecommendation,
    economics: CampaignEconomics,
    *,
    title: str | None = None,
) -> plt.Figure:
    """Predicted net profit vs targeted fraction with the optimum marked.

    The curve typically rises early (high-uplift customers pay for their
    contact), peaks somewhere in the middle, and falls past the
    break-even uplift as additional contacts cost more than they earn.
    The marked point is the cut-off the budget optimizer recommends.
    """

    profit_curve = recommendation.profit_curve
    with _styled():
        fig, ax = plt.subplots()
        ax.plot(
            profit_curve["fraction"],
            profit_curve["net_profit"],
            color=PALETTE["primary"],
            label="Predicted profit",
        )
        ax.axhline(0, color="black", linewidth=0.6)
        ax.scatter(
            [recommendation.fraction],
            [recommendation.predicted_net_profit],
            color=PALETTE["secondary"],
            zorder=5,
            label=(
                f"Optimum: top {recommendation.fraction:.2%} "
                f"({recommendation.n_selected:,} customers)"
            ),
        )
        ax.set_xlabel("Fraction targeted")
        ax.set_ylabel(f"Predicted net profit ({economics.currency})")
        ax.set_title(title or "Predicted profit curve and optimum cut-off")
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        ax.legend()
    return fig


def fig_uplift_by_decile(
    scored: pd.DataFrame,
    *,
    treatment_col: str = "treatment",
    outcome_col: str = "conversion",
    uplift_col: str = "uplift_score",
    title: str | None = None,
) -> plt.Figure:
    """Observed treated-control uplift by uplift-score decile.

    Decile 1 contains the highest-predicted-uplift customers. A
    well-ranked model produces a monotone (or nearly monotone) decreasing
    bar chart from decile 1 to decile 10.
    """

    if "uplift_decile" in scored.columns:
        deciles = scored["uplift_decile"]
    else:
        deciles = make_uplift_deciles(scored, uplift_col=uplift_col)
    frame = scored.assign(uplift_decile=deciles)
    table = uplift_by_group(
        frame,
        group_col="uplift_decile",
        treatment_col=treatment_col,
        outcome_col=outcome_col,
    ).sort_values("uplift_decile")

    with _styled():
        fig, ax = plt.subplots()
        bar_colors = [
            PALETTE["primary"] if value >= 0 else PALETTE["secondary"]
            for value in table["observed_uplift"]
        ]
        ax.bar(
            table["uplift_decile"].astype(int),
            table["observed_uplift"],
            color=bar_colors,
        )
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xticks(table["uplift_decile"].astype(int))
        ax.set_xlabel("Uplift decile (1 = highest predicted)")
        ax.set_ylabel("Observed uplift (treated − control)")
        ax.set_title(title or "Observed uplift by predicted decile")
    return fig


def fig_scenarios(
    scenarios_table: pd.DataFrame,
    *,
    currency: str = "EUR",
    title: str | None = None,
) -> plt.Figure:
    """Predicted net profit by (scenario × policy) at each policy's optimum.

    Reads the output of
    :func:`uplift_marketing.economics.scenario_table` and pivots it into
    a grouped bar chart, one bar per policy within each economics
    scenario. The point of the chart is to show how the policy *ranking*
    changes with cost / value assumptions — not to commit to absolute
    numbers.
    """

    required = {"scenario", "policy", "predicted_net_profit"}
    missing = required - set(scenarios_table.columns)
    if missing:
        raise ValueError(f"scenarios_table is missing columns: {sorted(missing)}")

    pivot = scenarios_table.pivot(
        index="scenario", columns="policy", values="predicted_net_profit"
    )
    policies = list(pivot.columns)

    with _styled():
        fig, ax = plt.subplots()
        x_pos = np.arange(len(pivot.index))
        width = 0.8 / max(len(policies), 1)
        # Cycle the palette so each policy gets a stable color.
        colors_cycle = [
            PALETTE["primary"],
            PALETTE["secondary"],
            PALETTE["neutral"],
            PALETTE["accent"],
        ]
        for i, policy in enumerate(policies):
            ax.bar(
                x_pos + (i - (len(policies) - 1) / 2) * width,
                pivot[policy].to_numpy(),
                width,
                label=policy,
                color=colors_cycle[i % len(colors_cycle)],
            )
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(pivot.index, rotation=15, ha="right")
        ax.set_ylabel(f"Predicted net profit at policy optimum ({currency})")
        ax.set_title(title or "Sensitivity of predicted profit to economics")
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        ax.legend()
    return fig


def fig_uplift_curve(
    scored: pd.DataFrame,
    *,
    treatment_col: str = "treatment",
    outcome_col: str = "conversion",
    uplift_col: str = "uplift_score",
    title: str | None = None,
) -> plt.Figure:
    """Cumulative population-rescaled uplift curve (AUUC).

    Uplift curve is the population-rescaled cousin of Qini: the y-axis
    rescales the cumulative incremental gain so curves from samples of
    different size remain comparable.
    """

    auc = uplift_curve(
        outcome=scored[outcome_col].to_numpy(),
        treatment=scored[treatment_col].to_numpy(),
        uplift_score=scored[uplift_col].to_numpy(),
    )
    with _styled():
        fig, ax = plt.subplots()
        ax.plot(
            auc["population_fraction"],
            auc["incremental_gain"],
            color=PALETTE["primary"],
        )
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xlabel("Fraction targeted")
        ax.set_ylabel("Population-rescaled cumulative uplift")
        ax.set_title(title or "Uplift curve (AUUC)")
    return fig


def export_campaign_figures(
    scored: pd.DataFrame,
    *,
    recommendation: BudgetRecommendation,
    economics: CampaignEconomics,
    scenarios_table: pd.DataFrame | None = None,
    output_dir: str | Path,
    treatment_col: str = "treatment",
    outcome_col: str = "conversion",
    uplift_col: str = "uplift_score",
    prefix: str = "campaign",
) -> dict[str, Path]:
    """Write the canonical campaign figures as PNGs and return their paths.

    The set is deliberately small — one figure per question a stakeholder
    actually asks:

    - ``qini`` — does the model rank better than random?
    - ``profit`` — what's the optimal contact budget at this economics?
    - ``decile`` — does the predicted ranking line up with realised uplift?
    - ``uplift_curve`` — population-rescaled cumulative uplift (AUUC).
    - ``scenarios`` — how does the recommendation move with cost / value?
      Only written when ``scenarios_table`` is supplied.

    Each figure is closed after writing so the function is safe to call
    repeatedly without leaking matplotlib state.
    """

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    figures: dict[str, plt.Figure] = {
        "qini": fig_qini_curve(
            scored,
            treatment_col=treatment_col,
            outcome_col=outcome_col,
            uplift_col=uplift_col,
        ),
        "profit": fig_profit_curve(recommendation, economics),
        "decile": fig_uplift_by_decile(
            scored,
            treatment_col=treatment_col,
            outcome_col=outcome_col,
            uplift_col=uplift_col,
        ),
        "uplift_curve": fig_uplift_curve(
            scored,
            treatment_col=treatment_col,
            outcome_col=outcome_col,
            uplift_col=uplift_col,
        ),
    }
    if scenarios_table is not None and not scenarios_table.empty:
        figures["scenarios"] = fig_scenarios(
            scenarios_table, currency=economics.currency
        )

    for label, fig in figures.items():
        path = out_dir / f"{prefix}_{label}.png"
        fig.savefig(path)
        plt.close(fig)
        paths[label] = path
    return paths
