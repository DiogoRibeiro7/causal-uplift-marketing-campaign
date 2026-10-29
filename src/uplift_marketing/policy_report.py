"""Campaign policy report generator.

The model card is the *due-diligence* artifact — the one a deployment
reviewer signs off. The policy report is the *decision* artifact — the
one that answers "what should we do, why, and how much will it move the
needle?". Both are markdown by design so they live next to the code,
diff cleanly, and embed straight into a wiki.

:func:`build_policy_report` collects the recommendation, the economics,
the metrics, the scenario sweep, and the canonical figures into a
:class:`PolicyReport`. :func:`write_policy_report` renders it to a
markdown document under ``reports/`` so the campaign owner can read it
without opening a notebook.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from uplift_marketing.evaluation import (
    area_under_qini,
    area_under_uplift,
    normalized_qini,
    qini_curve,
    uplift_curve,
)
from uplift_marketing.treatment import naive_average_treatment_effect

if TYPE_CHECKING:
    from uplift_marketing.economics import BudgetRecommendation, CampaignEconomics


@dataclass(frozen=True)
class PolicyReport:
    """Inputs the policy-report renderer reads.

    Build with :func:`build_policy_report`; render with :meth:`to_markdown`
    or write directly via :func:`write_policy_report`.
    """

    model_name: str
    dataset_source: str
    n_total: int
    n_treated: int
    n_control: int
    naive_ate: float
    auq: float
    auuc: float
    normalized_qini_score: float
    economics: CampaignEconomics
    recommendation: BudgetRecommendation
    scenarios_table: pd.DataFrame | None
    figure_paths: dict[str, Path]
    model_card_path: Path | None
    generated_at: datetime
    notes: list[str] = field(default_factory=list)

    def to_markdown(self, *, base_dir: Path | None = None) -> str:
        """Render the report to markdown.

        Figure paths in ``figure_paths`` are rewritten as relative paths
        from ``base_dir`` so the markdown embeds cleanly when written
        next to the figures. If ``base_dir`` is omitted, the figure
        paths are left as-is.
        """

        return _render_markdown(self, base_dir=base_dir)


def build_policy_report(
    scored: pd.DataFrame,
    *,
    economics: CampaignEconomics,
    recommendation: BudgetRecommendation,
    scenarios_table: pd.DataFrame | None = None,
    figure_paths: dict[str, Path] | None = None,
    model_card_path: Path | None = None,
    model_name: str = "T-learner",
    dataset_source: str = "unknown",
    treatment_col: str = "treatment",
    outcome_col: str = "conversion",
    uplift_col: str = "uplift_score",
    notes: list[str] | None = None,
) -> PolicyReport:
    """Assemble a :class:`PolicyReport` from a scored DataFrame.

    Computes the headline metrics that the report prints in the executive
    summary and bundles the rest of the inputs (economics, recommendation,
    scenarios, figures, model card pointer) into a single value object.
    """

    required = {treatment_col, outcome_col, uplift_col}
    missing = required - set(scored.columns)
    if missing:
        raise ValueError(f"scored is missing required columns: {sorted(missing)}")

    treatment = scored[treatment_col].to_numpy()
    outcome = scored[outcome_col].to_numpy()
    uplift = scored[uplift_col].to_numpy()

    n_total = int(len(scored))
    n_treated = int((treatment == 1).sum())
    n_control = int((treatment == 0).sum())
    naive_ate = float(
        naive_average_treatment_effect(
            scored, treatment_col=treatment_col, outcome_col=outcome_col
        )
    )
    qini = qini_curve(outcome=outcome, treatment=treatment, uplift_score=uplift)
    auq = float(area_under_qini(qini))
    uplift_c = uplift_curve(outcome=outcome, treatment=treatment, uplift_score=uplift)
    auuc = float(area_under_uplift(uplift_c))
    norm = normalized_qini(outcome=outcome, treatment=treatment, uplift_score=uplift)
    norm_score = float(norm["normalized"]) if not np.isnan(norm["normalized"]) else float("nan")

    return PolicyReport(
        model_name=model_name,
        dataset_source=dataset_source,
        n_total=n_total,
        n_treated=n_treated,
        n_control=n_control,
        naive_ate=naive_ate,
        auq=auq,
        auuc=auuc,
        normalized_qini_score=norm_score,
        economics=economics,
        recommendation=recommendation,
        scenarios_table=scenarios_table,
        figure_paths=dict(figure_paths or {}),
        model_card_path=model_card_path,
        generated_at=datetime.now(timezone.utc),
        notes=list(notes or []),
    )


def write_policy_report(report: PolicyReport, output_path: str | Path) -> Path:
    """Render ``report`` to markdown and write it to ``output_path``.

    Figure paths are rewritten relative to the report's parent directory
    so the document embeds cleanly without absolute paths.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    md = report.to_markdown(base_dir=path.parent)
    path.write_text(md, encoding="utf-8")
    return path


def _relative(target: Path, base: Path) -> str:
    """Best-effort relative path; falls back to the absolute path on failure."""

    try:
        return target.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return target.as_posix()


def _render_markdown(report: PolicyReport, *, base_dir: Path | None) -> str:
    lines: list[str] = []
    lines.append(f"# Campaign policy report — {report.model_name}")
    lines.append("")
    lines.append(
        f"_Dataset: **{report.dataset_source}** · Generated: "
        f"{report.generated_at:%Y-%m-%d %H:%M UTC}_"
    )
    lines.append("")

    lines.extend(_executive_summary(report))
    lines.extend(_data_section(report))
    lines.extend(_performance_section(report, base_dir))
    lines.extend(_recommendation_section(report, base_dir))
    lines.extend(_sensitivity_section(report, base_dir))
    lines.extend(_due_diligence_section(report, base_dir))

    if report.notes:
        lines.append("## Notes")
        lines.append("")
        for note in report.notes:
            lines.append(f"- {note}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _executive_summary(report: PolicyReport) -> list[str]:
    rec = report.recommendation
    economics = report.economics
    norm_str = (
        f"{report.normalized_qini_score:+.4f}"
        if not np.isnan(report.normalized_qini_score)
        else "n/a"
    )
    if rec.n_selected == 0:
        policy_summary = (
            "The profit-optimal policy is to **contact no customers** "
            "under the supplied economics "
            f"({economics.currency} {economics.cost_per_contact:.4f}/contact, "
            f"{economics.currency} {economics.value_per_conversion:.2f}/conversion). "
            "Every evaluated contact policy has non-positive predicted "
            "net profit, so the recommended policy predicts zero "
            "incremental conversions and zero net profit."
        )
    else:
        policy_summary = (
            f"Targeting the **top {rec.fraction:.2%}** of customers by predicted "
            f"uplift ({rec.n_selected:,} customers, cut-off score "
            f"`{rec.threshold_score:+.4f}`) is the profit-optimal contact policy "
            f"under the supplied economics ({economics.currency} "
            f"{economics.cost_per_contact:.4f}/contact, {economics.currency} "
            f"{economics.value_per_conversion:.2f}/conversion). At this cut-off "
            f"the model predicts **{economics.currency} "
            f"{rec.predicted_net_profit:,.2f}** of incremental net profit "
            f"({rec.predicted_incremental_conversions:.1f} incremental "
            f"conversions)."
        )
    return [
        "## Executive summary",
        "",
        policy_summary,
        "",
        f"The model's normalized Qini score is **{norm_str}** on this "
        "evaluation split (see *Performance* below for the reading guide). "
        "Treat the predicted profit as the model's *belief* about the "
        "campaign — realised profit will diverge whenever the model is "
        "miscalibrated. The companion model card lists the causal "
        "assumptions every number above is contingent on.",
        "",
    ]


def _data_section(report: PolicyReport) -> list[str]:
    return [
        "## Campaign and data",
        "",
        f"- Source: **{report.dataset_source}**",
        f"- Sample size on the evaluation split: **{report.n_total:,}** "
        f"({report.n_treated:,} treated · {report.n_control:,} control)",
        f"- Naive ATE (treated-control conversion-rate gap): "
        f"`{report.naive_ate:+.4f}`",
        "",
    ]


def _performance_section(report: PolicyReport, base_dir: Path | None) -> list[str]:
    norm_str = (
        f"{report.normalized_qini_score:+.4f}"
        if not np.isnan(report.normalized_qini_score)
        else "n/a"
    )
    out = [
        "## Performance",
        "",
        "| Metric | Value | Reading |",
        "| --- | ---: | --- |",
        f"| Naive ATE | {report.naive_ate:+.4f} | "
        "Treated minus control conversion rate. Valid under randomization. |",
        f"| Area under Qini (AUQ) | {report.auq:+.4f} | "
        "Cumulative incremental gain integrated over targeted fractions. |",
        f"| Area under uplift curve (AUUC) | {report.auuc:+.4f} | "
        "Treatment-share-adjusted cumulative gain; scales with sample size. |",
        f"| Normalized Qini | {norm_str} | "
        "AUQ relative to a signed-outcome reference; can be negative or above 1. |",
        "",
        "Quote these against bootstrap confidence intervals "
        "(see NB05) — point estimates without CIs over-promise.",
        "",
    ]
    qini_path = report.figure_paths.get("qini")
    decile_path = report.figure_paths.get("decile")
    if qini_path is not None:
        out.append(f"![Qini curve]({_embed_path(qini_path, base_dir)})")
        out.append("")
    if decile_path is not None:
        out.append(f"![Observed uplift by decile]({_embed_path(decile_path, base_dir)})")
        out.append("")
    return out


def _recommendation_section(report: PolicyReport, base_dir: Path | None) -> list[str]:
    rec = report.recommendation
    economics = report.economics
    cutoff = f"`{rec.threshold_score:+.4f}`" if rec.threshold_score is not None else "n/a"
    out = [
        "## Profit-optimal recommendation",
        "",
        f"Economics assumed: **{economics.currency} {economics.cost_per_contact:.4f}** "
        f"per contact, **{economics.currency} {economics.value_per_conversion:.2f}** "
        f"per conversion (break-even uplift = `{economics.break_even_uplift:.4f}`).",
        "",
        f"- Recommended top fraction: **{rec.fraction:.4f}**",
        f"- Customers to contact: **{rec.n_selected:,}**",
        f"- Cut-off uplift score: {cutoff}",
        f"- Predicted incremental conversions: **{rec.predicted_incremental_conversions:.2f}**",
        f"- Predicted net profit: **{economics.currency} {rec.predicted_net_profit:,.2f}**",
        "",
    ]
    profit_path = report.figure_paths.get("profit")
    if profit_path is not None:
        out.append(f"![Profit curve and optimum]({_embed_path(profit_path, base_dir)})")
        out.append("")
    return out


def _sensitivity_section(report: PolicyReport, base_dir: Path | None) -> list[str]:
    if report.scenarios_table is None or report.scenarios_table.empty:
        return []
    table = report.scenarios_table
    columns = [
        c
        for c in [
            "policy",
            "scenario",
            "optimal_fraction",
            "n_selected",
            "predicted_net_profit",
        ]
        if c in table.columns
    ]
    out = [
        "## Sensitivity to economics",
        "",
        "Cost per contact and value per conversion are guesses until the "
        "campaign owner agrees them. The table below sweeps the supplied "
        "scenarios; the row that flips the recommendation is the one to "
        "escalate.",
        "",
    ]
    out.append(_render_markdown_table(table[columns]))
    out.append("")
    scenarios_path = report.figure_paths.get("scenarios")
    if scenarios_path is not None:
        out.append(f"![Scenario sensitivity]({_embed_path(scenarios_path, base_dir)})")
        out.append("")
    return out


def _due_diligence_section(report: PolicyReport, base_dir: Path | None) -> list[str]:
    out = [
        "## Due diligence",
        "",
        "Causal assumptions, randomization diagnostics, and named misuse "
        "risks are documented separately in the **model card**. A "
        "deployment reviewer should sign off the model card before this "
        "recommendation goes live.",
        "",
    ]
    if report.model_card_path is not None:
        out.append(
            f"- Model card: [`{report.model_card_path.name}`]"
            f"({_embed_path(report.model_card_path, base_dir)})"
        )
        out.append("")
    return out


def _embed_path(path: Path, base_dir: Path | None) -> str:
    """Render a path as a markdown-embeddable string, relative when possible."""

    if base_dir is None:
        return path.as_posix()
    return _relative(path, base_dir)


def _render_markdown_table(table: pd.DataFrame) -> str:
    """Render a small DataFrame as a GFM markdown table (no tabulate dep)."""

    if table.empty:
        return "_(no rows)_"
    headers = list(table.columns)
    rows = [
        [_format_cell(value) for value in record]
        for record in table.itertuples(index=False, name=None)
    ]
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(_alignment(value) for value in rows[0]) + " |"
    body_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, separator, *body_lines])


def _format_cell(value: object) -> str:
    if isinstance(value, float) and not np.isnan(value):
        if abs(value) >= 1000 or (value != 0 and abs(value) < 0.001):
            return f"{value:,.4g}"
        return f"{value:,.4f}"
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value)


def _alignment(value: str) -> str:
    """Right-align numeric-looking cells, left-align everything else."""

    stripped = value.replace(",", "").replace("-", "").replace(".", "").replace("+", "")
    return "---:" if stripped.isdigit() else "---"
