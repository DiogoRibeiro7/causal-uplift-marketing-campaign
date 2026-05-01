"""Model card generation focused on causal assumptions and misuse risks.

A model card is a stakeholder-facing document — the artifact a product
owner, legal reviewer, or campaign approver actually reads before
shipping a recommendation. For an uplift model the highest-stakes
content is *not* the headline metric: it is the list of causal
assumptions the metric is contingent on, and the list of ways the model
is wrong to use.

This module produces both. :func:`build_model_card` collects the
quantitative pieces (sample size, balance diagnostics, AUUC, normalized
Qini, profit-optimal recommendation) and bundles them into a
:class:`ModelCard` value object. :func:`write_model_card` renders the
card to markdown and writes it to disk.
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
from uplift_marketing.treatment import (
    naive_average_treatment_effect,
    validate_randomization,
)

if TYPE_CHECKING:
    from uplift_marketing.economics import BudgetRecommendation, CampaignEconomics
    from uplift_marketing.treatment import RandomizationReport


@dataclass(frozen=True)
class ModelCard:
    """Structured model-card content with a markdown renderer.

    The fields here are deliberately the *inputs* the markdown template
    needs — every section in :meth:`to_markdown` reads from one of them.
    Build one with :func:`build_model_card`; render with
    :meth:`to_markdown` or write directly via :func:`write_model_card`.
    """

    model_name: str
    dataset_source: str
    n_total: int
    n_treated: int
    n_control: int
    treatment_column: str
    outcome_column: str
    feature_columns: list[str]
    naive_ate: float
    auq: float
    auuc: float
    normalized_qini_score: float
    randomization: RandomizationReport | None
    economics: CampaignEconomics
    recommendation: BudgetRecommendation | None
    generated_at: datetime
    notes: list[str] = field(default_factory=list)

    @property
    def observed_treated_share(self) -> float:
        return self.n_treated / self.n_total if self.n_total else float("nan")

    def to_markdown(self) -> str:
        """Render the card to a stakeholder-facing markdown document."""

        return _render_markdown(self)


def build_model_card(
    scored: pd.DataFrame,
    *,
    economics: CampaignEconomics,
    recommendation: BudgetRecommendation | None = None,
    model_name: str = "T-learner",
    dataset_source: str = "unknown",
    feature_columns: list[str] | None = None,
    treatment_col: str = "treatment",
    outcome_col: str = "conversion",
    uplift_col: str = "uplift_score",
    extra_excluded: list[str] | None = None,
    expected_treated_share: float = 0.5,
    smd_threshold: float = 0.10,
    notes: list[str] | None = None,
    run_randomization_check: bool = True,
) -> ModelCard:
    """Assemble a :class:`ModelCard` from a scored DataFrame.

    Computes the metrics, diagnostics, and randomization checks the card
    needs and packages them with the dataset / economics / recommendation
    metadata. Missing optional pieces (e.g. ``recommendation=None`` or
    ``run_randomization_check=False``) cause those sections to render as
    "not provided" rather than raise.

    Parameters
    ----------
    scored:
        DataFrame produced by
        :func:`uplift_marketing.uplift.attach_uplift_scores`, containing
        the binary treatment, the binary outcome, and the per-customer
        uplift score.
    economics:
        Cost / value assumptions used to render the recommendation
        section. Must always be supplied — a model card without an
        explicit economics statement is the wrong default.
    recommendation:
        Output of :func:`uplift_marketing.economics.optimize_budget`. If
        omitted the recommendation section reports that no profit-optimal
        cut-off was provided.
    model_name:
        Display name for the modelling approach.
    dataset_source:
        Free-form description of the data origin (e.g. ``"hillstrom"``,
        ``"synthetic"``, ``"q3_2025_holdout"``).
    feature_columns:
        Columns the model used as features. If omitted, inferred as the
        columns that are not the treatment / outcome / uplift score /
        any ``extra_excluded`` columns.
    treatment_col, outcome_col, uplift_col:
        Column names in ``scored``.
    extra_excluded:
        Columns that are present in ``scored`` but should not be reported
        as features (e.g. ``customer_id``, secondary outcomes).
    expected_treated_share:
        Treated share the randomization check compares against.
    smd_threshold:
        ``|SMD|`` threshold above which a feature is reported as
        imbalanced.
    notes:
        Extra free-form bullets appended to the *Notes* section. Use this
        for context the structured sections cannot capture (e.g. "this
        card describes the calibrated variant, not the vanilla one").
    run_randomization_check:
        When ``True`` (default) runs
        :func:`uplift_marketing.treatment.validate_randomization` and
        embeds the result. Set ``False`` for synthetic datasets where the
        check is uninformative.
    """

    required = {treatment_col, outcome_col, uplift_col}
    missing = required - set(scored.columns)
    if missing:
        raise ValueError(f"scored is missing required columns: {sorted(missing)}")

    treatment = scored[treatment_col].to_numpy()
    outcome = scored[outcome_col].to_numpy()
    uplift_score = scored[uplift_col].to_numpy()

    n_total = int(len(scored))
    n_treated = int((treatment == 1).sum())
    n_control = int((treatment == 0).sum())

    naive_ate = naive_average_treatment_effect(
        scored, treatment_col=treatment_col, outcome_col=outcome_col
    )
    qini = qini_curve(outcome=outcome, treatment=treatment, uplift_score=uplift_score)
    auq = float(area_under_qini(qini))
    auc = uplift_curve(outcome=outcome, treatment=treatment, uplift_score=uplift_score)
    auuc = float(area_under_uplift(auc))
    norm = normalized_qini(
        outcome=outcome, treatment=treatment, uplift_score=uplift_score
    )
    normalized_qini_score = float(norm["normalized"]) if not np.isnan(
        norm["normalized"]
    ) else float("nan")

    randomization: RandomizationReport | None = None
    if run_randomization_check:
        randomization = validate_randomization(
            scored,
            treatment_col=treatment_col,
            outcome_col=outcome_col,
            expected_treated_share=expected_treated_share,
            smd_threshold=smd_threshold,
            extra_excluded=_excluded_for_balance(
                scored, uplift_col, extra_excluded
            ),
        )

    if feature_columns is None:
        excluded = {treatment_col, outcome_col, uplift_col} | set(extra_excluded or [])
        # Strip the columns the pipeline attaches downstream of feature
        # modelling: scores from upstream estimators, propensity-style
        # potential-outcome predictions, decile / segment labels, and the
        # recommendation flag. ID-like columns are also dropped so the
        # feature list reflects what the model actually trained on.
        excluded |= {c for c in scored.columns if c.endswith("_score")}
        excluded |= {c for c in scored.columns if c.endswith("_id") or c == "id"}
        excluded |= {
            "p_control",
            "p_treated",
            "uplift_decile",
            "campaign_segment",
            "in_recommendation",
            "recommended_action",
        }
        feature_columns = [c for c in scored.columns if c not in excluded]

    return ModelCard(
        model_name=model_name,
        dataset_source=dataset_source,
        n_total=n_total,
        n_treated=n_treated,
        n_control=n_control,
        treatment_column=treatment_col,
        outcome_column=outcome_col,
        feature_columns=list(feature_columns),
        naive_ate=float(naive_ate),
        auq=auq,
        auuc=auuc,
        normalized_qini_score=normalized_qini_score,
        randomization=randomization,
        economics=economics,
        recommendation=recommendation,
        generated_at=datetime.now(timezone.utc),
        notes=list(notes or []),
    )


def write_model_card(card: ModelCard, output_path: str | Path) -> Path:
    """Render ``card`` to markdown and write it to ``output_path``.

    The parent directory is created if missing. Returns the path written
    so callers can compose with :func:`uplift_marketing.pipeline.export_dashboard`.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(card.to_markdown(), encoding="utf-8")
    return path


def _excluded_for_balance(
    scored: pd.DataFrame,
    uplift_col: str,
    extra_excluded: list[str] | None,
) -> list[str]:
    """Columns to keep out of the balance check (scores, IDs, derived labels)."""

    excluded = {uplift_col, "uplift_decile", "campaign_segment", "in_recommendation"}
    excluded |= {c for c in scored.columns if c.endswith("_score")}
    excluded |= set(extra_excluded or [])
    return sorted(excluded)


def _render_markdown(card: ModelCard) -> str:
    """Markdown template for a model card. Kept private; use ``card.to_markdown()``."""

    lines: list[str] = []
    lines.append(f"# Uplift Model Card — {card.model_name}")
    lines.append("")
    lines.append(
        f"_Dataset: **{card.dataset_source}** · Generated: "
        f"{card.generated_at:%Y-%m-%d %H:%M UTC}_"
    )
    lines.append("")

    lines.extend(_intended_use_section())
    lines.extend(_data_section(card))
    lines.extend(_causal_assumptions_section(card))
    lines.extend(_performance_section(card))
    lines.extend(_recommendation_section(card))
    lines.extend(_misuse_section())
    lines.extend(_provenance_section(card))

    if card.notes:
        lines.append("## Notes")
        lines.append("")
        for note in card.notes:
            lines.append(f"- {note}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _intended_use_section() -> list[str]:
    return [
        "## Intended use",
        "",
        "This model estimates **incremental treatment effect** (uplift) on a "
        "binary marketing outcome. It is intended to support **profit-optimal "
        "contact selection**: rank customers by predicted uplift, contact the "
        "top fraction whose predicted profit at the supplied economics is "
        "positive, hold the rest back.",
        "",
        "It is *not* designed to estimate baseline conversion probability, "
        "customer lifetime value, churn risk, or any non-causal target. Using "
        "the score for those tasks will be wrong by construction.",
        "",
    ]


def _data_section(card: ModelCard) -> list[str]:
    feature_str = ", ".join(f"`{c}`" for c in card.feature_columns) or "_none_"
    return [
        "## Data",
        "",
        f"- Source: **{card.dataset_source}**",
        f"- Sample size: **{card.n_total:,}** "
        f"({card.n_treated:,} treated · {card.n_control:,} control)",
        f"- Observed treated share: {card.observed_treated_share:.4f}",
        f"- Treatment column: `{card.treatment_column}` (binary 0/1)",
        f"- Outcome column: `{card.outcome_column}` (binary 0/1)",
        f"- Features ({len(card.feature_columns)}): {feature_str}",
        "",
    ]


def _causal_assumptions_section(card: ModelCard) -> list[str]:
    out = [
        "## Causal assumptions",
        "",
        "Every uplift number on this card is contingent on the following "
        "assumptions. Where automated checks exist they are summarised "
        "below; the rest depend on knowledge about the experiment that "
        "this card cannot verify.",
        "",
        "### A.1 Random treatment assignment _(verifiable)_",
        "",
        "Treatment must have been assigned independently of all features and "
        "potential outcomes. Sample-ratio mismatch (SRM) and per-feature "
        "standardized mean differences (SMD) are reported as automated "
        "checks.",
        "",
    ]

    randomization = card.randomization
    if randomization is None:
        out.append("- _Randomization check not run for this card._")
        out.append("")
    else:
        out.append(
            f"- Expected treated share **assumed by this card**: "
            f"`{randomization.expected_treated_share:.4f}` "
            "(supply the experiment's actual design share via "
            "`expected_treated_share` if different)."
        )
        out.append(
            f"- Sample-ratio chi-square: χ²={randomization.sample_ratio_chi2:.2f}, "
            f"p={randomization.sample_ratio_pvalue:.3f}"
        )
        out.append(
            f"  · observed treated share: {randomization.observed_treated_share:.4f}"
        )
        n_imbalanced = len(randomization.imbalanced_features)
        out.append(
            f"- Features with `|SMD| > {randomization.smd_threshold:.2f}`: "
            f"**{n_imbalanced}**"
        )
        if n_imbalanced:
            top = randomization.imbalanced_features.head(5)
            out.append("")
            out.append("  | feature | SMD | mean (treated) | mean (control) |")
            out.append("  | --- | ---: | ---: | ---: |")
            for _, row in top.iterrows():
                out.append(
                    f"  | `{row['feature']}` | {row['smd']:+.3f} | "
                    f"{row['mean_treated']:.4f} | {row['mean_control']:.4f} |"
                )
        n_null = len(randomization.null_share)
        out.append(f"- Columns with missing values: **{n_null}**")
        out.append("")

    out.extend(
        [
            "### A.2 SUTVA / no interference _(unverifiable)_",
            "",
            "We assume one customer's treatment does not affect another "
            "customer's outcome. This is plausible in arm's-length email or "
            "display campaigns and **dubious** wherever customers cluster — "
            "shared inboxes, household accounts, B2B with shared decision "
            "makers, social-graph spillover, or geo-targeted offers. Validate "
            "before deploying in those contexts.",
            "",
            "### A.3 Consistency _(unverifiable)_",
            "",
            "We assume the observed outcome under treatment equals the "
            "potential outcome under treatment — i.e. a single well-defined "
            "treatment. If you intend to deploy a different creative, "
            "channel, or send time than the one used in training, this card "
            "does not transfer; re-validate with a fresh A/B test.",
            "",
            "### A.4 Positivity _(partially verifiable)_",
            "",
            "Every customer in the target population must have nonzero "
            "probability of receiving treatment. Under randomized assignment "
            "this holds by construction. If treatment was conditional on "
            "operational eligibility (channel availability, opt-in status, "
            "regulatory exclusion) the model is only valid on the eligible "
            "subpopulation and must not be scored against ineligible "
            "customers.",
            "",
        ]
    )
    return out


def _performance_section(card: ModelCard) -> list[str]:
    norm_str = (
        f"{card.normalized_qini_score:+.4f}"
        if not np.isnan(card.normalized_qini_score)
        else "n/a"
    )
    return [
        "## Performance",
        "",
        "| Metric | Value | Reading |",
        "| --- | ---: | --- |",
        f"| Naive ATE | {card.naive_ate:+.4f} | "
        "Treated minus control conversion rate. Valid under A.1 only. |",
        f"| Area under Qini (AUQ) | {card.auq:+.4f} | "
        "Cumulative incremental gain integrated over targeted fractions. |",
        f"| Area under uplift curve (AUUC) | {card.auuc:+.4f} | "
        "Population-rescaled cousin of AUQ; comparable across sample sizes. |",
        f"| Normalized Qini | {norm_str} | "
        "AUQ scaled into `[0, 1]` against the in-sample oracle. |",
        "",
        "A model with no signal scores near zero on AUUC and normalized Qini. "
        "Quote these against bootstrap confidence intervals (see NB05) — point "
        "estimates without CIs over-promise.",
        "",
    ]


def _recommendation_section(card: ModelCard) -> list[str]:
    economics = card.economics
    rec = card.recommendation
    out = [
        "## Profit-optimal recommendation",
        "",
        f"Economics assumed: **{economics.currency} {economics.cost_per_contact:.4f}** "
        f"per contact, **{economics.currency} {economics.value_per_conversion:.2f}** "
        f"per conversion (break-even uplift = `{economics.break_even_uplift:.4f}`).",
        "",
    ]
    if rec is None:
        out.append("_No profit-optimal recommendation was supplied to this card._")
        out.append("")
        return out

    out.extend(
        [
            f"- Recommended top fraction: **{rec.fraction:.4f}**",
            f"- Customers to contact: **{rec.n_selected:,}**",
            f"- Cut-off uplift score: `{rec.threshold_score:+.4f}`",
            f"- Predicted incremental conversions: {rec.predicted_incremental_conversions:.2f}",
            f"- Predicted net profit: **{economics.currency} {rec.predicted_net_profit:,.2f}**",
            "",
            "These are the model's **belief** about the campaign's economics. "
            "Realised economics will diverge from predictions whenever the "
            "model is mis-calibrated; see the calibration analysis in NB06 "
            "before quoting predicted profit to stakeholders.",
            "",
        ]
    )
    return out


def _misuse_section() -> list[str]:
    return [
        "## Misuse risks",
        "",
        "The following ways of using this model will at best waste effort "
        "and at worst cause harm. Flag them in any deployment review.",
        "",
        "1. **Targeting on protected attributes (or proxies).** Review the "
        "feature list for variables that can act as proxies for protected "
        "attributes (age, gender, ethnicity, postcode-as-income). An uplift "
        "model trained on such proxies will systematically under- or "
        "over-target protected groups, which is both a legal and "
        "reputational hazard.",
        "2. **Self-fulfilling prophecies.** If you only contact "
        "\"persuadables\" and stop contacting the rest, future training data "
        "loses the variation needed to re-estimate the under-targeted "
        "segments. Maintain a randomized hold-out — a fraction of the "
        "population contacted independently of the model's score — to keep "
        "the next training cycle honest.",
        "3. **Generalising past the training context.** A different "
        "creative, channel, audience, or season is *out of distribution* "
        "for this model. Re-validate with a fresh A/B test before reusing.",
        "4. **Treating point estimates as ground truth.** AUUC, AUQ, and "
        "predicted profit are point estimates. Use the bootstrap CIs (NB05) "
        "as the reference for whether any difference between policies is "
        "real.",
        "5. **Leaking the score into other systems.** The uplift score is a "
        "model artifact, not a customer attribute. Do not pipe it into "
        "recommender systems, pricing engines, or downstream credit / "
        "eligibility decisions.",
        "",
    ]


def _provenance_section(card: ModelCard) -> list[str]:
    return [
        "## Provenance",
        "",
        f"- Generated by `uplift_marketing.model_card.build_model_card` at "
        f"{card.generated_at:%Y-%m-%d %H:%M:%S UTC}.",
        "- Code: `uplift_marketing` package — see `src/uplift_marketing/`.",
        "",
    ]
