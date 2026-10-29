from __future__ import annotations

import numpy as np
import pandas as pd

from uplift_marketing.economics import CampaignEconomics, optimize_budget
from uplift_marketing.model_card import (
    ModelCard,
    build_model_card,
    write_model_card,
)


def _scored_fixture(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    treatment = rng.binomial(1, 0.5, size=n)
    score = rng.normal(loc=0.0, scale=0.05, size=n)
    p = np.clip(0.10 + treatment * np.maximum(score, 0) * 4, 0, 1)
    conversion = rng.binomial(1, p)
    return pd.DataFrame(
        {
            "customer_id": np.arange(n),
            "feature_a": rng.normal(size=n),
            "feature_b": rng.choice(["x", "y"], size=n),
            "treatment": treatment,
            "conversion": conversion,
            "uplift_score": score,
        }
    )


def test_build_model_card_populates_core_fields() -> None:
    scored = _scored_fixture()
    economics = CampaignEconomics(cost_per_contact=0.10, value_per_conversion=20.0)
    rec = optimize_budget(scored, economics)
    card = build_model_card(
        scored,
        economics=economics,
        recommendation=rec,
        model_name="t-learner",
        dataset_source="synthetic",
        extra_excluded=["customer_id"],
    )
    assert isinstance(card, ModelCard)
    assert card.n_total == len(scored)
    assert card.n_treated + card.n_control == card.n_total
    assert card.recommendation is rec
    assert card.randomization is not None
    # feature_columns should exclude the treatment, outcome, score, and customer_id
    assert "treatment" not in card.feature_columns
    assert "conversion" not in card.feature_columns
    assert "uplift_score" not in card.feature_columns
    assert "customer_id" not in card.feature_columns
    assert "feature_a" in card.feature_columns
    assert "feature_b" in card.feature_columns


def test_build_model_card_without_recommendation_or_randomization() -> None:
    scored = _scored_fixture(n=300)
    economics = CampaignEconomics(cost_per_contact=0.10, value_per_conversion=20.0)
    card = build_model_card(
        scored,
        economics=economics,
        recommendation=None,
        run_randomization_check=False,
    )
    assert card.recommendation is None
    assert card.randomization is None
    md = card.to_markdown()
    assert "Randomization check not run" in md
    assert "No profit-optimal recommendation" in md


def test_model_card_markdown_contains_required_sections() -> None:
    scored = _scored_fixture()
    economics = CampaignEconomics(cost_per_contact=0.10, value_per_conversion=20.0)
    rec = optimize_budget(scored, economics)
    card = build_model_card(
        scored,
        economics=economics,
        recommendation=rec,
        notes=["Card describes the calibrated variant."],
    )
    md = card.to_markdown()
    for header in [
        "## Intended use",
        "## Data",
        "## Causal assumptions",
        "### A.1 Random treatment assignment",
        "### A.2 SUTVA / no interference",
        "### A.3 Consistency",
        "### A.4 Positivity",
        "## Performance",
        "## Profit-optimal recommendation",
        "## Misuse risks",
        "## Provenance",
        "## Notes",
    ]:
        assert header in md, f"missing section: {header}"
    # Misuse risks must call out the most-cited classes of harm.
    for risk in [
        "protected attributes",
        "Self-fulfilling prophecies",
        "out of distribution",
    ]:
        assert risk in md, f"misuse-risk fragment missing: {risk}"


def test_model_card_flags_a_perfectly_separated_covariate() -> None:
    """A fully confounded feature should be visible in the published model card."""
    scored = _scored_fixture()
    scored["assignment_proxy"] = scored["treatment"].astype(float)

    card = build_model_card(
        scored,
        economics=CampaignEconomics(cost_per_contact=0.1, value_per_conversion=20.0),
    )

    assert card.randomization is not None
    assert "assignment_proxy" in set(card.randomization.imbalanced_features["feature"])
    markdown = card.to_markdown()
    assert "`assignment_proxy`" in markdown
    assert "+inf" in markdown


def test_model_card_balance_ignores_model_outputs_and_identifiers() -> None:
    """Derived predictions and contact decisions are not baseline covariates."""
    scored = _scored_fixture(n=100)
    assigned = scored["treatment"].to_numpy()
    scored["assignment_proxy"] = assigned.astype(float)
    # Deliberately make these *post-model* outputs treatment-dependent; they
    # would produce false balance alarms if treated as original features.
    scored["p_control"] = assigned.astype(float)
    scored["p_treated"] = 1.0 - assigned
    scored["uplift_decile"] = np.where(assigned == 1, 1, 10)
    scored["campaign_segment"] = np.where(assigned == 1, "persuadable", "review")
    scored["in_recommendation"] = assigned
    scored["recommended_action"] = np.where(assigned == 1, "contact", "skip")
    scored["session_id"] = np.arange(len(scored))

    card = build_model_card(
        scored,
        economics=CampaignEconomics(cost_per_contact=0.1, value_per_conversion=20.0),
    )

    assert card.randomization is not None
    balance_features = set(card.randomization.balance["feature"])
    assert "assignment_proxy" in balance_features
    assert "assignment_proxy" in set(card.randomization.imbalanced_features["feature"])
    assert "feature_a" in balance_features
    assert "feature_b_x" in balance_features
    for derived_column in (
        "customer_id",
        "session_id",
        "p_control",
        "p_treated",
        "uplift_decile",
        "campaign_segment",
        "in_recommendation",
        "recommended_action",
    ):
        assert not any(
            name == derived_column or name.startswith(f"{derived_column}_")
            for name in balance_features
        )
    assert set(card.feature_columns) == {"feature_a", "feature_b", "assignment_proxy"}


def test_model_card_rejects_missing_columns() -> None:
    scored = _scored_fixture().drop(columns=["uplift_score"])
    economics = CampaignEconomics(cost_per_contact=0.10, value_per_conversion=20.0)
    try:
        build_model_card(scored, economics=economics)
    except ValueError as exc:
        assert "uplift_score" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing uplift_score column")


def test_write_model_card_writes_markdown(tmp_path) -> None:  # type: ignore[no-untyped-def]
    scored = _scored_fixture(n=400)
    economics = CampaignEconomics(cost_per_contact=0.10, value_per_conversion=20.0)
    rec = optimize_budget(scored, economics)
    card = build_model_card(scored, economics=economics, recommendation=rec)
    target = tmp_path / "nested" / "model_card.md"
    written = write_model_card(card, target)
    assert written == target
    assert target.exists()
    content = target.read_text(encoding="utf-8")
    assert content.startswith("# Uplift Model Card")
    assert "## Misuse risks" in content
