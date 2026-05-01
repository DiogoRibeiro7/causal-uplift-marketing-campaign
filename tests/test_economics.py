from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from uplift_marketing.economics import (
    BudgetRecommendation,
    CampaignEconomics,
    optimize_budget,
    scenario_table,
)
from uplift_marketing.pipeline import export_dashboard


def test_campaign_economics_break_even() -> None:
    e = CampaignEconomics(cost_per_contact=0.30, value_per_conversion=15.0)
    assert e.break_even_uplift == pytest.approx(0.02)
    assert e.net_profit(100, 1000) == pytest.approx(100 * 15.0 - 1000 * 0.30)
    np.testing.assert_allclose(
        e.net_profit(np.array([10.0, 20.0]), np.array([100.0, 200.0])),
        np.array([10 * 15.0 - 100 * 0.30, 20 * 15.0 - 200 * 0.30]),
    )


def test_campaign_economics_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="cost_per_contact"):
        CampaignEconomics(cost_per_contact=-1.0, value_per_conversion=10.0)
    with pytest.raises(ValueError, match="value_per_conversion"):
        CampaignEconomics(cost_per_contact=0.5, value_per_conversion=0.0)


def _scored_fixture(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    treatment = rng.binomial(1, 0.5, size=n)
    # Conversions correlate with the synthetic uplift score so the optimizer
    # has a real signal to pick up.
    score = rng.normal(loc=0.0, scale=0.05, size=n)
    p = np.clip(0.10 + treatment * np.maximum(score, 0) * 4, 0, 1)
    conversion = rng.binomial(1, p)
    return pd.DataFrame(
        {
            "customer_id": np.arange(n),
            "treatment": treatment,
            "conversion": conversion,
            "uplift_score": score,
        }
    )


def test_optimize_budget_returns_recommendation() -> None:
    scored = _scored_fixture()
    e = CampaignEconomics(cost_per_contact=0.10, value_per_conversion=20.0)
    rec = optimize_budget(scored, e)
    assert isinstance(rec, BudgetRecommendation)
    assert 0 < rec.fraction <= 1
    assert rec.n_selected > 0
    assert len(rec.targets) == rec.n_selected
    assert rec.profit_curve["net_profit"].max() == pytest.approx(rec.predicted_net_profit)
    # Targets should be ordered by score descending.
    diffs = np.diff(rec.targets["uplift_score"].to_numpy())
    assert (diffs <= 1e-9).all()


def test_optimize_budget_rejects_invalid_fractions() -> None:
    scored = _scored_fixture(n=100)
    e = CampaignEconomics(cost_per_contact=0.10, value_per_conversion=20.0)
    with pytest.raises(ValueError, match="fractions"):
        optimize_budget(scored, e, fractions=np.array([0.0, 0.5]))


def test_scenario_table_dimensions() -> None:
    scored = _scored_fixture()
    rng = np.random.default_rng(1)
    scored = scored.assign(random_score=rng.uniform(size=len(scored)))
    economics = {
        "baseline": CampaignEconomics(cost_per_contact=0.10, value_per_conversion=20.0),
        "expensive": CampaignEconomics(cost_per_contact=0.40, value_per_conversion=20.0),
    }
    policies = {"uplift_targeted": "uplift_score", "random": "random_score"}
    table = scenario_table(scored, economics, policies=policies)
    assert len(table) == len(economics) * len(policies)
    assert {"policy", "scenario", "optimal_fraction", "predicted_net_profit"}.issubset(
        table.columns
    )


def test_export_dashboard_writes_expected_files(tmp_path) -> None:  # type: ignore[no-untyped-def]
    scored = _scored_fixture(n=500)
    e = CampaignEconomics(cost_per_contact=0.10, value_per_conversion=20.0)
    paths = export_dashboard(
        scored,
        economics=e,
        output_dir=tmp_path,
        economics_scenarios={"low_value": CampaignEconomics(0.10, 5.0)},
    )
    expected = {"recommended_targets", "profit_curve", "model_summary", "budget_scenarios"}
    assert expected <= set(paths)
    for label, path in paths.items():
        assert path.exists(), f"{label} was not written"
    # Recommended targets are flagged with in_recommendation.
    targets = pd.read_csv(paths["recommended_targets"])
    assert "in_recommendation" in targets.columns
    assert set(targets["in_recommendation"].unique()).issubset({0, 1})
    assert targets["in_recommendation"].sum() > 0
    # Model summary JSON is well-formed and contains the headline metrics.
    summary = json.loads(paths["model_summary"].read_text())
    assert {"auq", "auuc", "normalized_qini", "economics", "recommendation"} <= set(summary)


def test_export_dashboard_skips_scenarios_when_none(tmp_path) -> None:  # type: ignore[no-untyped-def]
    scored = _scored_fixture(n=300)
    e = CampaignEconomics(cost_per_contact=0.10, value_per_conversion=20.0)
    paths = export_dashboard(scored, economics=e, output_dir=tmp_path)
    assert "budget_scenarios" not in paths
