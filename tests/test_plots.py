from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend for tests

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uplift_marketing.economics import (
    CampaignEconomics,
    optimize_budget,
    scenario_table,
)
from uplift_marketing.plots import (
    PALETTE,
    export_campaign_figures,
    fig_profit_curve,
    fig_qini_curve,
    fig_scenarios,
    fig_uplift_by_decile,
    fig_uplift_curve,
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
            "treatment": treatment,
            "conversion": conversion,
            "uplift_score": score,
        }
    )


def test_fig_qini_curve_renders() -> None:
    fig = fig_qini_curve(_scored_fixture())
    assert isinstance(fig, plt.Figure)
    ax = fig.axes[0]
    # The curve should have at least the model + random baseline.
    assert len(ax.lines) >= 2
    for line in ax.lines[:2]:
        assert line.get_xdata()[0] == 0.0
        assert line.get_ydata()[0] == 0.0
    plt.close(fig)


def test_fig_profit_curve_marks_optimum() -> None:
    scored = _scored_fixture()
    economics = CampaignEconomics(cost_per_contact=0.10, value_per_conversion=20.0)
    rec = optimize_budget(scored, economics)
    fig = fig_profit_curve(rec, economics)
    ax = fig.axes[0]
    # Scatter for the optimum should land at the recommended fraction.
    scatter_offsets = ax.collections[0].get_offsets()
    assert scatter_offsets.shape == (1, 2)
    assert scatter_offsets[0, 0] == rec.fraction
    plt.close(fig)


def test_fig_uplift_by_decile_returns_figure() -> None:
    fig = fig_uplift_by_decile(_scored_fixture())
    ax = fig.axes[0]
    bars = [p for p in ax.patches if p.get_width() > 0]
    # Up to 10 deciles, at least 2 observed.
    assert 2 <= len(bars) <= 10
    plt.close(fig)


def test_fig_uplift_curve_returns_figure() -> None:
    fig = fig_uplift_curve(_scored_fixture())
    assert isinstance(fig, plt.Figure)
    assert fig.axes[0].lines[0].get_xdata()[0] == 0.0
    assert fig.axes[0].lines[0].get_ydata()[0] == 0.0
    plt.close(fig)


def test_fig_scenarios_renders_grouped_bars() -> None:
    scored = _scored_fixture()
    rng = np.random.default_rng(1)
    scored = scored.assign(random_score=rng.uniform(size=len(scored)))
    economics = {
        "baseline": CampaignEconomics(cost_per_contact=0.10, value_per_conversion=20.0),
        "expensive": CampaignEconomics(cost_per_contact=0.40, value_per_conversion=20.0),
    }
    table = scenario_table(
        scored, economics, policies={"uplift": "uplift_score", "random": "random_score"}
    )
    fig = fig_scenarios(table, currency="EUR")
    ax = fig.axes[0]
    # 2 scenarios x 2 policies = 4 bars.
    assert len([p for p in ax.patches if p.get_width() > 0]) == 4
    plt.close(fig)


def test_fig_scenarios_rejects_invalid_table() -> None:
    bad = pd.DataFrame({"foo": [1], "bar": [2]})
    try:
        fig_scenarios(bad)
    except ValueError as exc:
        assert "predicted_net_profit" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing required columns")


def test_export_campaign_figures_writes_pngs(tmp_path) -> None:  # type: ignore[no-untyped-def]
    scored = _scored_fixture()
    economics = CampaignEconomics(cost_per_contact=0.10, value_per_conversion=20.0)
    rec = optimize_budget(scored, economics)
    paths = export_campaign_figures(
        scored,
        recommendation=rec,
        economics=economics,
        scenarios_table=None,
        output_dir=tmp_path,
    )
    assert {"qini", "profit", "decile", "uplift_curve"} <= set(paths)
    for label, path in paths.items():
        assert path.exists(), f"{label} not written"
        assert path.suffix == ".png"
        assert path.stat().st_size > 0


def test_export_campaign_figures_includes_scenarios_when_supplied(tmp_path) -> None:  # type: ignore[no-untyped-def]
    scored = _scored_fixture()
    rng = np.random.default_rng(2)
    scored = scored.assign(random_score=rng.uniform(size=len(scored)))
    economics = CampaignEconomics(cost_per_contact=0.10, value_per_conversion=20.0)
    rec = optimize_budget(scored, economics)
    table = scenario_table(
        scored,
        {"baseline": economics},
        policies={"uplift": "uplift_score", "random": "random_score"},
    )
    paths = export_campaign_figures(
        scored,
        recommendation=rec,
        economics=economics,
        scenarios_table=table,
        output_dir=tmp_path,
    )
    assert "scenarios" in paths
    assert paths["scenarios"].exists()


def test_palette_keys_stable() -> None:
    assert {"primary", "secondary", "neutral", "accent"} <= set(PALETTE)
