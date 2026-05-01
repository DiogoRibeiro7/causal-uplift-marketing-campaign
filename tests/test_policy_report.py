from __future__ import annotations

import numpy as np
import pandas as pd

from uplift_marketing.economics import (
    CampaignEconomics,
    optimize_budget,
    scenario_table,
)
from uplift_marketing.policy_report import (
    PolicyReport,
    build_policy_report,
    write_policy_report,
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


def test_build_policy_report_populates_metrics() -> None:
    scored = _scored_fixture()
    economics = CampaignEconomics(cost_per_contact=0.10, value_per_conversion=20.0)
    rec = optimize_budget(scored, economics)
    report = build_policy_report(
        scored,
        economics=economics,
        recommendation=rec,
        model_name="t-learner",
        dataset_source="synthetic",
    )
    assert isinstance(report, PolicyReport)
    assert report.recommendation is rec
    assert report.n_treated + report.n_control == report.n_total
    # Metrics should be populated as floats.
    assert isinstance(report.auq, float)
    assert isinstance(report.auuc, float)


def test_policy_report_markdown_has_required_sections() -> None:
    scored = _scored_fixture()
    economics = CampaignEconomics(cost_per_contact=0.10, value_per_conversion=20.0)
    rec = optimize_budget(scored, economics)
    rng = np.random.default_rng(1)
    scored = scored.assign(random_score=rng.uniform(size=len(scored)))
    scenarios = scenario_table(
        scored,
        {"baseline": economics},
        policies={"uplift": "uplift_score", "random": "random_score"},
    )
    report = build_policy_report(
        scored,
        economics=economics,
        recommendation=rec,
        scenarios_table=scenarios,
        notes=["Drafted from the calibrated T-learner."],
    )
    md = report.to_markdown()
    for header in [
        "# Campaign policy report",
        "## Executive summary",
        "## Campaign and data",
        "## Performance",
        "## Profit-optimal recommendation",
        "## Sensitivity to economics",
        "## Due diligence",
        "## Notes",
    ]:
        assert header in md, f"missing section: {header}"


def test_policy_report_skips_sensitivity_when_no_scenarios() -> None:
    scored = _scored_fixture()
    economics = CampaignEconomics(cost_per_contact=0.10, value_per_conversion=20.0)
    rec = optimize_budget(scored, economics)
    report = build_policy_report(scored, economics=economics, recommendation=rec)
    md = report.to_markdown()
    assert "## Sensitivity to economics" not in md


def test_write_policy_report_writes_markdown(tmp_path) -> None:  # type: ignore[no-untyped-def]
    scored = _scored_fixture()
    economics = CampaignEconomics(cost_per_contact=0.10, value_per_conversion=20.0)
    rec = optimize_budget(scored, economics)
    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()
    figure_paths = {
        "qini": figures_dir / "qini.png",
        "profit": figures_dir / "profit.png",
    }
    for path in figure_paths.values():
        path.write_bytes(b"\x89PNG\r\n\x1a\n")
    model_card_path = tmp_path / "model_card.md"
    model_card_path.write_text("# Model card", encoding="utf-8")
    report = build_policy_report(
        scored,
        economics=economics,
        recommendation=rec,
        figure_paths=figure_paths,
        model_card_path=model_card_path,
    )
    target = tmp_path / "policy_report.md"
    written = write_policy_report(report, target)
    content = written.read_text(encoding="utf-8")
    # Figure paths must be embedded relative to the report's directory.
    assert "(figures/qini.png)" in content
    assert "(figures/profit.png)" in content
    # Model card should be linked relative to the report.
    assert "(model_card.md)" in content


def test_policy_report_rejects_missing_columns() -> None:
    scored = _scored_fixture().drop(columns=["uplift_score"])
    economics = CampaignEconomics(cost_per_contact=0.10, value_per_conversion=20.0)
    rec_scored = _scored_fixture()
    rec = optimize_budget(rec_scored, economics)
    try:
        build_policy_report(scored, economics=economics, recommendation=rec)
    except ValueError as exc:
        assert "uplift_score" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing uplift_score column")
