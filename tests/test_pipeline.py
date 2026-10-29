"""End-to-end regression tests for the configured uplift pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from uplift_marketing.config import ProjectConfig
from uplift_marketing.data import generate_synthetic_uplift_data
from uplift_marketing.economics import CampaignEconomics
from uplift_marketing.evaluation import policy_value_at_fraction
from uplift_marketing.pipeline import run_pipeline
from uplift_marketing.treatment import naive_average_treatment_effect


def test_run_pipeline_honors_custom_treatment_and_outcome_names(tmp_path: Path) -> None:
    """Custom experiment schemas must work from input through report metrics."""
    data = generate_synthetic_uplift_data(n_rows=500, random_state=13).rename(
        columns={"treatment": "assigned", "conversion": "responded"}
    )
    input_path = tmp_path / "custom_campaign.csv"
    data.to_csv(input_path, index=False)
    config = ProjectConfig(
        root_dir=tmp_path,
        treatment_col="assigned",
        outcome_col="responded",
    )

    outputs = run_pipeline(
        input_path,
        config=config,
        model_name="s_learner",
        economics=CampaignEconomics(cost_per_contact=0.05, value_per_conversion=15.0),
    )

    summary_path = outputs["model_summary_path"]
    scored_path = outputs["scored_path"]
    assert isinstance(summary_path, Path)
    assert isinstance(scored_path, Path)
    scored = pd.read_csv(scored_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_ate = naive_average_treatment_effect(
        scored, treatment_col="assigned", outcome_col="responded"
    )
    observed_policy = policy_value_at_fraction(
        scored, treatment_col="assigned", outcome_col="responded", fraction=0.20
    )

    assert summary["naive_ate"] == pytest.approx(expected_ate)
    assert outputs["naive_ate"] == pytest.approx(
        naive_average_treatment_effect(
            data, treatment_col="assigned", outcome_col="responded"
        )
    )
    assert outputs["top_20_observed_uplift"] == pytest.approx(observed_policy["observed_uplift"])
