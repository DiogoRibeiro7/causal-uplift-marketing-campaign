from __future__ import annotations

import pandas as pd

from uplift_marketing.policy import assign_campaign_segments, campaign_action_table


def test_assign_campaign_segments_returns_expected_labels() -> None:
    scored = pd.DataFrame(
        {
            "p_control": [0.05, 0.60, 0.02, 0.30],
            "p_treated": [0.30, 0.61, 0.03, 0.20],
            "uplift_score": [0.25, 0.01, 0.01, -0.10],
        }
    )
    segments = assign_campaign_segments(scored, high_uplift_threshold=0.10)
    assert segments.iloc[0] == "persuadable"
    assert segments.iloc[1] == "sure_thing"
    assert segments.iloc[2] == "lost_cause"
    assert segments.iloc[3] == "do_not_disturb"


def test_campaign_action_table_has_actions() -> None:
    scored = pd.DataFrame(
        {
            "campaign_segment": ["persuadable", "sure_thing", "persuadable"],
            "uplift_score": [0.2, 0.01, 0.15],
            "p_control": [0.1, 0.6, 0.2],
            "p_treated": [0.3, 0.61, 0.35],
        }
    )
    table = campaign_action_table(scored)
    assert "recommended_action" in table.columns
    assert "caution" in table.columns
