from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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


def test_no_positive_uplift_does_not_create_persuadables() -> None:
    """No predicted incrementality means no customers qualify as persuadable."""
    scored = pd.DataFrame(
        {
            "p_control": [0.10, 0.60, 0.60],
            "p_treated": [0.10, 0.60, 0.50],
            "uplift_score": [0.0, 0.0, -0.10],
        }
    )

    segments = assign_campaign_segments(scored)

    assert segments.tolist() == ["review", "sure_thing", "do_not_disturb"]
    assert "persuadable" not in segments.values


def test_negative_uplift_never_overridden_by_conversion_probability() -> None:
    """High baseline conversion cannot justify targeting a harmful intervention."""
    scored = pd.DataFrame(
        {
            "p_control": [0.90, 0.09],
            "p_treated": [0.80, 0.01],
            "uplift_score": [-0.10, -0.08],
        }
    )

    segments = assign_campaign_segments(scored, high_uplift_threshold=0.0)

    assert segments.tolist() == ["do_not_disturb", "do_not_disturb"]


def test_zero_uplift_not_persuadable_when_threshold_is_zero() -> None:
    scored = pd.DataFrame(
        {
            "p_control": [0.10, 0.10],
            "p_treated": [0.10, 0.11],
            "uplift_score": [0.0, 0.01],
        }
    )

    segments = assign_campaign_segments(scored, high_uplift_threshold=0.0)

    assert segments.tolist() == ["review", "persuadable"]


@pytest.mark.parametrize(
    ("column", "bad_value"),
    [
        ("uplift_score", np.nan),
        ("uplift_score", np.inf),
        ("p_control", np.nan),
        ("p_control", -0.1),
        ("p_treated", 1.1),
        ("p_treated", np.inf),
    ],
)
def test_segment_assignment_rejects_invalid_model_outputs(column: str, bad_value: float) -> None:
    scored = pd.DataFrame(
        {"uplift_score": [0.2, 0.01], "p_control": [0.1, 0.5], "p_treated": [0.3, 0.51]}
    )
    scored.loc[0, column] = bad_value

    with pytest.raises(ValueError, match="finite|probabilities"):
        assign_campaign_segments(scored)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"high_uplift_threshold": np.nan}, "high_uplift_threshold"),
        ({"high_uplift_threshold": -0.01}, "high_uplift_threshold"),
        ({"low_probability_threshold": -0.01}, "probability thresholds"),
        ({"high_probability_threshold": np.inf}, "probability thresholds"),
        ({"low_probability_threshold": 0.6, "high_probability_threshold": 0.4},
         "probability thresholds"),
    ],
)
def test_segment_assignment_rejects_invalid_thresholds(
    kwargs: dict[str, float], message: str
) -> None:
    scored = pd.DataFrame(
        {"uplift_score": [0.2], "p_control": [0.1], "p_treated": [0.3]}
    )

    with pytest.raises(ValueError, match=message):
        assign_campaign_segments(scored, **kwargs)
