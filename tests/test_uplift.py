from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from uplift_marketing.uplift import (
    attach_uplift_scores,
    make_uplift_deciles,
    subgroup_uplift_with_ci,
    uplift_by_group,
)


def test_attach_uplift_scores_adds_three_columns() -> None:
    data = pd.DataFrame({"id": [1, 2, 3]})
    p_c = np.array([0.1, 0.2, 0.3])
    p_t = np.array([0.4, 0.5, 0.6])
    scored = attach_uplift_scores(data, p_control=p_c, p_treated=p_t)
    assert {"p_control", "p_treated", "uplift_score"}.issubset(scored.columns)
    assert scored["uplift_score"].iloc[0] == pytest.approx(0.3)


def test_make_uplift_deciles_assigns_one_through_ten() -> None:
    df = pd.DataFrame({"uplift_score": np.arange(1000) / 1000.0})
    deciles = make_uplift_deciles(df)
    assert set(deciles.unique()) == set(range(1, 11))


def test_uplift_by_group_returns_per_group_rows(synthetic_data) -> None:  # type: ignore[no-untyped-def]
    table = uplift_by_group(synthetic_data, group_col="device")
    assert {"device", "n", "treated_n", "control_n", "observed_uplift"}.issubset(table.columns)
    assert len(table) == synthetic_data["device"].nunique()


def test_subgroup_uplift_with_ci_brackets_observed(synthetic_data) -> None:  # type: ignore[no-untyped-def]
    table = subgroup_uplift_with_ci(synthetic_data, group_col="device", min_arm_size=10)
    assert {"observed_uplift", "ci_low", "ci_high"}.issubset(table.columns)
    populated = table.dropna(subset=["ci_low"])
    assert (populated["ci_low"] <= populated["observed_uplift"]).all()
    assert (populated["observed_uplift"] <= populated["ci_high"]).all()


def test_subgroup_uplift_with_ci_returns_nan_below_min_arm_size() -> None:
    # Two of the three groups have only one row in their treated arm; the
    # CI should be NaN there but the point estimate should still be present.
    df = pd.DataFrame(
        {
            "device": ["a"] * 200 + ["b"] * 4 + ["c"] * 200,
            "treatment": [0, 1] * 100 + [0, 1, 0, 1] + [0, 1] * 100,
            "conversion": list(np.random.default_rng(0).binomial(1, 0.2, 404)),
        }
    )
    table = subgroup_uplift_with_ci(df, group_col="device", min_arm_size=30)
    by_device = table.set_index("device")
    assert np.isnan(by_device.loc["b", "ci_low"])
    assert not np.isnan(by_device.loc["a", "ci_low"])
    assert not np.isnan(by_device.loc["c", "ci_low"])
