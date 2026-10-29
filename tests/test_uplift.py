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


def test_attach_uplift_scores_preserves_index_and_allows_negative_uplift() -> None:
    data = pd.DataFrame({"id": [1, 2]}, index=[19, 7])

    scored = attach_uplift_scores(
        data,
        p_control=np.array([0.0, 1.0]),
        p_treated=np.array([1.0, 0.0]),
        uplift_col="incremental_probability",
    )

    assert scored.index.tolist() == [19, 7]
    np.testing.assert_allclose(scored["incremental_probability"], [1.0, -1.0])
    assert "incremental_probability" not in data


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("parameter", ["p_control", "p_treated"])
def test_attach_uplift_scores_rejects_nonfinite_probabilities(
    invalid: float, parameter: str
) -> None:
    values = {"p_control": np.array([0.1, 0.3]), "p_treated": np.array([0.2, 0.4])}
    values[parameter][1] = invalid

    with pytest.raises(ValueError, match=f"{parameter}.*finite"):
        attach_uplift_scores(pd.DataFrame({"id": [1, 2]}), **values)


@pytest.mark.parametrize("invalid", [-0.1, 1.1])
@pytest.mark.parametrize("parameter", ["p_control", "p_treated"])
def test_attach_uplift_scores_rejects_out_of_range_probabilities(
    invalid: float, parameter: str
) -> None:
    values = {"p_control": np.array([0.1, 0.3]), "p_treated": np.array([0.2, 0.4])}
    values[parameter][1] = invalid

    with pytest.raises(ValueError, match=rf"{parameter}.*\[0, 1\]"):
        attach_uplift_scores(pd.DataFrame({"id": [1, 2]}), **values)


@pytest.mark.parametrize("parameter", ["p_control", "p_treated"])
def test_attach_uplift_scores_rejects_nonnumeric_probabilities(parameter: str) -> None:
    values = {"p_control": np.array([0.1, 0.3]), "p_treated": np.array([0.2, 0.4])}
    values[parameter] = np.array(["not-a-number", "0.3"])

    with pytest.raises(ValueError, match=f"{parameter}.*numeric"):
        attach_uplift_scores(pd.DataFrame({"id": [1, 2]}), **values)


@pytest.mark.parametrize("bad_vector", [np.array(0.2), np.array([[0.2], [0.4]])])
@pytest.mark.parametrize("parameter", ["p_control", "p_treated"])
def test_attach_uplift_scores_requires_one_dimensional_probabilities(
    bad_vector: np.ndarray, parameter: str
) -> None:
    values = {"p_control": np.array([0.1, 0.3]), "p_treated": np.array([0.2, 0.4])}
    values[parameter] = bad_vector

    with pytest.raises(ValueError, match=f"{parameter}.*one-dimensional"):
        attach_uplift_scores(pd.DataFrame({"id": [1, 2]}), **values)


@pytest.mark.parametrize("parameter", ["p_control", "p_treated"])
def test_attach_uplift_scores_rejects_mismatched_lengths(parameter: str) -> None:
    data = pd.DataFrame({"id": [1, 2]})
    values = {"p_control": np.array([0.1, 0.2]), "p_treated": np.array([0.3, 0.4])}
    values[parameter] = np.array([0.2])

    with pytest.raises(ValueError, match=f"{parameter}.*same length"):
        attach_uplift_scores(data, **values)


@pytest.mark.parametrize("reserved", ["p_control", "p_treated"])
def test_attach_uplift_scores_rejects_reserved_uplift_column(reserved: str) -> None:
    data = pd.DataFrame({"id": [1, 2]})

    with pytest.raises(ValueError, match="uplift_col"):
        attach_uplift_scores(
            data, p_control=np.array([0.1, 0.2]), p_treated=np.array([0.3, 0.4]),
            uplift_col=reserved,
        )


def test_make_uplift_deciles_assigns_one_through_ten() -> None:
    df = pd.DataFrame({"uplift_score": np.arange(1000) / 1000.0})
    deciles = make_uplift_deciles(df)
    assert set(deciles.unique()) == set(range(1, 11))


def test_make_uplift_deciles_preserves_score_ties_and_row_order_invariance() -> None:
    data = pd.DataFrame({"uplift_score": [0.9] * 20 + [0.5] * 40 + [0.1] * 40})

    original = make_uplift_deciles(data)
    reordered = make_uplift_deciles(data.sample(frac=1.0, random_state=7))

    pd.testing.assert_series_equal(original, reordered.sort_index())
    assert set(original.unique()) == {1, 2, 3}
    assert original.groupby(data["uplift_score"]).nunique().eq(1).all()
    assert original.iloc[0] == 1  # Highest scores get the first occupied bin.


def test_make_uplift_deciles_keeps_constant_scores_in_one_group() -> None:
    data = pd.DataFrame({"uplift_score": [0.1] * 30})

    assert set(make_uplift_deciles(data).unique()) == {1}
    assert set(make_uplift_deciles(data, n_deciles=5).unique()) == {1}


def test_make_uplift_deciles_supports_custom_bin_count() -> None:
    data = pd.DataFrame({"uplift_score": np.arange(100, dtype=float)})

    assert set(make_uplift_deciles(data, n_deciles=5).unique()) == {1, 2, 3, 4, 5}


def test_make_uplift_deciles_accepts_numeric_scores_as_strings() -> None:
    data = pd.DataFrame({"uplift_score": ["0.9", "0.5", "0.1"]})

    assert make_uplift_deciles(data).tolist() == [1, 2, 3]


@pytest.mark.parametrize("n_deciles", [0, 1, 3.5])
def test_make_uplift_deciles_rejects_invalid_bin_count(n_deciles: int) -> None:
    with pytest.raises(ValueError, match="n_deciles"):
        make_uplift_deciles(pd.DataFrame({"uplift_score": [0.9, 0.1]}), n_deciles=n_deciles)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_make_uplift_deciles_rejects_nonfinite_scores(invalid: float) -> None:
    data = pd.DataFrame({"uplift_score": [0.3, invalid, 0.1]})

    with pytest.raises(ValueError, match="finite"):
        make_uplift_deciles(data)


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


@pytest.mark.parametrize("outcome", [0, 1])
def test_subgroup_ci_retains_uncertainty_for_identical_boundary_rates(outcome: int) -> None:
    # Both arms at 0% or 100% conversions should not imply certainty about
    # the true difference, even though the plug-in Wald standard error is 0.
    n_per_arm = 40
    data = pd.DataFrame(
        {
            "segment": ["boundary"] * (2 * n_per_arm),
            "treatment": [1, 0] * n_per_arm,
            "conversion": [outcome] * (2 * n_per_arm),
        }
    )

    interval = subgroup_uplift_with_ci(data, group_col="segment").iloc[0]
    z_squared = 1.959963984540054**2
    expected_width = z_squared / (n_per_arm + z_squared)

    assert interval["observed_uplift"] == 0.0
    assert interval["ci_low"] == pytest.approx(-expected_width)
    assert interval["ci_high"] == pytest.approx(expected_width)


def test_subgroup_ci_handles_perfect_separation_without_zero_width() -> None:
    n_per_arm = 40
    data = pd.DataFrame(
        {
            "segment": ["separated"] * (2 * n_per_arm),
            "treatment": [1, 0] * n_per_arm,
            "conversion": [1, 0] * n_per_arm,
        }
    )

    positive = subgroup_uplift_with_ci(data, group_col="segment").iloc[0]
    negative = subgroup_uplift_with_ci(
        data.assign(treatment=1 - data["treatment"]), group_col="segment"
    ).iloc[0]
    z_squared = 1.959963984540054**2
    wilson_width = z_squared / (n_per_arm + z_squared)

    assert positive["observed_uplift"] == 1.0
    assert positive["ci_low"] == pytest.approx(1.0 - np.sqrt(2.0) * wilson_width)
    assert positive["ci_high"] == pytest.approx(1.0)
    # Reversing treatment and control should negate and reverse the interval.
    assert negative["observed_uplift"] == -1.0
    assert negative["ci_low"] == pytest.approx(-positive["ci_high"])
    assert negative["ci_high"] == pytest.approx(-positive["ci_low"])


def test_subgroup_ci_widens_with_confidence_level() -> None:
    data = pd.DataFrame(
        {
            "segment": ["mixed"] * 200,
            "treatment": [1] * 100 + [0] * 100,
            "conversion": [1] * 35 + [0] * 65 + [1] * 20 + [0] * 80,
        }
    )

    lower_confidence = subgroup_uplift_with_ci(
        data, group_col="segment", confidence=0.90
    ).iloc[0]
    higher_confidence = subgroup_uplift_with_ci(
        data, group_col="segment", confidence=0.99
    ).iloc[0]

    assert lower_confidence["observed_uplift"] == pytest.approx(0.15)
    assert higher_confidence["ci_low"] < lower_confidence["ci_low"]
    assert higher_confidence["ci_high"] > lower_confidence["ci_high"]


@pytest.mark.parametrize("invalid", [0.5, -1.0, 2.0, np.nan, np.inf, "unknown"])
def test_group_uplift_metrics_reject_invalid_treatment(invalid: float | str) -> None:
    data = pd.DataFrame(
        {
            "segment": ["a", "a", "b", "b"],
            "treatment": [1, 0, 1, invalid],
            "conversion": [1, 0, 0, 1],
        }
    )

    # Bad labels in any segment must not disappear from treatment counts.
    for metric in (uplift_by_group, subgroup_uplift_with_ci):
        with pytest.raises(ValueError, match="treatment.*binary 0/1"):
            metric(data, group_col="segment")


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf, "not-numeric"])
def test_group_uplift_metrics_reject_nonfinite_outcomes(invalid: float | str) -> None:
    data = pd.DataFrame(
        {
            "segment": ["a", "a", "b", "b"],
            "treatment": [1, 0, 1, 0],
            "conversion": [1.0, 0.0, 0.0, invalid],
        }
    )

    for metric in (uplift_by_group, subgroup_uplift_with_ci):
        with pytest.raises(ValueError, match="outcome.*finite"):
            metric(data, group_col="segment")


def test_group_uplift_metrics_accept_numeric_string_labels() -> None:
    treatment = np.tile(["1", "0"], 80)
    outcomes = np.tile([1, 0, 0, 0, 1, 1, 0, 0], 20)
    data = pd.DataFrame(
        {"segment": ["north"] * 160, "treatment": treatment, "conversion": outcomes}
    )

    observed = uplift_by_group(data, group_col="segment")
    confidence = subgroup_uplift_with_ci(data, group_col="segment", min_arm_size=5)

    assert observed.loc[0, "treated_n"] == 80
    assert observed.loc[0, "control_n"] == 80
    assert observed.loc[0, "observed_uplift"] == pytest.approx(0.25)
    assert confidence.loc[0, "observed_uplift"] == pytest.approx(0.25)
    assert np.isfinite(confidence.loc[0, ["ci_low", "ci_high"]].to_numpy(dtype=float)).all()


def test_group_uplift_allows_continuous_outcomes_but_binary_ci_rejects_them() -> None:
    data = pd.DataFrame(
        {
            "segment": ["a"] * 80,
            "treatment": [1, 0] * 40,
            "conversion": [2.5, 0.5] * 40,
        }
    )

    assert uplift_by_group(data, group_col="segment").loc[0, "observed_uplift"] == 2.0
    with pytest.raises(ValueError, match="outcome.*binary 0/1"):
        subgroup_uplift_with_ci(data, group_col="segment", min_arm_size=5)


def test_group_uplift_metrics_reject_missing_groups() -> None:
    data = pd.DataFrame(
        {
            "segment": ["a", "a", "b", None],
            "treatment": [1, 0, 1, 0],
            "conversion": [1, 0, 1, 0],
        }
    )

    for metric in (uplift_by_group, subgroup_uplift_with_ci):
        with pytest.raises(ValueError, match="group.*missing"):
            metric(data, group_col="segment")


def test_group_uplift_metrics_reject_empty_data() -> None:
    data = pd.DataFrame(columns=["segment", "treatment", "conversion"])

    for metric in (uplift_by_group, subgroup_uplift_with_ci):
        with pytest.raises(ValueError, match="data must not be empty"):
            metric(data, group_col="segment")


def test_group_uplift_metrics_preserve_unidentified_single_arm_groups() -> None:
    data = pd.DataFrame(
        {
            "segment": ["treated"] * 2 + ["control"] * 2 + ["both"] * 2,
            "treatment": [1, 1, 0, 0, 1, 0],
            "conversion": [1, 0, 1, 0, 1, 0],
        }
    )

    observed = uplift_by_group(data, group_col="segment").set_index("segment")
    confidence = subgroup_uplift_with_ci(
        data, group_col="segment", min_arm_size=2
    ).set_index("segment")

    for group in ("treated", "control"):
        assert np.isnan(observed.loc[group, "observed_uplift"])
        assert np.isnan(confidence.loc[group, "observed_uplift"])
        assert np.isnan(confidence.loc[group, "ci_low"])
    assert observed.loc["both", "observed_uplift"] == pytest.approx(1.0)
    assert confidence.loc["both", "observed_uplift"] == pytest.approx(1.0)


@pytest.mark.parametrize("min_arm_size", [0, -1, 1.5, np.inf, True])
def test_subgroup_uplift_ci_rejects_invalid_min_arm_size(min_arm_size: float) -> None:
    data = pd.DataFrame(
        {"segment": ["a"] * 4, "treatment": [1, 0, 1, 0], "conversion": [1, 0, 1, 0]}
    )

    with pytest.raises(ValueError, match="min_arm_size"):
        subgroup_uplift_with_ci(data, group_col="segment", min_arm_size=min_arm_size)
