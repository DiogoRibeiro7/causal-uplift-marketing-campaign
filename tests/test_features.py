from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from uplift_marketing.features import (
    build_preprocessor,
    honest_train_test_split,
    infer_feature_spec,
    split_features_treatment_outcome,
)


def test_infer_feature_spec_excludes_treatment_and_outcome(synthetic_data) -> None:  # type: ignore[no-untyped-def]
    spec = infer_feature_spec(synthetic_data, extra_excluded=["customer_id", "visit"])
    assert "treatment" not in spec.feature_columns
    assert "conversion" not in spec.feature_columns
    assert "customer_id" not in spec.feature_columns
    assert spec.feature_columns


def test_build_preprocessor_fits(synthetic_data) -> None:  # type: ignore[no-untyped-def]
    spec = infer_feature_spec(synthetic_data, extra_excluded=["customer_id", "visit"])
    x, _, _ = split_features_treatment_outcome(synthetic_data, spec)
    transformer = build_preprocessor(spec)
    matrix = transformer.fit_transform(x)
    assert matrix.shape[0] == len(synthetic_data)


def test_honest_train_test_split_preserves_arm_and_outcome_balance(  # type: ignore[no-untyped-def]
    synthetic_data,
) -> None:
    spec = infer_feature_spec(synthetic_data, extra_excluded=["customer_id", "visit"])
    x, treatment, y = split_features_treatment_outcome(synthetic_data, spec)
    x_tr, x_te, t_tr, t_te, y_tr, y_te = honest_train_test_split(
        x, treatment, y,
        treatment=treatment,
        outcome=y,
        test_size=0.30,
        random_state=0,
    )
    assert len(x_tr) + len(x_te) == len(x)
    # Both arms must appear on each side of the split.
    assert set(t_tr.unique()) == {0, 1}
    assert set(t_te.unique()) == {0, 1}
    # Treated share preserved within ~1pp on a 500-row fixture.
    assert abs(t_tr.mean() - t_te.mean()) < 0.02


def test_honest_train_test_split_rejects_non_binary_treatment(synthetic_data) -> None:  # type: ignore[no-untyped-def]
    bad = synthetic_data["treatment"].copy()
    bad.iloc[0] = 2
    with pytest.raises(ValueError, match="binary"):
        honest_train_test_split(synthetic_data, treatment=bad, test_size=0.3, random_state=0)


@pytest.mark.parametrize("invalid", [0.4, np.nan, np.inf, -1.0])
def test_honest_split_rejects_treatment_before_integer_coercion(invalid: float) -> None:
    """Invalid assignments must not be truncated into valid treatment arms."""
    treatment = np.array([0.0, 1.0, invalid, 0.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="treatment.*binary 0/1"):
        honest_train_test_split(np.arange(6), treatment=treatment, test_size=0.5)


def test_honest_split_rejects_nonbinary_and_multidimensional_outcomes() -> None:
    """Joint stratification requires a fully observed binary outcome."""
    treatment = np.array([0, 1, 0, 1, 0, 1])
    outcome = np.array([0.0, 1.0, 0.5, 0.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="outcome.*binary 0/1"):
        honest_train_test_split(np.arange(6), treatment=treatment, outcome=outcome)
    with pytest.raises(ValueError, match="one-dimensional"):
        honest_train_test_split(np.arange(6), treatment=treatment.reshape(-1, 1))


def test_honest_split_falls_back_to_arm_stratification_for_rare_outcomes() -> None:
    """A singleton conversion cell must not make an honest arm split impossible."""
    treatment = np.r_[np.zeros(15, dtype=int), np.ones(15, dtype=int)]
    outcome = np.zeros(30, dtype=int)
    outcome[[0, 15]] = 1

    with pytest.warns(UserWarning, match="treatment-only"):
        tr, te, treated_tr, treated_te = honest_train_test_split(
            np.arange(30), treatment, treatment=treatment, outcome=outcome, test_size=0.3
        )

    assert len(tr) + len(te) == 30
    assert set(np.unique(treated_tr)) == {0, 1}
    assert set(np.unique(treated_te)) == {0, 1}


def test_honest_split_requires_enough_rows_per_treatment_arm() -> None:
    treatment = np.array([0, 0, 0, 0, 0, 1])
    with pytest.raises(ValueError, match="at least two observations"):
        honest_train_test_split(np.arange(6), treatment=treatment, test_size=0.5)


def test_honest_split_rejects_allocation_without_both_arms_in_test() -> None:
    """A highly imbalanced arm may disappear from an otherwise legal split."""
    treatment = np.r_[np.zeros(98, dtype=int), np.ones(2, dtype=int)]
    with pytest.raises(ValueError, match="Both treatment arms must appear"):
        honest_train_test_split(np.arange(100), treatment=treatment, test_size=0.1)


@pytest.mark.parametrize("column", ["treatment", "conversion"])
def test_split_features_rejects_fractional_binary_labels(
    synthetic_data: pd.DataFrame, column: str
) -> None:
    """Feature extraction must not silently truncate the labels used by models."""
    data = synthetic_data.copy()
    data[column] = data[column].astype(float)
    data.loc[data.index[0], column] = 0.4
    spec = infer_feature_spec(data, extra_excluded=["customer_id", "visit"])

    with pytest.raises(ValueError, match="binary 0/1"):
        split_features_treatment_outcome(data, spec)


def test_honest_split_accepts_boolean_treatment_and_outcome() -> None:
    """Boolean experimental flags are valid binary labels."""
    treatment: NDArray[np.bool_] = np.array([True, False] * 20)
    outcome: NDArray[np.bool_] = np.array([True, False, False, True] * 10)
    tr, te = honest_train_test_split(
        np.arange(40), treatment=treatment, outcome=outcome, test_size=0.3
    )
    assert len(tr) + len(te) == 40
