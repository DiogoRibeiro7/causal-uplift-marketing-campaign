"""Feature selection and preprocessing for uplift models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, cast
from warnings import warn

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class FeatureSpec:
    """Feature specification used to build preprocessing pipelines."""

    numerical: list[str]
    categorical: list[str]
    excluded: list[str]

    @property
    def feature_columns(self) -> list[str]:
        """Return all feature columns before preprocessing."""

        return self.numerical + self.categorical


def _validated_binary_indicator(
    values: pd.Series | NDArray[Any], *, name: str, require_both: bool = False
) -> NDArray[np.int64]:
    """Return binary labels only after verifying the original observed values.

    Checking before conversion prevents fractional assignments and NaN/Inf
    values from being silently truncated or converted into valid labels.
    """
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    try:
        numeric = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain only binary 0/1 values") from exc
    if not np.isfinite(numeric).all() or not np.isin(numeric, (0.0, 1.0)).all():
        raise ValueError(f"{name} must contain only binary 0/1 values")
    if require_both and set(np.unique(numeric)) != {0.0, 1.0}:
        raise ValueError("Both treatment arms 0 and 1 are required")
    return np.asarray(numeric, dtype=np.int64)


def infer_feature_spec(
    data: pd.DataFrame,
    *,
    treatment_col: str = "treatment",
    outcome_col: str = "conversion",
    extra_excluded: Sequence[str] | None = None,
) -> FeatureSpec:
    """Infer numerical and categorical feature columns.

    Treatment and outcome columns are excluded from the regular feature matrix.
    The S-learner later adds treatment explicitly as a controlled feature.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    excluded = {treatment_col, outcome_col}
    if extra_excluded is not None:
        excluded.update(extra_excluded)

    available = [column for column in data.columns if column not in excluded]
    numerical = [column for column in available if pd.api.types.is_numeric_dtype(data[column])]
    categorical = [column for column in available if column not in numerical]

    return FeatureSpec(numerical=numerical, categorical=categorical, excluded=sorted(excluded))


def build_preprocessor(spec: FeatureSpec) -> ColumnTransformer:
    """Build a scikit-learn preprocessing transformer.

    Numerical features are median-imputed and scaled. Categorical features are
    most-frequent imputed and one-hot encoded.
    """

    if not spec.feature_columns:
        raise ValueError("At least one feature column is required")

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, spec.numerical),
            ("cat", categorical_pipeline, spec.categorical),
        ],
        remainder="drop",
    )


def split_features_treatment_outcome(
    data: pd.DataFrame,
    spec: FeatureSpec,
    *,
    treatment_col: str = "treatment",
    outcome_col: str = "conversion",
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Split a frame into features, treatment, and outcome."""

    missing = [
        column
        for column in [*spec.feature_columns, treatment_col, outcome_col]
        if column not in data
    ]
    if missing:
        raise ValueError(f"Missing columns required for modelling: {missing}")

    x = data[spec.feature_columns].copy()
    treatment = pd.Series(
        _validated_binary_indicator(data[treatment_col], name="treatment"),
        index=data.index,
        name=treatment_col,
    )
    outcome = pd.Series(
        _validated_binary_indicator(data[outcome_col], name="outcome"),
        index=data.index,
        name=outcome_col,
    )
    return x, treatment, outcome


def honest_train_test_split(
    *arrays: pd.DataFrame | pd.Series | NDArray[Any],
    treatment: pd.Series | NDArray[Any],
    outcome: pd.Series | NDArray[Any] | None = None,
    test_size: float = 0.30,
    random_state: int = 42,
) -> list[pd.DataFrame | pd.Series | NDArray[Any]]:
    """Split arrays into train and test, preserving treatment-arm balance.

    A naive random split can over- or under-represent one arm in the test
    set, which biases AUQ and policy-value estimates. This helper stratifies
    the split on the treatment indicator (and on the joint treatment-outcome
    label when an outcome is supplied), so that:

    - both arms appear in both train and test,
    - the treated share is preserved on both sides of the split,
    - if an outcome is supplied and each treatment-outcome stratum is large
      enough, the conversion rate is preserved per arm; otherwise a warning
      is raised and stratification falls back to treatment alone.

    The function is a thin wrapper around :func:`sklearn.model_selection.train_test_split`
    whose name documents the intent. It returns the split arrays in the same
    order as the inputs (matching sklearn's convention).

    Parameters
    ----------
    *arrays:
        Arrays to split. Must all have the same length as `treatment`.
    treatment:
        Binary treatment indicator used for stratification.
    outcome:
        Optional binary outcome. When provided, stratification uses the
        joint treatment-outcome label so per-arm conversion rates are also
        preserved.
    test_size:
        Share of rows to put in the test split.
    random_state:
        Seed for the underlying sklearn split.
    """

    treatment_arr = _validated_binary_indicator(treatment, name="treatment", require_both=True)
    if not arrays:
        raise ValueError("At least one array to split is required")
    if any(len(array) != len(treatment_arr) for array in arrays):
        raise ValueError("All arrays and treatment must have the same length")
    if not np.isfinite(test_size) or not 0 < test_size < 1:
        raise ValueError("test_size must be a finite fraction in (0, 1)")

    n_test = int(np.ceil(len(treatment_arr) * test_size))
    n_train = len(treatment_arr) - n_test
    if np.bincount(treatment_arr, minlength=2).min() < 2:
        raise ValueError("Each treatment arm needs at least two observations for an honest split")
    if min(n_train, n_test) < 2:
        raise ValueError("Both train and test splits need at least two rows")

    if outcome is not None:
        outcome_arr = _validated_binary_indicator(outcome, name="outcome")
        if len(outcome_arr) != len(treatment_arr):
            raise ValueError("treatment and outcome must have the same length")
        # The four possible treatment × outcome cells are encoded distinctly.
        joint_labels = 2 * treatment_arr + outcome_arr
        stratum_counts = np.bincount(joint_labels, minlength=4)
        present_strata = stratum_counts[stratum_counts > 0]
        if present_strata.min() >= 2 and min(n_train, n_test) >= len(present_strata):
            stratify = joint_labels
        else:
            warn(
                "Rare treatment-outcome strata prevent joint stratification; "
                "using treatment-only stratification instead. Per-arm outcome "
                "balance is not guaranteed.",
                UserWarning,
                stacklevel=2,
            )
            stratify = treatment_arr
    else:
        stratify = treatment_arr

    # Split treatment alongside the inputs to verify the advertised guarantee.
    # sklearn returns alternating train/test slices for each input.
    splits = cast(
        list[pd.DataFrame | pd.Series | NDArray[Any]],
        train_test_split(
            *arrays,
            treatment_arr,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        ),
    )
    train_treatment = np.asarray(splits[-2])
    test_treatment = np.asarray(splits[-1])
    if any(set(np.unique(labels)) != {0, 1} for labels in (train_treatment, test_treatment)):
        raise ValueError(
            "Both treatment arms must appear in train and test; "
            "adjust test_size or collect more observations"
        )
    return splits[:-2]
