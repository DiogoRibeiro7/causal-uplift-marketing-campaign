"""Feature selection and preprocessing for uplift models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, cast

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
    treatment = data[treatment_col].astype(int)
    outcome = data[outcome_col].astype(int)
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
    - if an outcome is supplied, the conversion rate is preserved per arm.

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

    treatment_arr = np.asarray(treatment).astype(int)
    if set(np.unique(treatment_arr)) - {0, 1}:
        raise ValueError("treatment must be binary 0/1")

    if outcome is not None:
        outcome_arr = np.asarray(outcome).astype(int)
        if len(outcome_arr) != len(treatment_arr):
            raise ValueError("treatment and outcome must have the same length")
        stratify = pd.Series(treatment_arr.astype(str)) + "_" + pd.Series(outcome_arr.astype(str))
        stratify = stratify.to_numpy()
    else:
        stratify = treatment_arr

    # sklearn returns a list alternating train/test slices for each input.
    # Cast the broad upstream stub to the input types preserved by the split.
    return cast(
        list[pd.DataFrame | pd.Series | NDArray[Any]],
        train_test_split(
            *arrays,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        ),
    )
