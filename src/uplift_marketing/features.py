"""Feature selection and preprocessing for uplift models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
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

    missing = [column for column in [*spec.feature_columns, treatment_col, outcome_col] if column not in data]
    if missing:
        raise ValueError(f"Missing columns required for modelling: {missing}")

    x = data[spec.feature_columns].copy()
    treatment = data[treatment_col].astype(int)
    outcome = data[outcome_col].astype(int)
    return x, treatment, outcome
