"""Data loading, validation, and synthetic uplift data generation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"treatment", "conversion"}


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV or compressed CSV file.

    Parameters
    ----------
    path:
        Path to a `.csv`, `.csv.gz`, or other pandas-readable CSV file.

    Returns
    -------
    pandas.DataFrame
        Loaded tabular data.

    Raises
    ------
    FileNotFoundError
        If the input path does not exist.
    ValueError
        If the loaded dataset is empty.
    """

    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input data file does not exist: {input_path}")

    data = pd.read_csv(input_path)
    if data.empty:
        raise ValueError(f"Input data file is empty: {input_path}")
    return data


def save_csv(data: pd.DataFrame, path: str | Path) -> None:
    """Save a DataFrame to CSV, creating parent folders if needed."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)


def validate_required_columns(data: pd.DataFrame, required: Iterable[str] = REQUIRED_COLUMNS) -> None:
    """Validate that a DataFrame contains required columns.

    Parameters
    ----------
    data:
        Dataset to validate.
    required:
        Required column names.

    Raises
    ------
    TypeError
        If `data` is not a DataFrame.
    ValueError
        If required columns are missing.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def normalize_binary_column(data: pd.DataFrame, column: str) -> pd.Series:
    """Normalize a binary column to integer values 0 and 1.

    This helper accepts booleans, integers, floats containing 0/1, and common
    string labels such as yes/no or true/false.
    """

    if column not in data.columns:
        raise ValueError(f"Column not found: {column}")

    values = data[column]
    if values.isna().any():
        raise ValueError(f"Binary column contains missing values: {column}")

    if pd.api.types.is_bool_dtype(values):
        return values.astype(int)

    if pd.api.types.is_numeric_dtype(values):
        unique_values = set(values.astype(float).unique())
        if not unique_values.issubset({0.0, 1.0}):
            raise ValueError(f"Column {column!r} must contain only 0/1 values")
        return values.astype(int)

    mapping = {
        "0": 0,
        "1": 1,
        "false": 0,
        "true": 1,
        "no": 0,
        "yes": 1,
        "control": 0,
        "treated": 1,
        "treatment": 1,
    }
    normalized = values.astype(str).str.strip().str.lower().map(mapping)
    if normalized.isna().any():
        invalid = sorted(values[normalized.isna()].astype(str).unique())[:10]
        raise ValueError(f"Invalid binary labels in column {column!r}: {invalid}")
    return normalized.astype(int)


def prepare_uplift_frame(
    data: pd.DataFrame,
    *,
    treatment_col: str = "treatment",
    outcome_col: str = "conversion",
) -> pd.DataFrame:
    """Return a clean frame with normalized treatment and outcome columns."""

    validate_required_columns(data, required={treatment_col, outcome_col})
    prepared = data.copy()
    prepared[treatment_col] = normalize_binary_column(prepared, treatment_col)
    prepared[outcome_col] = normalize_binary_column(prepared, outcome_col)
    return prepared


def generate_synthetic_uplift_data(
    n_rows: int = 5_000,
    *,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic randomized marketing campaign dataset.

    The generator creates four behavioural groups:

    - Persuadables: treatment increases conversion.
    - Sure things: high baseline conversion, low incremental lift.
    - Lost causes: low conversion under both treatment and control.
    - Do-not-disturb: treatment can reduce conversion.

    The dataset is not meant to represent a real campaign. It exists so tests,
    examples, and the pipeline can run without external data.
    """

    if n_rows <= 0:
        raise ValueError("n_rows must be positive")

    rng = np.random.default_rng(random_state)

    age_band = rng.choice(["18-24", "25-34", "35-44", "45-54", "55+"], size=n_rows)
    tenure_months = rng.gamma(shape=2.0, scale=8.0, size=n_rows).clip(0, 72)
    prior_purchases = rng.poisson(lam=2.5, size=n_rows)
    days_since_last_visit = rng.exponential(scale=15.0, size=n_rows).clip(0, 120)
    email_opens_30d = rng.poisson(lam=1.8, size=n_rows)
    discount_affinity = rng.beta(a=2.0, b=5.0, size=n_rows)
    website_sessions_30d = rng.poisson(lam=3.0, size=n_rows)
    device = rng.choice(["mobile", "desktop", "tablet"], size=n_rows, p=[0.6, 0.3, 0.1])

    # Randomized campaign assignment. This is the key assumption behind clean uplift analysis.
    treatment = rng.binomial(1, 0.5, size=n_rows)

    baseline_logit = (
        -2.2
        + 0.04 * prior_purchases
        + 0.015 * email_opens_30d
        + 0.010 * website_sessions_30d
        - 0.015 * days_since_last_visit
        + 1.0 * discount_affinity
        + 0.15 * (device == "desktop")
    )

    treatment_effect_logit = (
        0.9 * discount_affinity
        + 0.25 * (email_opens_30d >= 2)
        - 0.50 * (prior_purchases >= 6)
        - 0.45 * (days_since_last_visit > 60)
        - 0.25 * (device == "tablet")
    )

    conversion_probability = 1.0 / (1.0 + np.exp(-(baseline_logit + treatment * treatment_effect_logit)))
    conversion = rng.binomial(1, conversion_probability)

    # A secondary outcome often present in uplift data.
    visit_probability = np.clip(conversion_probability + rng.normal(0.08, 0.03, size=n_rows), 0, 1)
    visit = rng.binomial(1, visit_probability)

    return pd.DataFrame(
        {
            "customer_id": np.arange(1, n_rows + 1),
            "age_band": age_band,
            "tenure_months": tenure_months.round(2),
            "prior_purchases": prior_purchases,
            "days_since_last_visit": days_since_last_visit.round(2),
            "email_opens_30d": email_opens_30d,
            "discount_affinity": discount_affinity.round(4),
            "website_sessions_30d": website_sessions_30d,
            "device": device,
            "treatment": treatment,
            "visit": visit,
            "conversion": conversion,
        }
    )
