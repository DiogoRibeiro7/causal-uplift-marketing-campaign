from __future__ import annotations

import pandas as pd
import pytest

from uplift_marketing.data import (
    HILLSTROM_FEATURE_COLUMNS,
    UpliftDataset,
    generate_synthetic_uplift_data,
    load_default_uplift_data,
    load_hillstrom,
    prepare_uplift_frame,
    validate_required_columns,
)


def test_generate_synthetic_uplift_data_has_required_columns() -> None:
    data = generate_synthetic_uplift_data(n_rows=100, random_state=1)
    assert {"treatment", "conversion"}.issubset(data.columns)
    assert len(data) == 100
    assert set(data["treatment"].unique()).issubset({0, 1})
    assert set(data["conversion"].unique()).issubset({0, 1})


def test_generate_synthetic_uplift_data_rejects_non_positive_rows() -> None:
    with pytest.raises(ValueError, match="n_rows must be positive"):
        generate_synthetic_uplift_data(n_rows=0)


def test_validate_required_columns_raises_for_missing_column() -> None:
    data = pd.DataFrame({"treatment": [1, 0]})
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_required_columns(data)


def test_prepare_uplift_frame_normalizes_yes_no_labels() -> None:
    data = pd.DataFrame({"treatment": ["yes", "no"], "conversion": ["1", "0"], "x": [1, 2]})
    prepared = prepare_uplift_frame(data)
    assert prepared["treatment"].tolist() == [1, 0]
    assert prepared["conversion"].tolist() == [1, 0]


def _hillstrom_fixture(tmp_path):  # type: ignore[no-untyped-def]
    """Write a tiny CSV with the canonical Hillstrom schema."""

    raw = pd.DataFrame(
        {
            "recency": [1, 6, 10, 4, 8, 2],
            "history_segment": ["1) $0 - $100"] * 6,
            "history": [50.0, 200.0, 100.0, 500.0, 75.0, 300.0],
            "mens": [1, 0, 1, 1, 0, 1],
            "womens": [0, 1, 1, 0, 1, 1],
            "zip_code": ["Urban", "Urban", "Rural", "Rural", "Suburban", "Suburban"],
            "newbie": [1, 0, 1, 0, 1, 0],
            "channel": ["Web", "Phone", "Web", "Multichannel", "Phone", "Web"],
            "segment": [
                "Mens E-Mail",
                "Womens E-Mail",
                "No E-Mail",
                "Mens E-Mail",
                "Womens E-Mail",
                "No E-Mail",
            ],
            "visit": [1, 0, 0, 1, 1, 0],
            "conversion": [1, 0, 0, 0, 1, 0],
            "spend": [25.5, 0.0, 0.0, 0.0, 12.0, 0.0],
        }
    )
    path = tmp_path / "hillstrom_tiny.csv"
    raw.to_csv(path, index=False)
    return path


def test_load_hillstrom_any_combines_email_arms(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = _hillstrom_fixture(tmp_path)
    df = load_hillstrom(path, treated_arm="any")
    assert set(df.columns) >= {
        "customer_id",
        *HILLSTROM_FEATURE_COLUMNS,
        "treatment",
        "conversion",
    }
    assert len(df) == 6
    # Any e-mail arm = any of the two email segments. Original file has 4 emails.
    assert df["treatment"].sum() == 4
    assert set(df["conversion"].unique()).issubset({0, 1})


def test_load_hillstrom_filters_to_single_arm(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = _hillstrom_fixture(tmp_path)
    df = load_hillstrom(path, treated_arm="mens")
    # 2 Mens E-Mail rows + 2 No E-Mail rows in the fixture.
    assert len(df) == 4
    assert df["treatment"].sum() == 2


def test_load_hillstrom_picks_visit_outcome(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = _hillstrom_fixture(tmp_path)
    df = load_hillstrom(path, outcome="visit")
    # Visit count from the fixture is 3.
    assert df["conversion"].sum() == 3


def test_load_hillstrom_rejects_unknown_arm(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = _hillstrom_fixture(tmp_path)
    with pytest.raises(ValueError, match="treated_arm"):
        load_hillstrom(path, treated_arm="other")  # type: ignore[arg-type]


def test_load_default_uplift_data_falls_back_to_synthetic(tmp_path) -> None:  # type: ignore[no-untyped-def]
    ds = load_default_uplift_data(tmp_path, n_synthetic_rows=200, random_state=7)
    assert isinstance(ds, UpliftDataset)
    assert ds.source == "synthetic"
    assert len(ds.data) == 200
    assert ds.expected_treated_share == 0.5
    assert ds.break_even_uplift == ds.cost_per_contact / ds.value_per_conversion
