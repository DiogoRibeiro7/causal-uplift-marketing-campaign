from __future__ import annotations

import pandas as pd
import pytest

from uplift_marketing.data import generate_synthetic_uplift_data, prepare_uplift_frame, validate_required_columns


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
