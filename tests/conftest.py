from __future__ import annotations

import pandas as pd
import pytest

from uplift_marketing.data import generate_synthetic_uplift_data


@pytest.fixture()
def synthetic_data() -> pd.DataFrame:
    return generate_synthetic_uplift_data(n_rows=500, random_state=123)
