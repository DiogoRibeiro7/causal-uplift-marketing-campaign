"""Project configuration and filesystem paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectConfig:
    """Central configuration for the uplift modelling project.

    Parameters
    ----------
    root_dir:
        Repository root directory.
    random_state:
        Seed used for reproducible modelling and data generation.
    treatment_col:
        Name of the binary treatment indicator column.
    outcome_col:
        Name of the binary outcome column used for uplift estimation.
    """

    root_dir: Path = Path(__file__).resolve().parents[2]
    random_state: int = 42
    treatment_col: str = "treatment"
    outcome_col: str = "conversion"

    @property
    def data_dir(self) -> Path:
        """Return the data directory."""

        return self.root_dir / "data"

    @property
    def raw_dir(self) -> Path:
        """Return the raw data directory."""

        return self.data_dir / "raw"

    @property
    def interim_dir(self) -> Path:
        """Return the interim data directory."""

        return self.data_dir / "interim"

    @property
    def processed_dir(self) -> Path:
        """Return the processed data directory."""

        return self.data_dir / "processed"

    @property
    def reports_dir(self) -> Path:
        """Return the reports directory."""

        return self.root_dir / "reports"

    @property
    def figures_dir(self) -> Path:
        """Return the report figures directory."""

        return self.reports_dir / "figures"


DEFAULT_CONFIG = ProjectConfig()
