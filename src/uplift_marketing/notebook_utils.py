"""Helpers shared by the analytical notebooks.

The notebooks import these utilities to keep their setup short, their plots
visually consistent, and their figure outputs discoverable under
`reports/figures/`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

import matplotlib as mpl
from matplotlib.figure import Figure

from uplift_marketing.config import DEFAULT_CONFIG, ProjectConfig

PLOT_STYLE = {
    "figure.figsize": (8.0, 4.5),
    "figure.dpi": 110,
    "savefig.dpi": 140,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "axes.titlesize": 12,
    "axes.titleweight": "semibold",
    "axes.labelsize": 10,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "font.family": "DejaVu Sans",
}


def setup_notebook(config: ProjectConfig = DEFAULT_CONFIG) -> ProjectConfig:
    """Configure imports, plotting style, and project paths for a notebook.

    Returns the active `ProjectConfig` so the notebook can reach `data_dir`,
    `reports_dir`, and `figures_dir` without re-deriving them.

    Loads variables from a `.env` file at the repository root if `python-dotenv`
    is installed, so notebooks see `KAGGLE_USERNAME` etc. without any extra
    boilerplate.
    """

    src_path = str(config.root_dir / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    try:
        from dotenv import load_dotenv

        load_dotenv(config.root_dir / ".env")
    except ImportError:
        pass

    config.figures_dir.mkdir(parents=True, exist_ok=True)
    config.reports_dir.mkdir(parents=True, exist_ok=True)

    # Matplotlib's typed rcParams key literals cannot describe a dynamic
    # dictionary, but Matplotlib validates the keys and values at runtime.
    mpl.rcParams.update(cast(Any, PLOT_STYLE))
    return config


def save_figure(
    fig: Figure,
    name: str,
    *,
    config: ProjectConfig = DEFAULT_CONFIG,
    formats: tuple[str, ...] = ("png",),
) -> list[Path]:
    """Save a matplotlib figure to `reports/figures/` and return the paths."""

    config.figures_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for fmt in formats:
        path = config.figures_dir / f"{name}.{fmt}"
        fig.savefig(path)
        paths.append(path)
    return paths
