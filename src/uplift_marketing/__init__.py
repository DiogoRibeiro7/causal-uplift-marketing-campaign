"""Causal uplift modelling utilities for marketing campaign incrementality."""

from uplift_marketing.config import ProjectConfig
from uplift_marketing.models import DRLearner, SLearner, TLearner, XLearner

__all__ = ["DRLearner", "ProjectConfig", "SLearner", "TLearner", "XLearner"]

__version__ = "0.1.0"
