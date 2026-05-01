"""Causal uplift modelling utilities for marketing campaign incrementality."""

from uplift_marketing.config import ProjectConfig
from uplift_marketing.economics import (
    BudgetRecommendation,
    CampaignEconomics,
    optimize_budget,
    scenario_table,
)
from uplift_marketing.model_card import ModelCard, build_model_card, write_model_card
from uplift_marketing.models import DRLearner, SLearner, TLearner, XLearner
from uplift_marketing.plots import export_campaign_figures
from uplift_marketing.policy_report import (
    PolicyReport,
    build_policy_report,
    write_policy_report,
)

__all__ = [
    "BudgetRecommendation",
    "CampaignEconomics",
    "DRLearner",
    "ModelCard",
    "PolicyReport",
    "ProjectConfig",
    "SLearner",
    "TLearner",
    "XLearner",
    "build_model_card",
    "build_policy_report",
    "export_campaign_figures",
    "optimize_budget",
    "scenario_table",
    "write_model_card",
    "write_policy_report",
]

__version__ = "0.1.0"
