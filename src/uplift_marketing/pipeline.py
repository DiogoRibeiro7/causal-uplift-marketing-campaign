"""End-to-end uplift modelling pipeline."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from uplift_marketing.config import DEFAULT_CONFIG, ProjectConfig
from uplift_marketing.data import load_csv, prepare_uplift_frame, save_csv
from uplift_marketing.evaluation import area_under_qini, policy_value_at_fraction, qini_curve
from uplift_marketing.features import infer_feature_spec, split_features_treatment_outcome
from uplift_marketing.models import SLearner, TLearner
from uplift_marketing.policy import assign_campaign_segments, campaign_action_table
from uplift_marketing.treatment import naive_average_treatment_effect, outcome_by_treatment
from uplift_marketing.uplift import attach_uplift_scores, make_uplift_deciles, uplift_by_group


def run_pipeline(
    input_path: str | Path,
    *,
    config: ProjectConfig = DEFAULT_CONFIG,
    model_name: str = "t_learner",
) -> dict[str, Path | float]:
    """Run the full uplift modelling workflow.

    Parameters
    ----------
    input_path:
        Input CSV file containing treatment, outcome, and feature columns.
    config:
        Project configuration.
    model_name:
        Either `s_learner` or `t_learner`.

    Returns
    -------
    dict[str, Path | float]
        Paths to generated outputs and key summary metrics.
    """

    data = load_csv(input_path)
    data = prepare_uplift_frame(
        data,
        treatment_col=config.treatment_col,
        outcome_col=config.outcome_col,
    )

    # Exclude IDs and secondary outcomes from feature modelling by default.
    extra_excluded = [column for column in ["customer_id", "visit"] if column in data.columns]
    spec = infer_feature_spec(
        data,
        treatment_col=config.treatment_col,
        outcome_col=config.outcome_col,
        extra_excluded=extra_excluded,
    )
    x, treatment, outcome = split_features_treatment_outcome(
        data,
        spec,
        treatment_col=config.treatment_col,
        outcome_col=config.outcome_col,
    )

    x_train, x_test, treatment_train, treatment_test, outcome_train, outcome_test, raw_train, raw_test = train_test_split(
        x,
        treatment,
        outcome,
        data,
        test_size=0.30,
        random_state=config.random_state,
        stratify=treatment.astype(str) + "_" + outcome.astype(str),
    )

    if model_name == "s_learner":
        model = SLearner(spec=spec, random_state=config.random_state)
    elif model_name == "t_learner":
        model = TLearner(spec=spec, random_state=config.random_state)
    else:
        raise ValueError("model_name must be either 's_learner' or 't_learner'")

    model.fit(x_train, treatment_train, outcome_train)
    p_control, p_treated = model.predict_potential_outcomes(x_test)

    scored = attach_uplift_scores(raw_test.reset_index(drop=True), p_control=p_control, p_treated=p_treated)
    scored["uplift_decile"] = make_uplift_deciles(scored)
    scored["campaign_segment"] = assign_campaign_segments(scored)

    qini = qini_curve(
        outcome=scored[config.outcome_col].to_numpy(),
        treatment=scored[config.treatment_col].to_numpy(),
        uplift_score=scored["uplift_score"].to_numpy(),
    )
    auq = area_under_qini(qini)
    policy_top_20 = policy_value_at_fraction(scored, fraction=0.20)

    config.reports_dir.mkdir(parents=True, exist_ok=True)
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    config.processed_dir.mkdir(parents=True, exist_ok=True)

    scored_path = config.reports_dir / "scored_customers.csv"
    action_table_path = config.reports_dir / "campaign_action_table.csv"
    qini_path = config.reports_dir / "qini_curve.csv"
    treatment_summary_path = config.reports_dir / "treatment_summary.csv"
    model_path = config.reports_dir / "uplift_model.joblib"

    save_csv(scored, scored_path)
    save_csv(campaign_action_table(scored), action_table_path)
    save_csv(qini, qini_path)
    save_csv(outcome_by_treatment(data), treatment_summary_path)
    save_csv(uplift_by_group(scored, group_col="uplift_decile"), config.reports_dir / "uplift_by_decile.csv")
    joblib.dump(model, model_path)

    return {
        "scored_path": scored_path,
        "action_table_path": action_table_path,
        "qini_path": qini_path,
        "model_path": model_path,
        "naive_ate": naive_average_treatment_effect(data),
        "auq": auq,
        "top_20_observed_uplift": float(policy_top_20["observed_uplift"]),
    }
