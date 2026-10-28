"""End-to-end uplift modelling pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from uplift_marketing.config import DEFAULT_CONFIG, ProjectConfig
from uplift_marketing.data import load_csv, prepare_uplift_frame, save_csv
from uplift_marketing.economics import (
    CampaignEconomics,
    optimize_budget,
    scenario_table,
)
from uplift_marketing.evaluation import (
    area_under_qini,
    area_under_uplift,
    normalized_qini,
    policy_value_at_fraction,
    qini_curve,
    uplift_curve,
)
from uplift_marketing.features import infer_feature_spec, split_features_treatment_outcome
from uplift_marketing.model_card import build_model_card, write_model_card
from uplift_marketing.models import SLearner, TLearner
from uplift_marketing.plots import export_campaign_figures
from uplift_marketing.policy import assign_campaign_segments, campaign_action_table
from uplift_marketing.policy_report import build_policy_report, write_policy_report
from uplift_marketing.treatment import naive_average_treatment_effect, outcome_by_treatment
from uplift_marketing.uplift import attach_uplift_scores, make_uplift_deciles, uplift_by_group


def export_dashboard(
    scored: pd.DataFrame,
    *,
    economics: CampaignEconomics,
    output_dir: str | Path,
    treatment_col: str = "treatment",
    outcome_col: str = "conversion",
    uplift_col: str = "uplift_score",
    score_col: str = "uplift_score",
    fractions: list[float] | None = None,
    economics_scenarios: dict[str, CampaignEconomics] | None = None,
    extra_target_columns: list[str] | None = None,
    write_model_card_md: bool = True,
    model_card_kwargs: dict[str, object] | None = None,
    write_figures: bool = True,
    write_policy_report_md: bool = True,
    policy_report_kwargs: dict[str, object] | None = None,
) -> dict[str, Path]:
    """Write dashboard-ready output tables to ``output_dir``.

    Produces a small set of files designed for direct ingestion into a BI
    tool or downstream CRM:

    - ``recommended_targets.csv`` — customer-level table sorted by uplift
      score, with the segment label and the recommended action joined in.
      The selection count matches the profit-optimal budget under the
      supplied economics; exactly those rows carry
      ``in_recommendation = 1``, including when scores are tied or the
      no-contact policy wins.
    - ``profit_curve.csv`` — full predicted-profit sweep over the budget
      grid.
    - ``budget_scenarios.csv`` — scenario-table output across multiple
      economics assumptions, when ``economics_scenarios`` is provided.
    - ``model_summary.json`` — single-page summary of headline metrics
      (AUQ, normalized Qini, AUUC, naive ATE, profit-optimal selection).
    - ``model_card.md`` — stakeholder-facing model card focused on
      causal assumptions and misuse risks. Skipped when
      ``write_model_card_md=False``. Pass ``model_card_kwargs`` to
      override defaults like ``model_name`` or ``dataset_source``.
    - ``figures/`` — PNG figures (Qini curve, profit curve, decile
      uplift, AUUC, scenario sensitivity when scenarios are supplied).
      Skipped when ``write_figures=False``.
    - ``policy_report.md`` — campaign policy report tying the
      recommendation, economics, performance, sensitivity sweep, and
      embedded figures into a stakeholder-facing markdown document.
      Links to the model card for due diligence. Skipped when
      ``write_policy_report_md=False``. Pass ``policy_report_kwargs``
      to override defaults like ``model_name`` or ``dataset_source``.

    The function does not depend on a fitted model — it operates entirely
    on the scored DataFrame, which is what the rest of the pipeline already
    persists. This means notebooks and ad-hoc analyses can call it
    independently of ``run_pipeline``.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required = {treatment_col, outcome_col, uplift_col, score_col}
    missing = required - set(scored.columns)
    if missing:
        raise ValueError(f"scored is missing required columns: {sorted(missing)}")

    fractions_array = (
        None if fractions is None else pd.array(fractions, dtype="float64").to_numpy()
    )
    recommendation = optimize_budget(
        scored,
        economics,
        score_col=score_col,
        uplift_col=uplift_col,
        fractions=fractions_array,
    )

    target_columns = ["customer_id"] if "customer_id" in scored.columns else []
    target_columns += [
        c for c in [
            uplift_col,
            "uplift_decile",
            "campaign_segment",
            treatment_col,
            outcome_col,
        ]
        if c in scored.columns
    ]
    if extra_target_columns:
        target_columns += [c for c in extra_target_columns if c in scored.columns]
    seen: set[str] = set()
    deduped = [c for c in target_columns if not (c in seen or seen.add(c))]

    targets_full = scored.sort_values(score_col, ascending=False).copy()
    # Select by ranked position, not score threshold: tied scores at the
    # boundary must not increase the budget or overstate the CRM target list.
    targets_full["in_recommendation"] = 0
    targets_full.iloc[
        : recommendation.n_selected,
        targets_full.columns.get_loc("in_recommendation"),
    ] = 1
    if "campaign_segment" in scored.columns:
        action_table = campaign_action_table(scored)
        action_map = dict(
            zip(
                action_table["campaign_segment"],
                action_table["recommended_action"],
                strict=True,
            )
        )
        targets_full["recommended_action"] = targets_full["campaign_segment"].map(action_map)
        deduped.append("recommended_action")
    deduped = [c for c in deduped if c in targets_full.columns]
    deduped.append("in_recommendation")

    targets_path = output_dir / "recommended_targets.csv"
    save_csv(targets_full[deduped], targets_path)

    profit_curve_path = output_dir / "profit_curve.csv"
    save_csv(recommendation.profit_curve, profit_curve_path)

    scenarios_path: Path | None = None
    scenarios: pd.DataFrame | None = None
    if economics_scenarios:
        merged_scenarios = {"baseline": economics, **economics_scenarios}
        scenarios = scenario_table(
            scored,
            merged_scenarios,
            policies={"uplift_targeted": score_col},
            uplift_col=uplift_col,
            fractions=fractions_array,
        )
        scenarios_path = output_dir / "budget_scenarios.csv"
        save_csv(scenarios, scenarios_path)

    qini = qini_curve(
        outcome=scored[outcome_col].to_numpy(),
        treatment=scored[treatment_col].to_numpy(),
        uplift_score=scored[uplift_col].to_numpy(),
    )
    auuc = uplift_curve(
        outcome=scored[outcome_col].to_numpy(),
        treatment=scored[treatment_col].to_numpy(),
        uplift_score=scored[uplift_col].to_numpy(),
    )
    norm = normalized_qini(
        outcome=scored[outcome_col].to_numpy(),
        treatment=scored[treatment_col].to_numpy(),
        uplift_score=scored[uplift_col].to_numpy(),
    )
    summary = {
        "n_customers": int(len(scored)),
        "naive_ate": float(naive_average_treatment_effect(scored)),
        "auq": float(area_under_qini(qini)),
        "auuc": float(area_under_uplift(auuc)),
        "normalized_qini": norm,
        "economics": {
            "currency": economics.currency,
            "cost_per_contact": economics.cost_per_contact,
            "value_per_conversion": economics.value_per_conversion,
            "break_even_uplift": economics.break_even_uplift,
        },
        "recommendation": {
            "fraction": recommendation.fraction,
            "n_selected": recommendation.n_selected,
            "threshold_score": recommendation.threshold_score,
            "predicted_incremental_conversions": recommendation.predicted_incremental_conversions,
            "predicted_revenue": recommendation.predicted_revenue,
            "predicted_cost": recommendation.predicted_cost,
            "predicted_net_profit": recommendation.predicted_net_profit,
        },
    }
    summary_path = output_dir / "model_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    paths: dict[str, Path] = {
        "recommended_targets": targets_path,
        "profit_curve": profit_curve_path,
        "model_summary": summary_path,
    }
    if scenarios_path is not None:
        paths["budget_scenarios"] = scenarios_path

    if write_model_card_md:
        card = build_model_card(
            scored,
            economics=economics,
            recommendation=recommendation,
            treatment_col=treatment_col,
            outcome_col=outcome_col,
            uplift_col=uplift_col,
            **(model_card_kwargs or {}),
        )
        paths["model_card"] = write_model_card(card, output_dir / "model_card.md")

    figure_paths: dict[str, Path] = {}
    if write_figures:
        figure_paths = export_campaign_figures(
            scored,
            recommendation=recommendation,
            economics=economics,
            scenarios_table=scenarios,
            output_dir=output_dir / "figures",
            treatment_col=treatment_col,
            outcome_col=outcome_col,
            uplift_col=uplift_col,
        )
        for label, path in figure_paths.items():
            paths[f"figure_{label}"] = path

    if write_policy_report_md:
        report = build_policy_report(
            scored,
            economics=economics,
            recommendation=recommendation,
            scenarios_table=scenarios,
            figure_paths=figure_paths,
            model_card_path=paths.get("model_card"),
            treatment_col=treatment_col,
            outcome_col=outcome_col,
            uplift_col=uplift_col,
            **(policy_report_kwargs or {}),
        )
        paths["policy_report"] = write_policy_report(
            report, output_dir / "policy_report.md"
        )
    return paths


def run_pipeline(
    input_path: str | Path,
    *,
    config: ProjectConfig = DEFAULT_CONFIG,
    model_name: str = "t_learner",
    economics: CampaignEconomics | None = None,
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
    economics:
        Cost / value assumptions used to write dashboard outputs and the
        profit-optimal recommendation. When omitted, conservative defaults
        (1 EUR per contact, 15 EUR per conversion) are used; replace with
        the campaign owner's numbers before treating the recommendation as
        production-ready.

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

    (
        x_train,
        x_test,
        treatment_train,
        treatment_test,
        outcome_train,
        outcome_test,
        raw_train,
        raw_test,
    ) = train_test_split(
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

    scored = attach_uplift_scores(
        raw_test.reset_index(drop=True),
        p_control=p_control,
        p_treated=p_treated,
    )
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
    save_csv(
        uplift_by_group(scored, group_col="uplift_decile"),
        config.reports_dir / "uplift_by_decile.csv",
    )
    joblib.dump(model, model_path)

    economics = economics or CampaignEconomics(
        cost_per_contact=1.0,
        value_per_conversion=15.0,
    )
    dashboard_paths = export_dashboard(
        scored,
        economics=economics,
        output_dir=config.reports_dir / "dashboard",
        treatment_col=config.treatment_col,
        outcome_col=config.outcome_col,
    )

    return {
        "scored_path": scored_path,
        "action_table_path": action_table_path,
        "qini_path": qini_path,
        "model_path": model_path,
        "dashboard_dir": (config.reports_dir / "dashboard"),
        "recommended_targets_path": dashboard_paths["recommended_targets"],
        "model_summary_path": dashboard_paths["model_summary"],
        "naive_ate": naive_average_treatment_effect(data),
        "auq": auq,
        "top_20_observed_uplift": float(policy_top_20["observed_uplift"]),
    }
