# Roadmap

## v0.1 — Repository foundation

- [x] Create package structure.
- [x] Add synthetic uplift data generator.
- [x] Add S-learner and T-learner model utilities.
- [x] Add Qini and uplift evaluation helpers.
- [x] Add policy segmentation and action mapping.
- [x] Add notebook skeletons with decision-support framing.

## v0.2 — Real data integration

- [x] Validate the downloader against the selected real dataset (Hillstrom MineThatData via Kaggle, plus `.env` loading for credentials).
- [x] Add a robust data schema adapter for Criteo-like uplift data (`load_hillstrom`, `load_default_uplift_data`, `UpliftDataset`).
- [x] Add feature metadata and column documentation (see [`docs/DATA_SOURCE.md`](docs/DATA_SOURCE.md)).
- [x] Add data checks for treatment/control randomization assumptions (`validate_randomization` with chi-square SRM, SMD per feature, null share).

## v0.3 — Uplift modelling depth

- [x] Add cross-fitting for treatment-effect estimation (`cross_fit_uplift_oof`, plus K-fold cross-fitting baked into `DRLearner`).
- [x] Add calibrated probability models (`make_calibrated_classifier`, with a NB06 panel showing the predicted-vs-realised gap shrinks ~50× under isotonic calibration on Hillstrom).
- [x] Add doubly robust uplift estimation (`DRLearner` — AIPW pseudo-outcomes via cross-fitting, second-stage Ridge regressor).
- [x] Add honest validation splits by treatment arm (`honest_train_test_split`, stratified on treatment × outcome).

## v0.4 — Evaluation maturity

- [x] Add AUUC and normalized Qini metrics (`uplift_curve`, `area_under_uplift`, `normalized_qini`; reported in NB05 alongside raw AUQ).
- [x] Add bootstrap confidence intervals for policy value (`bootstrap_policy_value`; NB05 top-k chart now carries CIs).
- [x] Add decile-level uplift stability analysis (`bootstrap_decile_uplift`; NB05 decile chart now carries error bars).
- [x] Add subgroup-level uplift analysis (`subgroup_uplift_with_ci`; NB07 has a fairness/segmentation panel).

## v0.5 — Campaign simulation

- [x] Add campaign budget optimizer (`optimize_budget` in `uplift_marketing.economics`; sweeps the predicted-profit curve and returns a `BudgetRecommendation` with the optimal fraction, cut-off score, and selected customer list).
- [x] Add cost-per-contact and conversion-value assumptions (`CampaignEconomics` value object with `break_even_uplift` and `net_profit`; surfaced through `UpliftDataset.economics`).
- [x] Add scenario analysis for contact budgets (`scenario_table` sweeps policy × economics combinations; NB06 §9 plots sensitivity to halved value / doubled contact cost).
- [x] Add dashboard-ready output tables (`export_dashboard` writes `recommended_targets.csv`, `profit_curve.csv`, `budget_scenarios.csv`, `model_summary.json` under `reports/dashboard/`; wired into `run_pipeline` and NB06 §10).

## v0.6 — Reporting

- [x] Generate a campaign policy report automatically (`uplift_marketing.policy_report.build_policy_report` / `write_policy_report`; produces a stakeholder-facing markdown report tying executive summary, performance metrics, profit-optimal recommendation, scenario sensitivity, and embedded figures into one document. Links to the model card for due diligence. `export_dashboard` writes `reports/dashboard/policy_report.md` automatically).
- [x] Export plots to `reports/figures` (`uplift_marketing.plots` module: `fig_qini_curve`, `fig_profit_curve`, `fig_uplift_by_decile`, `fig_uplift_curve`, `fig_scenarios`; `export_campaign_figures` wraps the canonical set. `export_dashboard` writes the PNGs under `reports/dashboard/figures/` automatically).
- [x] Add a model card focused on causal assumptions and misuse risks (`uplift_marketing.model_card.build_model_card` / `write_model_card`; renders structured markdown covering randomization diagnostics, SUTVA / consistency / positivity, performance metrics with reading guidance, and named misuse risks. `export_dashboard` writes `reports/dashboard/model_card.md` automatically; NB06 §11 demonstrates the workflow).
- [x] Add an executive summary notebook ([NB09](notebooks/09_executive_summary.ipynb): 2-minute read with the question, what we did, headline result, sensitivity, what we cannot say, and pointers to the deeper notebooks. Reads from `reports/dashboard/` so it stays in sync with the latest pipeline run).
