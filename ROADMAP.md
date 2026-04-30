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

- [ ] Add campaign budget optimizer.
- [ ] Add cost-per-contact and conversion-value assumptions.
- [ ] Add scenario analysis for contact budgets.
- [ ] Add dashboard-ready output tables.

## v0.6 — Reporting

- [ ] Generate a campaign policy report automatically.
- [ ] Export plots to `reports/figures`.
- [ ] Add a model card focused on causal assumptions and misuse risks.
- [ ] Add an executive summary notebook.
