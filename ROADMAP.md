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

- [ ] Add cross-fitting for treatment-effect estimation.
- [ ] Add calibrated probability models.
- [ ] Add doubly robust uplift estimation.
- [ ] Add honest validation splits by treatment arm.

## v0.4 — Evaluation maturity

- [ ] Add AUUC and normalized Qini metrics.
- [ ] Add bootstrap confidence intervals for policy value.
- [ ] Add decile-level uplift stability analysis.
- [ ] Add subgroup-level uplift analysis.

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
