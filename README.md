# causal-uplift-marketing-campaign

End-to-end uplift modelling and campaign-policy evaluation for marketing
incrementality. The repository pairs a small Python package
(`uplift_marketing`) with a sequence of decision-support notebooks
(NB01–NB09) and produces a stakeholder-facing hand-off — model card,
policy report, dashboard CSVs, and embedded figures — from a single
pipeline call.

The framing is deliberately causal, not predictive. The questions are:

> *Which customers will respond to the campaign **because** of the
> campaign?*
> *What is the profit-optimal contact budget under the supplied
> economics?*
> *What can we say with confidence, and what assumptions is every
> number contingent on?*

Built on the public Hillstrom MineThatData email dataset; falls back to
a synthetic generator when the real data is unavailable.

## Highlights

- **Uplift estimators** — `SLearner`, `TLearner`, `XLearner`, and a
  cross-fitted doubly-robust `DRLearner` (AIPW pseudo-outcomes, Ridge
  second stage). Optional isotonic calibration via
  `make_calibrated_classifier` closes the predicted-vs-realised profit
  gap on rare-event data.
- **Honest evaluation** — Qini, treatment-share-adjusted AUUC, normalized
  Qini, bootstrap CIs on policy value, decile stability, subgroup-level
  uplift with CIs. Qini/AUUC areas include the no-contact origin, and
  require curves to cover the complete population. Tied scores receive neutral
  interpolation instead of arbitrary ordering.
- **Profit-optimal targeting** — `CampaignEconomics`, `optimize_budget`
  (searches every possible top-k size by default, returning the cut-off
  score, customer list, and predicted economics;
  recommends no contacts when no evaluated campaign offers positive profit),
  `scenario_table` (sweeps cost / value scenarios). Pass `fractions` to restrict
  eligible budgets to a supplied grid.
- **Stakeholder hand-off** — `export_dashboard` writes a self-contained
  `reports/dashboard/` directory: BI-ready CSVs, a model card focused
  on causal assumptions and misuse risks, a campaign policy report
  with embedded figures, and a `model_summary.json` model card.
- **Decision-support notebooks** — NB01–NB09 build the story end to end,
  with NB09 acting as a 2-minute executive summary that reads from the
  dashboard outputs.

## Citation

Use [CITATION.cff](CITATION.cff) for the repository's software citation. The
[Zenodo concept DOI](https://doi.org/10.5281/zenodo.23803160) identifies the
evolving project; to cite the precise v0.6.0 release, use
[10.5281/zenodo.23803161](https://doi.org/10.5281/zenodo.23803161).
Zenodo release descriptions and keywords are maintained in
[.zenodo.json](.zenodo.json) and checked alongside the citation file in CI.

## Quickstart

This project uses [Poetry](https://python-poetry.org/). After cloning:

```bash
poetry install
poetry run pytest                  # unit and regression tests
```

To explore the notebooks, install the optional JupyterLab group:

```bash
poetry install --with notebooks
poetry run jupyter lab notebooks/  # NB01..NB09
```

To use the real Hillstrom dataset, install the optional Kaggle integration
and supply credentials in a `.env` file at the repo root. The loader falls
back to synthetic data without credentials, so the notebooks can still run:

```bash
poetry install -E kaggle
```

Then configure your credentials:

```dotenv
KAGGLE_USERNAME=your-username
KAGGLE_KEY=your-key
```

Optional XGBoost and LightGBM comparison models are available through
`poetry install -E boosters`.

### Build distributions

Build the source archive and wheel locally:

```bash
poetry build
```

This creates both artifacts under `dist/`. CI verifies that they build.
Stable GitHub Releases attach validated wheel and source distributions;
the project does not publish to PyPI. See [the release guide](docs/RELEASING.md).

### Programmatic use

The package is structured so you can drop the notebooks entirely and
run the pipeline from Python:

```python
from uplift_marketing import (
    CampaignEconomics,
    TLearner,
    optimize_budget,
)
from uplift_marketing.data import load_default_uplift_data
from uplift_marketing.features import (
    infer_feature_spec,
    split_features_treatment_outcome,
)
from uplift_marketing.uplift import attach_uplift_scores
from uplift_marketing.pipeline import export_dashboard
from sklearn.model_selection import train_test_split

dataset = load_default_uplift_data("data/raw")
data = dataset.data
spec = infer_feature_spec(data, extra_excluded=dataset.extra_excluded)
x, t, y = split_features_treatment_outcome(data, spec)
x_tr, x_te, t_tr, t_te, y_tr, y_te, _, raw_te = train_test_split(
    x, t, y, data, test_size=0.30, random_state=42,
    stratify=t.astype(str) + "_" + y.astype(str),
)

model = TLearner(spec=spec, random_state=42).fit(x_tr, t_tr, y_tr)
p_c, p_t = model.predict_potential_outcomes(x_te)
scored = attach_uplift_scores(raw_te.reset_index(drop=True),
                              p_control=p_c, p_treated=p_t)

economics = dataset.economics
recommendation = optimize_budget(scored, economics)
print(f"Contact top {recommendation.fraction:.2%} "
      f"({recommendation.n_selected:,} customers); "
      f"predicted net profit "
      f"{economics.currency} {recommendation.predicted_net_profit:,.0f}")

paths = export_dashboard(
    scored,
    economics=economics,
    output_dir="reports/dashboard",
    economics_scenarios={
        "halved_value": CampaignEconomics(
            cost_per_contact=economics.cost_per_contact,
            value_per_conversion=economics.value_per_conversion / 2,
        ),
    },
    model_card_kwargs={
        "model_name": "T-learner",
        "dataset_source": dataset.source,
        "expected_treated_share": dataset.expected_treated_share,
    },
    policy_report_kwargs={
        "model_name": "T-learner",
        "dataset_source": dataset.source,
    },
)
```

## The notebooks

| # | Notebook | Question |
| --- | --- | --- |
| 01 | [Problem framing](notebooks/01_problem_framing_incrementality_not_conversion.ipynb) | Why uplift, not response rate? |
| 02 | [Data exploration](notebooks/02_data_exploration_treatment_control_outcomes.ipynb) | Is the experiment clean? |
| 03 | [Baseline response model](notebooks/03_baseline_response_model_and_why_it_fails.ipynb) | What's wrong with response models? |
| 04 | [S/T/X-learner](notebooks/04_uplift_models_s_learner_t_learner_x_learner.ipynb) | How do meta-learners compare? |
| 05 | [Qini, AUUC, policy CIs](notebooks/05_qini_gain_and_policy_evaluation.ipynb) | How do we evaluate uplift honestly? |
| 06 | [Budget allocation](notebooks/06_budget_allocation_and_campaign_simulation.ipynb) | What's the profit-optimal budget? |
| 07 | [Bias and assumptions](notebooks/07_bias_limitations_and_causal_assumptions.ipynb) | Where can this break? |
| 08 | [Black-box comparison](notebooks/08_comparison_with_black_box_uplift_models.ipynb) | How do meta-learners compare to black-box uplift methods? |
| 09 | [Executive summary](notebooks/09_executive_summary.ipynb) | The 2-minute read. |

## The package

| Module | Responsibility |
| --- | --- |
| [`config`](src/uplift_marketing/config.py) | Project paths and `ProjectConfig` |
| [`data`](src/uplift_marketing/data.py) | Loading, schema adapters, `UpliftDataset`, synthetic generator |
| [`features`](src/uplift_marketing/features.py) | Feature spec inference and preprocessing |
| [`models`](src/uplift_marketing/models.py) | `SLearner`, `TLearner`, `XLearner`, `DRLearner`, calibration |
| [`uplift`](src/uplift_marketing/uplift.py) | Score attachment, deciles, subgroup uplift with CIs |
| [`treatment`](src/uplift_marketing/treatment.py) | Naive ATE and outcome summaries (reject invalid assignments and non-finite outcomes), raw assignment counts, `validate_randomization` (SRM + SMD + null share) |
| [`evaluation`](src/uplift_marketing/evaluation.py) | Qini / AUUC / normalized Qini, bootstrap policy & decile CIs |
| [`policy`](src/uplift_marketing/policy.py) | Campaign segmentation and action mapping |
| [`economics`](src/uplift_marketing/economics.py) | `CampaignEconomics`, `optimize_budget`, `scenario_table` |
| [`plots`](src/uplift_marketing/plots.py) | Programmatic figure generation |
| [`model_card`](src/uplift_marketing/model_card.py) | Causal-assumption and misuse-risk model card |
| [`policy_report`](src/uplift_marketing/policy_report.py) | Stakeholder-facing campaign policy report |
| [`pipeline`](src/uplift_marketing/pipeline.py) | `run_pipeline`, `export_dashboard` orchestration |

## Hand-off artifacts

A single call to `export_dashboard` writes everything a stakeholder
needs under `reports/dashboard/`:

```text
reports/dashboard/
├── recommended_targets.csv     # every customer, score, segment, in_recommendation flag
├── profit_curve.csv            # compact budget grid plus the exact optimum
├── budget_scenarios.csv        # scenario × policy sensitivity sweep
├── model_summary.json          # AUQ / AUUC / normalized Qini / recommendation
├── model_card.md               # causal assumptions + misuse risks (due diligence)
├── policy_report.md            # decision artifact with embedded figures
└── figures/
    ├── campaign_qini.png
    ├── campaign_profit.png
    ├── campaign_decile.png
    ├── campaign_uplift_curve.png
    └── campaign_scenarios.png
```

The CRM target export preserves the exact ranked customers selected by the
budget optimizer, including when a budget cuts through equal ranking scores.
It retains both the ranking column (`score_col`) and the estimated incremental
effect (`uplift_col`), which can be different. Custom treatment and outcome
column names are supported in both `export_dashboard` and the `ProjectConfig`
used by `run_pipeline`.

The model card and policy report are split deliberately: the report is
the *decision* artifact ("what should we do, why, and how much will it
move the needle?"), the model card is the *due-diligence* artifact
("what assumptions is every number contingent on, and how can it be
misused?"). A deployment reviewer signs off the model card before the
recommendation goes live.

Normalized Qini is a ratio against an optimistic in-sample signed-outcome
reference, not a probability. It can be negative or exceed 1, particularly
for small or imbalanced experiments; zero is the no-skill baseline. Raw AUUC
also scales with the number of customers, so do not compare its absolute
value across datasets of different sizes.

## Causal assumptions and limitations

Every uplift number this project produces is contingent on:

- **Random treatment assignment** — verified for the Hillstrom dataset
  via SRM and per-feature SMD checks in `validate_randomization`.
- **SUTVA / no interference** — *unverifiable*. Plausible for
  arm's-length email, dubious for B2B, shared-inbox, or social-graph
  scenarios.
- **Consistency** — *unverifiable*. A different creative or channel is
  out-of-distribution for any model trained here.
- **Positivity** — holds under random assignment by construction; if
  real-world contact eligibility is conditional, restrict scoring to
  the eligible subpopulation.

Named misuse risks are enumerated in the generated model card —
proxy targeting, self-fulfilling prophecies, OOD reuse, treating point
estimates as ground truth, and score leakage into other systems.

Policy-value bootstrap intervals need treated and control observations inside
the selected top-k group. The estimator excludes resamples without both arms
and returns `n_valid_boot` / `n_invalid_boot` so that sparse-selection
uncertainty is visible. If fewer than two resamples support an estimate, it
raises an error instead of reporting an undefined interval. Profit intervals
require both `cost_per_contact` and `value_per_conversion`. A large invalid
resample count signals weak overlap; do not interpret the remaining percentile
interval as a well-supported population confidence interval.

## Layout

```text
.
├── data/                  # raw, interim, processed (gitignored except .gitkeep)
├── docs/                  # DATA_SOURCE.md and supporting docs
├── notebooks/             # NB01–NB09
├── reports/               # generated artifacts (gitignored except a tracked report stub)
├── scripts/               # CLI entry points
├── src/uplift_marketing/  # the package
├── tests/                 # pytest suite (78 tests)
├── pyproject.toml
├── ROADMAP.md             # versioned feature ledger
└── README.md
```

## Tests

```bash
poetry run pytest      # full test suite
poetry run ruff check  # project lint config
```

The suite covers data loading, feature inference, every meta-learner,
calibration, evaluation metrics, randomization checks, budget
optimisation, scenario sweeps, dashboard exports, plot rendering, model
card generation, and policy report rendering.

## License

MIT — see [LICENSE](LICENSE).
