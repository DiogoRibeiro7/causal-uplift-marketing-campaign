# Data Source

The project supports two data paths: a small synthetic generator (always
available) and the public Hillstrom MineThatData email-marketing uplift
dataset (downloaded from Kaggle on demand). Notebooks prefer Hillstrom when
the CSV is present and fall back to synthetic otherwise.

## Schema expected by the pipeline

Every dataset that flows through the pipeline conforms to a single schema:

```text
treatment   Binary campaign assignment: 1 treated, 0 control
conversion  Binary outcome: 1 converted, 0 did not convert
```

Optional columns:

```text
visit       Binary upstream-funnel outcome
exposure    Whether the ad was actually seen (Criteo-style)
f0..fN      Anonymous engineered features
```

Anything else is treated as a candidate covariate by `infer_feature_spec`.

## Hillstrom MineThatData (default real dataset)

Public uplift benchmark from Kevin Hillstrom (2008). 64,000 rows, three-arm
treatment, three outcomes, eight customer features. Mirrored on Kaggle as
[`bofulee/kevin-hillstrom-minethatdata-e-mailanalytics`][kaggle-hillstrom].

[kaggle-hillstrom]: https://www.kaggle.com/datasets/bofulee/kevin-hillstrom-minethatdata-e-mailanalytics

### Raw schema

| Column            | Type     | Description |
|---|---|---|
| `recency`         | int      | Months since last purchase. |
| `history_segment` | string   | Bucketed historical spend (e.g. ``2) $100 - $200``). |
| `history`         | float    | Historical spend in USD over the prior year. |
| `mens`            | int 0/1  | Purchased mens merchandise in the past 12 months. |
| `womens`          | int 0/1  | Purchased womens merchandise in the past 12 months. |
| `zip_code`        | string   | ``Urban`` / ``Suburban`` / ``Rural``. |
| `newbie`          | int 0/1  | New customer in the past 12 months. |
| `channel`         | string   | ``Phone`` / ``Web`` / ``Multichannel``. |
| `segment`         | string   | Treatment arm: ``Mens E-Mail``, ``Womens E-Mail``, or ``No E-Mail``. |
| `visit`           | int 0/1  | Visited the site within two weeks. |
| `conversion`      | int 0/1  | Converted within two weeks. |
| `spend`           | float    | USD spent within two weeks. |

### Adapter

[`uplift_marketing.data.load_hillstrom`](../src/uplift_marketing/data.py)
binarises the three-arm `segment` and selects one binary outcome so
the rest of the pipeline can consume the data through its standard
`treatment` / `conversion` schema.

```python
from uplift_marketing.data import load_hillstrom

df = load_hillstrom(
    "data/raw/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv",
    treated_arm="any",      # "any" | "mens" | "womens"
    outcome="conversion",   # "conversion" | "visit"
)
```

`treated_arm`:

- `"any"` — pools both Mens and Womens email arms vs the No E-Mail control
  (64,000 rows, the original challenge convention; mixes two creatives,
  which slightly weakens the treatment-consistency assumption).
- `"mens"` / `"womens"` — keep only the chosen arm and the No E-Mail control
  (≈42,600 rows, cleaner experimental contrast at the cost of fewer rows).

The adapter requires known, non-missing treatment labels and both treatment
arms after filtering. It also checks that the selected outcome is fully
observed and binary before converting it to integers. Invalid labels raise
an error instead of silently turning unknown treatments into controls or
fractional outcomes into binary events.

### Validation

[`uplift_marketing.treatment.validate_randomization`](../src/uplift_marketing/treatment.py)
runs the three checks every uplift workflow should pass before any modelling:

1. **Sample ratio mismatch.** Chi-square against the expected treated share.
2. **Covariate balance.** Standardized mean difference (SMD) per feature,
   flagging anything with ``|SMD| > 0.10``. If both arms have zero
   within-arm variance, identical means get SMD 0; different means get
   signed infinite SMD and are flagged. Missing arm moments remain undefined.
   Infinite or complex numeric covariate values are rejected before SMD
   calculation; ordinary nulls remain visible in the missingness check.
3. **Missingness.** Per-column null share for downstream feature-spec
   inference.

On the default `treated_arm="any"` configuration (expected treated share
= 2/3), Hillstrom passes with chi² ≈ 0.05 (p ≈ 0.82) and no feature
exceeding the SMD threshold.

## Synthetic fallback

[`uplift_marketing.data.generate_synthetic_uplift_data`](../src/uplift_marketing/data.py)
produces a small randomized 50/50 dataset with the same conventional column
names. It runs offline and is what the CI / unit tests use.

## How to fetch the real data

`KAGGLE_USERNAME` and `KAGGLE_KEY` are read from the process environment or
from a `.env` file at the repository root (see `.env.example`).

```bash
# Default mode: Hillstrom from Kaggle.
poetry run python scripts/download_data.py --source hillstrom

# Other modes:
poetry run python scripts/download_data.py --source criteo
poetry run python scripts/download_data.py --source kaggle --slug owner/dataset
```

Raw data should remain outside version control (`data/raw/*` is gitignored
except for `.gitkeep`).
