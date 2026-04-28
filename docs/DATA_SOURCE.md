# Data Source

This project is designed for Criteo-style uplift data with randomized treatment/control assignment.

Expected columns:

```text
treatment   Binary campaign assignment: 1 treated, 0 control
conversion  Binary outcome: 1 converted, 0 did not convert
```

Optional columns:

```text
visit       Binary visit outcome
exposure    Whether the ad was actually seen
f0..fN      Anonymous or engineered customer features
```

For local development, use:

```bash
make sample-data
```

For real data, use:

```bash
poetry run python scripts/download_data.py --source criteo
```

or configure a Kaggle dataset slug:

```bash
export KAGGLE_DATASET_SLUG="<owner>/<dataset>"
poetry run python scripts/download_data.py --source kaggle
```

Raw data should remain outside version control.
