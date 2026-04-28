.PHONY: install data sample-data pipeline train test lint format typecheck clean

install:
	poetry install --with dev

data:
	poetry run python scripts/download_data.py --source criteo

sample-data:
	poetry run python scripts/generate_sample_data.py --rows 5000 --output data/raw/sample_uplift.csv

pipeline:
	poetry run python scripts/run_pipeline.py --input data/raw/sample_uplift.csv

train:
	poetry run python scripts/train_model.py --input data/raw/sample_uplift.csv

test:
	poetry run pytest -q

lint:
	poetry run ruff check .

format:
	poetry run ruff format .

typecheck:
	poetry run mypy src

clean:
	rm -rf data/interim/* data/processed/* reports/figures/* reports/*.csv reports/*.json reports/*.joblib
