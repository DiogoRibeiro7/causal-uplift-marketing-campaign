"""Train an uplift model and write scoring outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from uplift_marketing.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/sample_uplift.csv"))
    parser.add_argument("--model", choices=["s_learner", "t_learner"], default="t_learner")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_pipeline(args.input, model_name=args.model)
    print(f"Saved model to {outputs['model_path']}")
    print(f"Saved scored customers to {outputs['scored_path']}")


if __name__ == "__main__":
    main()
