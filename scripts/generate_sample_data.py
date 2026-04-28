"""Generate a local synthetic uplift dataset for development."""

from __future__ import annotations

import argparse
from pathlib import Path

from uplift_marketing.data import generate_synthetic_uplift_data, save_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=5_000, help="Number of synthetic rows to generate.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/sample_uplift.csv"),
        help="Output CSV path.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = generate_synthetic_uplift_data(n_rows=args.rows, random_state=args.seed)
    save_csv(data, args.output)
    print(f"Generated {len(data):,} synthetic rows at {args.output}")


if __name__ == "__main__":
    main()
