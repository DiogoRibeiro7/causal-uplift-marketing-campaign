"""Download uplift marketing data.

The script supports two modes:

1. Direct URL download for a Criteo-style CSV or CSV.GZ file.
2. Kaggle dataset download using the `KAGGLE_DATASET_SLUG` environment variable.

Raw data is not committed to the repository.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

DEFAULT_CRITEO_URL = "https://storage.googleapis.com/criteo-uplift-v2.1/criteo-uplift-v2.1.csv.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["criteo", "kaggle"], default="criteo")
    parser.add_argument("--url", default=DEFAULT_CRITEO_URL, help="Direct URL used when source=criteo.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    return parser.parse_args()


def download_direct_url(url: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = url.rstrip("/").split("/")[-1] or "uplift_data.csv.gz"
    output_path = output_dir / filename
    print(f"Downloading {url}")
    print(f"Saving to {output_path}")
    urllib.request.urlretrieve(url, output_path)  # noqa: S310 - user-controlled CLI utility.
    return output_path


def ensure_kaggle_ready() -> None:
    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "Kaggle CLI not found. Install it with `poetry install` or `pip install kaggle`."
        )

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists() and not (
        os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")
    ):
        raise RuntimeError(
            "Kaggle credentials are missing. Configure ~/.kaggle/kaggle.json or set "
            "KAGGLE_USERNAME and KAGGLE_KEY."
        )


def download_kaggle_dataset(output_dir: Path) -> None:
    ensure_kaggle_ready()
    slug = os.getenv("KAGGLE_DATASET_SLUG")
    if not slug:
        raise RuntimeError(
            "Set KAGGLE_DATASET_SLUG before using --source kaggle, for example: "
            "export KAGGLE_DATASET_SLUG='<owner>/<dataset>'"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    command = ["kaggle", "datasets", "download", "-d", slug, "-p", str(output_dir)]
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)

    for archive in output_dir.glob("*.zip"):
        print(f"Unzipping {archive}")
        with zipfile.ZipFile(archive) as zip_file:
            zip_file.extractall(output_dir)


def main() -> None:
    args = parse_args()
    try:
        if args.source == "criteo":
            download_direct_url(args.url, args.output_dir)
        else:
            download_kaggle_dataset(args.output_dir)
    except Exception as exc:  # pragma: no cover - CLI guard with clear message.
        print(f"Data download failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
