"""Regression tests for guarded PyPI release artifact validation."""

from __future__ import annotations

import subprocess
import sys
import tarfile
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pytest

PROJECT_NAME = "causal-uplift-marketing-campaign"
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_release.py"


def _run_verifier(tag: str, dist: Path) -> subprocess.CompletedProcess[str]:
    """Execute the standalone script as the workflow does."""
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--tag", tag, "--dist", str(dist)],
        capture_output=True,
        text=True,
        check=False,
    )


def _assert_rejected(tag: str, dist: Path, reason: str) -> None:
    """Assert invalid releases produce actionable command-line errors."""
    result = _run_verifier(tag, dist)
    assert result.returncode != 0
    assert reason in result.stderr


def _fake_artifacts(
    dist: Path,
    *,
    wheel_version: str = "0.6.0",
    sdist_version: str = "0.6.0",
    module_version: str = "0.6.0",
    project_name: str = PROJECT_NAME,
    include_module: bool = True,
) -> None:
    """Build small, valid-enough archives with independently configurable metadata."""
    dist.mkdir()
    normalized_name = "causal_uplift_marketing_campaign"
    wheel = dist / f"{normalized_name}-0.6.0-py3-none-any.whl"
    wheel_metadata = (
        f"Metadata-Version: 2.4\nName: {project_name}\nVersion: {wheel_version}\n"
    )
    with ZipFile(wheel, "w") as archive:
        archive.writestr(f"{normalized_name}-0.6.0.dist-info/METADATA", wheel_metadata)
        if include_module:
            archive.writestr(
                "uplift_marketing/__init__.py", f'__version__ = "{module_version}"\n'
            )

    source = dist / f"{normalized_name}-0.6.0.tar.gz"
    source_metadata = (
        f"Metadata-Version: 2.4\nName: {project_name}\nVersion: {sdist_version}\n"
    ).encode()
    with tarfile.open(source, "w:gz") as archive:
        member = tarfile.TarInfo(f"{normalized_name}-0.6.0/PKG-INFO")
        member.size = len(source_metadata)
        archive.addfile(member, BytesIO(source_metadata))


def test_matching_tag_and_artifacts(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    _fake_artifacts(dist)

    result = _run_verifier("v0.6.0", dist)
    assert result.returncode == 0, result.stderr
    assert "Validated wheel, sdist" in result.stdout


@pytest.mark.parametrize("tag", ["0.6.0", "v0.6", "v0.6.0rc1", "v01.6.0"])
def test_requires_stable_tag(tmp_path: Path, tag: str) -> None:
    dist = tmp_path / "dist"
    _fake_artifacts(dist)

    _assert_rejected(tag, dist, "stable vMAJOR.MINOR.PATCH")


def test_rejects_wrong_wheel_version(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    _fake_artifacts(dist, wheel_version="0.5.0")

    _assert_rejected("v0.6.0", dist, "metadata version")


def test_rejects_wrong_sdist_version(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    _fake_artifacts(dist, sdist_version="0.5.0")

    _assert_rejected("v0.6.0", dist, "metadata version")


def test_rejects_wrong_module_version(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    _fake_artifacts(dist, module_version="0.5.0")

    _assert_rejected("v0.6.0", dist, "module version")


def test_rejects_wrong_distribution_name(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    _fake_artifacts(dist, project_name="not-this-package")

    _assert_rejected("v0.6.0", dist, "unexpected distribution name")


def test_rejects_missing_package_module(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    _fake_artifacts(dist, include_module=False)

    _assert_rejected("v0.6.0", dist, "__init__.py is missing")


def test_requires_both_distribution_formats(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()

    _assert_rejected("v0.6.0", dist, "exactly one wheel and one")


def test_rejects_distributions_from_different_releases(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    _fake_artifacts(dist, wheel_version="0.7.0", sdist_version="0.7.0", module_version="0.7.0")

    _assert_rejected("v0.6.0", dist, "metadata version")
