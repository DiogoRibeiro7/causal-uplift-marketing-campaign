"""Validate built distributions against a stable GitHub release tag.

The check reads package metadata from both artifacts, without installing or
executing the package. It also verifies the public ``__version__`` in the wheel.
"""

from __future__ import annotations

import argparse
import ast
import re
import tarfile
from email import message_from_bytes
from pathlib import Path
from zipfile import ZipFile

PROJECT_NAME = "causal-uplift-marketing-campaign"
STABLE_TAG = re.compile(r"^v(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


def _normalize_name(name: str) -> str:
    """Normalize a distribution name using the PEP 503 separator rule."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _check_metadata(metadata_bytes: bytes, expected_version: str, origin: Path) -> None:
    """Check the package identity and version from an artifact's metadata."""
    metadata = message_from_bytes(metadata_bytes)
    name = metadata.get("Name")
    version = metadata.get("Version")

    if not isinstance(name, str) or _normalize_name(name) != PROJECT_NAME:
        raise ValueError(f"{origin.name}: unexpected distribution name: {name!r}")
    if not isinstance(version, str) or version != expected_version:
        raise ValueError(
            f"{origin.name}: metadata version {version!r} does not match "
            f"release version {expected_version!r}"
        )


def _wheel_package_version(source: bytes) -> str:
    """Read the literal module version without executing package imports."""
    module = ast.parse(source, filename="uplift_marketing/__init__.py")

    for statement in module.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in statement.targets
        ):
            if isinstance(statement.value, ast.Constant) and isinstance(
                statement.value.value, str
            ):
                return statement.value.value
            raise ValueError("uplift_marketing.__version__ must be a string literal")

    raise ValueError("uplift_marketing.__version__ is missing from the wheel")


def verify_release(tag: str, dist_dir: Path) -> None:
    """Raise ``ValueError`` when the release tag and artifacts disagree.

    Args:
        tag: Stable GitHub release tag such as ``v0.6.0``.
        dist_dir: Directory containing exactly one wheel and one source archive.
    """
    if not isinstance(tag, str):
        raise TypeError("tag must be a string")
    if not isinstance(dist_dir, Path):
        raise TypeError("dist_dir must be a pathlib.Path")

    match = STABLE_TAG.fullmatch(tag)
    if match is None:
        raise ValueError(f"release tag {tag!r} must be a stable vMAJOR.MINOR.PATCH tag")
    version = tag[1:]

    wheels = sorted(dist_dir.glob("*.whl"))
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError("release must contain exactly one wheel and one .tar.gz sdist")

    wheel = wheels[0]
    with ZipFile(wheel) as archive:
        metadata_paths = [
            path for path in archive.namelist() if path.endswith(".dist-info/METADATA")
        ]
        if len(metadata_paths) != 1:
            raise ValueError(f"{wheel.name}: expected one dist-info/METADATA file")
        _check_metadata(archive.read(metadata_paths[0]), version, wheel)

        try:
            module_source = archive.read("uplift_marketing/__init__.py")
        except KeyError as error:
            raise ValueError(f"{wheel.name}: package __init__.py is missing") from error
        module_version = _wheel_package_version(module_source)
        if module_version != version:
            raise ValueError(
                f"{wheel.name}: module version {module_version!r} does not match {version!r}"
            )

    sdist = sdists[0]
    with tarfile.open(sdist, mode="r:gz") as archive:
        metadata_files = [
            member
            for member in archive.getmembers()
            if member.isfile() and member.name.endswith("/PKG-INFO")
        ]
        if len(metadata_files) != 1:
            raise ValueError(f"{sdist.name}: expected one PKG-INFO file")
        metadata_stream = archive.extractfile(metadata_files[0])
        if metadata_stream is None:
            raise ValueError(f"{sdist.name}: cannot read PKG-INFO")
        _check_metadata(metadata_stream.read(), version, sdist)


def main() -> None:
    """Run the check as a GitHub Actions release preflight."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="Release tag, e.g. v0.6.0")
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    arguments = parser.parse_args()
    try:
        verify_release(arguments.tag, arguments.dist)
    except (OSError, ValueError, tarfile.TarError) as error:
        parser.error(str(error))
    print(f"Validated wheel, sdist, and package version for {arguments.tag}")


if __name__ == "__main__":
    main()
