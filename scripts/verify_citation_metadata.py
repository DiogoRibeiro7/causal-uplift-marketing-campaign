"""Validate Zenodo and GitHub citation metadata before a GitHub release.

Both files are versioned with the source code. Zenodo reads ``.zenodo.json``
and GitHub's citation interface reads ``CITATION.cff``. This check prevents
different software versions, titles, authors, or keywords being advertised by
those two services.

Requires PyYAML, installed with the project's development dependencies.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from datetime import date
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STABLE_TAG = re.compile(r"^v(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
CONCEPT_DOI = "10.5281/zenodo.23803160"


def _require(condition: bool, message: str) -> None:
    """Raise an actionable validation error when a metadata invariant fails."""
    if not condition:
        raise ValueError(message)


def _poetry_field(root: Path, field: str) -> str:
    """Read a quoted scalar from the project's ``[tool.poetry]`` section."""
    source = (root / "pyproject.toml").read_text(encoding="utf-8")
    section = re.search(r"(?ms)^\[tool\.poetry\]\s*\n(.*?)(?=^\[|\Z)", source)
    _require(section is not None, "pyproject.toml: [tool.poetry] is missing")
    assert section is not None  # Narrow the type for static analyzers.

    match = re.search(rf'(?m)^{re.escape(field)}\s*=\s*"([^"]+)"\s*$', section[1])
    _require(match is not None, f"pyproject.toml: tool.poetry.{field} is missing")
    assert match is not None
    return match[1]


def _package_version(root: Path) -> str:
    """Extract the package's literal version without importing its dependencies."""
    module_path = root / "src" / "uplift_marketing" / "__init__.py"
    module = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    for statement in module.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in statement.targets
        ):
            _require(
                isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str),
                "uplift_marketing.__version__ must be a string literal",
            )
            assert isinstance(statement.value, ast.Constant)
            assert isinstance(statement.value.value, str)
            return statement.value.value
    raise ValueError("uplift_marketing.__version__ is missing")


def _load_mapping(path: Path) -> dict[str, Any]:
    """Parse JSON or YAML and require a top-level metadata mapping."""
    source = path.read_text(encoding="utf-8")
    parsed = json.loads(source) if path.suffix == ".json" else yaml.safe_load(source)
    _require(isinstance(parsed, dict), f"{path.name}: expected a metadata mapping")
    assert isinstance(parsed, dict)
    return parsed


def verify_citation_metadata(tag: str, root: Path = PROJECT_ROOT) -> None:
    """Check source, citation, and Zenodo metadata for one stable release.

    Args:
        tag: Intended stable GitHub tag, e.g. ``v0.6.0``.
        root: Project directory containing the checked metadata files.

    Raises:
        TypeError: An argument has an unsupported type.
        ValueError: Metadata disagree or required fields are missing.
        OSError: A required repository file cannot be read.
    """
    if not isinstance(tag, str):
        raise TypeError("tag must be a string")
    if not isinstance(root, Path):
        raise TypeError("root must be a pathlib.Path")

    _require(bool(STABLE_TAG.fullmatch(tag)), "tag must be stable vMAJOR.MINOR.PATCH")
    package_version = _package_version(root)
    poetry_version = _poetry_field(root, "version")
    _require(poetry_version == package_version, "pyproject and Python package versions differ")
    _require(tag == f"v{poetry_version}", "release tag does not match package version")

    zenodo = _load_mapping(root / ".zenodo.json")
    citation = _load_mapping(root / "CITATION.cff")

    _require(zenodo.get("version") == tag, ".zenodo.json: version does not match tag")
    _require(
        citation.get("version") == poetry_version,
        "CITATION.cff: version does not match package",
    )
    _require(citation.get("cff-version") == "1.2.0", "CITATION.cff: unsupported schema")
    _require(citation.get("type") == "software", "CITATION.cff: type must be software")
    _require(zenodo.get("upload_type") == "software", ".zenodo.json: software type required")
    _require(zenodo.get("access_right") == "open", ".zenodo.json: access must be open")

    expected_title = citation.get("title")
    _require(isinstance(expected_title, str) and bool(expected_title.strip()), "Missing title")
    _require(
        zenodo.get("title") == f"{expected_title} ({tag})",
        ".zenodo.json: title must agree with CITATION.cff and the release tag",
    )
    description = zenodo.get("description")
    _require(
        isinstance(description, str) and len(description.strip()) >= 300,
        ".zenodo.json: a substantive software description is required",
    )
    abstract = citation.get("abstract")
    _require(isinstance(abstract, str) and len(abstract.strip()) >= 100, "Missing abstract")

    license_id = _poetry_field(root, "license")
    _require(citation.get("license") == license_id, "CITATION.cff: license differs")
    _require(zenodo.get("license") == license_id.lower(), ".zenodo.json: license differs")
    repository_url = _poetry_field(root, "repository")
    _require(
        citation.get("repository-code") == repository_url,
        "CITATION.cff: repository-code differs from package metadata",
    )
    _require(
        citation.get("doi") == CONCEPT_DOI,
        "CITATION.cff: use the permanent concept DOI, not a version-specific DOI",
    )

    zenodo_keywords = zenodo.get("keywords")
    citation_keywords = citation.get("keywords")
    _require(
        isinstance(zenodo_keywords, list)
        and len(zenodo_keywords) >= 6
        and all(isinstance(keyword, str) and keyword.strip() for keyword in zenodo_keywords)
        and zenodo_keywords == citation_keywords,
        "Zenodo and CITATION keywords must match (at least six terms)",
    )

    creators = zenodo.get("creators")
    authors = citation.get("authors")
    _require(
        isinstance(creators, list)
        and len(creators) == 1
        and isinstance(creators[0], dict)
        and isinstance(authors, list)
        and len(authors) == 1
        and isinstance(authors[0], dict),
        "Zenodo creators and CITATION authors must each contain the software author",
    )
    creator, author = creators[0], authors[0]
    _require(
        creator.get("name") == f"{author.get('family-names')}, {author.get('given-names')}",
        "Zenodo creator name differs from CITATION author",
    )
    orcid = creator.get("orcid")
    _require(
        isinstance(orcid, str)
        and bool(re.fullmatch(r"\d{4}-\d{4}-\d{4}-[\dX]{4}", orcid))
        and author.get("orcid") == f"https://orcid.org/{orcid}",
        "Zenodo and CITATION author ORCID identifiers differ",
    )

    release_date = citation.get("date-released")
    _require(isinstance(release_date, str), "CITATION.cff: quoted release date is required")
    try:
        date.fromisoformat(release_date)
    except ValueError as error:
        raise ValueError("CITATION.cff: invalid ISO release date") from error


def main() -> None:
    """Run a release preflight with a nonzero exit code on bad metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="Stable GitHub release tag")
    args = parser.parse_args()
    try:
        verify_citation_metadata(args.tag)
    except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))
    print(f"Validated Zenodo, citation, and package metadata for {args.tag}")


if __name__ == "__main__":
    main()
