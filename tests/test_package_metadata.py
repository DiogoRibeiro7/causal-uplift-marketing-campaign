"""Tests that keep the published distribution and package version in sync."""

from __future__ import annotations

from importlib.metadata import metadata, version

from uplift_marketing import __version__


def test_distribution_version_matches_package_version() -> None:
    """The installed package and Python API must advertise the same version."""

    assert version("causal-uplift-marketing-campaign") == __version__


def test_distribution_metadata_has_expected_identity() -> None:
    """Expose the expected project name, license, and repository location."""

    project = metadata("causal-uplift-marketing-campaign")
    assert project["Name"] == "causal-uplift-marketing-campaign"
    assert project["License"] == "MIT"
    project_urls = project.get_all("Project-URL", [])
    assert any(
        "github.com/DiogoRibeiro7/causal-uplift-marketing-campaign" in url
        for url in project_urls
    )


def test_optional_integrations_are_not_base_dependencies() -> None:
    """Kaggle and booster packages must only install when extras are requested."""

    requires = metadata("causal-uplift-marketing-campaign").get_all("Requires-Dist", [])
    optional_extras = {
        "kaggle": "kaggle",
        "python-dotenv": "kaggle",
        "xgboost": "boosters",
        "lightgbm": "boosters",
    }
    for dependency, extra in optional_extras.items():
        matching = [
            requirement for requirement in requires if requirement.split(" ", 1)[0] == dependency
        ]
        assert len(matching) == 1, f"Expected one requirement for {dependency}"
        assert f'extra == "{extra}"' in matching[0]

    assert not any(requirement.startswith("typer ") for requirement in requires)
