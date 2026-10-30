# GitHub-only releases

This project publishes stable versions as GitHub Releases. It does **not**
publish packages to PyPI or any other package registry.

The [GitHub Release Assets workflow](../.github/workflows/release-github.yml)
starts when a stable GitHub Release is published. It builds and validates a
wheel and source distribution from the release tag, then attaches both files
directly to that GitHub Release. The workflow uses the built-in
`GITHUB_TOKEN`, scoped to repository contents, and requires no PyPI credentials
or Trusted Publisher configuration.

## Release checklist

1. Confirm that `tool.poetry.version` in `pyproject.toml` and `__version__` in
   `src/uplift_marketing/__init__.py` match the intended stable version.
2. Run the release preflight locally against the release-candidate commit:

   ```bash
   poetry install --with dev
   poetry run pytest
   poetry run ruff check .
   poetry run mypy src
   poetry build
   poetry run python scripts/verify_release.py --tag v0.6.0 --dist dist
   ```

   Replace `v0.6.0` with the intended release tag. Check that CI is green
   across all supported Python versions. If real experimental data is not
   available, record that limitation; synthetic data is not a substitute
   for a real-data validation.
3. Merge any release changes through a pull request. Confirm that the desired
   commit is on `main` before creating a tag.
4. In GitHub, create a new Release from the `main` commit using the stable tag
   `vMAJOR.MINOR.PATCH` (for example, `v0.6.0`). Include concise release notes.
5. Publish the GitHub Release. Watch the **GitHub Release Assets** workflow and
   confirm both the `.whl` and `.tar.gz` appear under its downloadable assets.
   The workflow never uploads either artifact to PyPI.
6. Optionally verify a downloaded wheel in a clean environment. This is a
   direct GitHub asset installation, not an installation from a package index.

The release event triggers the asset workflow only for stable releases;
prereleases are skipped because `verify_release.py` requires a stable
`vMAJOR.MINOR.PATCH` tag. If the upload step fails, inspect the workflow logs
and repository Actions permissions before retrying. A rerun against existing
assets may report that the files already exist, which prevents an unintended
overwrite.
