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

1. Confirm that `tool.poetry.version` in `pyproject.toml`, `__version__` in
   `src/uplift_marketing/__init__.py`, `.zenodo.json` (`version` and versioned
   title), and `CITATION.cff` (`version` and `date-released`) describe the same
   stable release. Keep the substantive description and keywords up to date.
2. Run the release preflight locally against the release-candidate commit:

   ```bash
   poetry install --with dev
   poetry run pytest
   poetry run ruff check .
   poetry run mypy src
   poetry run python scripts/verify_citation_metadata.py --tag v0.6.0
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

## Zenodo archiving and citation

- Zenodo reads `.zenodo.json` at the release tag. When both files exist,
  `.zenodo.json` takes precedence over `CITATION.cff` for Zenodo metadata.
  GitHub uses `CITATION.cff` to offer **Cite this repository**.
- The CFF file uses the **concept DOI**
  [10.5281/zenodo.23803160](https://doi.org/10.5281/zenodo.23803160), which
  identifies the evolving project. For reproducible citations of this exact
  version, use the **v0.6.0 DOI**
  [10.5281/zenodo.23803161](https://doi.org/10.5281/zenodo.23803161).
  Future Zenodo archives receive their own version DOIs; do not copy a previous
  version's DOI into `.zenodo.json` or `CITATION.cff`.
- Always update both metadata files *before* publishing a new tag. CI tests
  compare their authors, identifiers, keywords, titles, and versions. The
  GitHub Release Assets workflow also checks the metadata in the tagged source
  snapshot before attaching distributions. There is no PyPI publication.
- Adding these files after the v0.6.0 GitHub Release does **not** change its
  existing Zenodo archive automatically. Corrections to the published record's
  descriptive metadata must be made by editing that record on Zenodo.
