# Releasing to PyPI

This repository builds wheel and source distributions in CI. A separate
GitHub Actions workflow, [publish-pypi.yml](../.github/workflows/publish-pypi.yml),
publishes a stable release to PyPI using short-lived OpenID Connect credentials.
The workflow does not require a PyPI API token secret.

## One-time configuration

1. In GitHub, open **Settings → Environments** for this repository and create an
   environment named `pypi`. Configure required reviewers for approval before
   deployment. Where possible, restrict deployment tags to `v*`.
2. On [PyPI](https://pypi.org/), configure the GitHub Actions Trusted Publisher:
   - If `causal-uplift-marketing-campaign` already exists, open the project,
     then **Manage → Publishing → Add a new publisher**.
   - If it does not exist, use the account **Publishing** page to register a
     *pending publisher* for the project name. This does not reserve the name
     until the first successful publication.
3. Enter the following details in PyPI, matching the workflow exactly:

   | Field | Value |
   | --- | --- |
   | Project name (pending publisher only) | `causal-uplift-marketing-campaign` |
   | GitHub owner | `DiogoRibeiro7` |
   | Repository name | `causal-uplift-marketing-campaign` |
   | Workflow filename | `publish-pypi.yml` |
   | Environment name | `pypi` |

The exact GitHub environment and workflow filename matter: PyPI verifies both
when it grants a temporary publishing credential. Keep deployment approval
enabled so publishing cannot happen without an explicit review.

See the [PyPI Trusted Publishing guide](https://docs.pypi.org/trusted-publishers/)
and the [GitHub environment protection guide](https://docs.github.com/en/actions/reference/workflows-and-actions/deployments-and-environments).

## Release checklist

1. Update both `tool.poetry.version` in `pyproject.toml` and `__version__` in
   `src/uplift_marketing/__init__.py` to the same new version. PyPI does not
   allow uploading a second distribution with an already-published version.
2. Open and merge a pull request with the version change only after the test,
   lint, type-check, and distribution-build CI jobs pass.
3. In GitHub, create a new Release from the merged `main` commit. Use the tag
   `vMAJOR.MINOR.PATCH`, for example `v0.6.0`; do not mark it as a prerelease.
4. Publish the GitHub Release. The `Publish to PyPI` workflow will:
   - Build a wheel and source archive from the release tag.
   - Check both distribution metadata files, the wheel's `__version__`, and
     the release tag agree (via `scripts/verify_release.py`).
   - Pause at the `pypi` environment for deployment approval.
   - Publish the already-checked artifacts to PyPI using Trusted Publishing.
5. Confirm the new version on PyPI and test installation in a clean environment.

The release workflow is not triggered by ordinary pushes or pull requests.
Prerelease GitHub Releases may start a workflow run, but both jobs are skipped.
Do not create a stable GitHub Release until the PyPI publisher and protected
environment have been configured.

To verify local release artifacts before creating a tag:

```bash
poetry build
python scripts/verify_release.py --tag v0.6.0 --dist dist
```

Replace `v0.6.0` with the new version tag when releasing a newer version.
