# Tasks

## 1. Repository and package skeleton

- [x] 1.1 Initialize Git and add `.gitignore` entries for `.venv`, `__pycache__`, `.pytest_cache`, `.ruff_cache`, `.coverage`, `dist`, `docs/_build`, and `*.egg-info`; verify each path is ignored with `git check-ignore` and that `git status` shows only intended files.
- [x] 1.2 Create `pyproject.toml` with Hatchling, project metadata, `requires-python = ">=3.13"`, runtime dependencies `networkx>=3.7` and `matplotlib>=3.10`, a `docs` extra, dynamic version from `src/dynkinlab/__about__.py`, and Ruff, Pyrefly, pytest, coverage, and Hatch test configuration; verify `hatch project metadata` reports the Python and dependency floors.
- [x] 1.3 Create `src/dynkinlab/__about__.py`, `src/dynkinlab/__init__.py`, `src/dynkinlab/py.typed`, and `tests/test_package.py` with SPDX headers; verify `hatch check --fix` and `hatch test` pass and the public surface exposes only `__version__`.
- [x] 1.4 Add the MIT `LICENSE` and a `CHANGELOG.md` with an `Unreleased` section following Keep a Changelog; verify the files exist and the changelog has no released version entry.
- [x] 1.5 Build the wheel with `uv build` and install it into a clean virtual environment; verify `import dynkinlab` succeeds, the version matches package metadata, and the wheel contains `dynkinlab/py.typed` and `LICENSE`.

## 2. Documentation baseline

- [x] 2.1 Create `README.md` with install and quickstart instructions and one fenced Python snippet, plus `tests/test_readme.py` that executes the snippet; verify `hatch test tests/test_readme.py` passes.
- [x] 2.2 Create the Sphinx/MyST/furo skeleton under `docs/` with a landing page, quickstart page, reference placeholder, and toctree; add the docs dependencies to the `docs` extra; verify `hatch run docs:build` passes with warnings treated as errors.
- [x] 2.3 Create `notebooks/README.md` describing the Colab workflow and the future quickstart notebook; verify the documentation links to it and `hatch run docs:build` still passes.

## 3. Colab compatibility gate

- [x] 3.1 Create a single-source Colab image reference containing the digest `sha256:17929077afbae4b805c774450b50ed6945daa122107ac0fb492ce057bc0c3539`; verify the reference contains a `sha256:` digest and does not use `latest`.
- [x] 3.2 Create `scripts/colab-smoke` to run the pinned CPU image read-only, assert Python 3.13, assert NetworkX `>=3.7` and matplotlib `>=3.10`, and import `dynkinlab` from `src`; verify the script prints all versions and exits successfully.
- [x] 3.3 Create `scripts/verify` with `--skip-colab`; verify `scripts/verify --skip-colab` runs `hatch check`, `hatch test`, and reports the Colab gate as skipped.
- [x] 3.4 Add the versioned `.githooks/pre-push` hook that runs `scripts/verify --skip-colab`; verify `bash -n .githooks/pre-push` passes and the hook exits successfully after enabling `git config core.hooksPath .githooks`.
- [x] 3.5 Add `.github/workflows/ci.yml` with `check`, `test`, `package`, `colab`, and `docs` jobs; verify the workflow declares all five jobs and the `colab` job references the pinned image digest.
- [x] 3.6 Document the Colab image digest, smoke command, `--skip-colab` behavior, and CI split in `docs/contributing/colab.md`; verify the page is linked from the docs navigation and `hatch run docs:build` passes.

## 4. Integration verification

- [x] 4.1 Run `hatch check --fix` twice and then `hatch test`; verify the second check leaves the tree unchanged and the test run enforces at least 80% coverage.
- [x] 4.2 Run `scripts/verify --skip-colab`; verify the host quality gates pass and the Colab gate is explicitly reported as skipped.
- [x] 4.3 Run `scripts/verify` without `--skip-colab`; verify the pinned Colab smoke gate passes with Python 3.13, NetworkX, matplotlib, and the package import.
- [x] 4.4 Rebuild the wheel and docs; verify the clean wheel install still imports, the wheel contains `py.typed` and `LICENSE`, and the docs build passes with warnings treated as errors.
- [x] 4.5 Inspect the public package surface and source tree; verify only `__version__` is public and no `QuasiCartanMatrix`, `Bigraph`, inertia, flation, or classifier code was added.

## Workflow follow-up

- Archive the change after review and a green full verification run.
- Create the Git commit for the archived change.
- Start the follow-up change for `QuasiCartanMatrix`, `QuadraticForm`, `Inertia`, and exact inertia; confirm whether inertia or flation dynamics is the first mathematical milestone.
