# Proposal

## Why

The thesis's A/D classification is executable and certifiable, but there is no single artifact a
reader can run to see the matrix-first theory, both D certificates, and the boundary between
matrices and faithful bigraphs. The structural API is ready; the narrative, computed examples, and
runnable Colab entry point are not. A short paper-like notebook closes that gap without turning
the library documentation into a tutorial.

## What Changes

- Add `notebooks/thesis-companion.py`, a percent-format Python source with `# %%` code-cell and
  `# %% [markdown]` Markdown-cell markers, and generate the paired
  `notebooks/thesis-companion.ipynb` for Colab.
- Write the notebook as a self-contained English computational paper; the Python source is the
  editable source of truth, while the generated notebook carries the rendered Colab artifact.
- Make labeled quasi-Cartan matrices the primary objects. Display matrices with labeled rows and
  columns, and draw a bigraph only when the matrix admits a faithful bigraph.
- State and compute three structural results: the A-type certificate for A9, the D-core gluing for
  D13, and the A-cyclization A9 to D8.
- Include deterministic NetworkX/matplotlib figures for A9 and its block-cut tree, D13 with its
  core highlighted, the A9 to D8 cyclization, and a small A3 flation example.
- Use a notebook-local path notation (`--`, `..`, `->`, `<-`, `.>`, `<.`) and local matrix and
  drawing helpers. The notation is a presentation device, not a public library API.
- Add a boundary table covering A9, D13, D8, E6, the thesis mixed B6 example, affine A2, and a
  matrix-only `2/4` case, with inertia, classifier status, drawability, and certificate kind.
- State explicitly that `classify()` is an independent exact bounded-flation oracle, not the
  thesis's Bareiss plus structural polynomial algorithm.
- Add a Colab setup that installs the latest DynkinLab from the project's git default branch, never
  from PyPI, and does not pin a package or dependency version.
- Add a smoke test that parses the Python source and executes the resulting notebook as part of
  `hatch test`, plus the test-only `jupytext`, `nbclient`, and `nbformat` dependencies.
- Update the notebook guide, the quickstart notebook pointer, the API reference's planned surface,
  and the changelog.

Out of scope:

- Changing `Bigraph`'s representability behavior. The notebook uses a local predicate to identify
  matrices without a faithful bigraph; the lossy `2/4` edge behavior is a separate change.
- Making labels first-class on `QuasiCartanMatrix`. Labels remain on the `Bigraph` wrapper, and the
  notebook pairs `matrix` with `labels` for display.
- Implementing the thesis's polynomial classification algorithm or reimplementing the classifier.
- Adding Python files, public names, or runtime dependencies to the package.

## Capabilities

### New Capabilities

- `thesis-notebook`: the script-first, executable, matrix-first thesis companion notebook, its
  worked certificates and figures, its boundary table, its latest-version Colab setup, and its
  smoke execution.

### Modified Capabilities

None. The library's A/D, display, and Colab behavior is unchanged; the notebook consumes the
existing public surface.

## Impact

- New: `notebooks/thesis-companion.py`, generated `notebooks/thesis-companion.ipynb`,
  `tests/test_notebook.py`.
- Modified: `notebooks/README.md`, `docs/getting-started/quickstart.md`, `docs/reference/index.md`,
  `CHANGELOG.md`, and the `hatch-test` extra dependencies in `pyproject.toml`.
- No changes to `src/dynkinlab`, `__all__`, the wheel contents, or runtime dependency floors.
- Local notebook execution and the smoke test use the source checkout. The notebook's Colab setup
  tracks the latest `main`; manual Colab verification requires the implemented notebook commit to
  be pushed to `origin`.
