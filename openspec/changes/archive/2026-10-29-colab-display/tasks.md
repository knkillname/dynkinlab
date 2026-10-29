# Tasks

## 1. Display options

- [x] 1.1 Create `src/dynkinlab/display.py` with a mutable `DisplayOptions` exposing `expanded_form_dimension_limit` and `matrix_index_base` as validated integer fields, a `reset()` method, and a canonical `options` instance; add `tests/test_display.py` covering field defaults, valid mutation, invalid values, and reset; verify `hatch test tests/test_display.py --no-cov` passes.
- [x] 1.2 Import the `display` submodule from `dynkinlab/__init__.py`, add a test that `dynkinlab.display.options` is the canonical instance, and add a DisplayOptions section to `docs/reference/index.md`; verify `hatch test tests/test_display.py --no-cov` and `hatch run docs:build` pass.

## 2. Matrix representations

- [x] 2.1 Add `QuasiCartanMatrix.to_markdown(options=None)` and `to_html(options=None)` that render row and column headers using the effective `matrix_index_base`; add tests for zero-based and one-based Markdown tables, HTML table structure, and entries; verify `hatch test tests/test_display.py --no-cov` passes.
- [x] 2.2 Add `QuasiCartanMatrix._repr_html_` delegating to `to_html()` with the canonical options, test the hook and canonical-options behavior, and add a matrix display section to `docs/reference/index.md`; verify the focused display tests and `hatch run docs:build` pass.

## 3. Quadratic form representation

- [x] 3.1 Add `QuadraticForm.to_markdown(options=None)` with the general formula and an expanded polynomial when the dimension is at or below `expanded_form_dimension_limit`, omitting zero terms and simplifying unit coefficients; add tests for expansion, the general-form fallback, zero coefficients, and negative coefficients; verify `hatch test tests/test_display.py --no-cov` passes.
- [x] 3.2 Add `QuadraticForm._repr_markdown_` delegating to `to_markdown()`, test the hook and canonical-options behavior, and add a form display section to `docs/reference/index.md`; verify the focused display tests and `hatch run docs:build` pass.

## 4. Inertia representation

- [x] 4.1 Add `Inertia.to_markdown()` with positive, negative, zero, rank, and signature counts plus a definiteness or degeneracy label; add tests for positive definite, negative definite, degenerate, and indefinite results; verify `hatch test tests/test_display.py --no-cov` passes.
- [x] 4.2 Add `Inertia._repr_markdown_` delegating to `to_markdown()`, test the hook, and add an inertia display section to `docs/reference/index.md`; verify the focused display tests and `hatch run docs:build` pass.

## 5. Cross-object display behavior and user documentation

- [x] 5.1 Add tests that per-call `DisplayOptions` beats the canonical object, canonical changes affect hooks, `reset()` restores behavior, and `repr()` remains dataclass-style; verify `hatch test tests/test_display.py --no-cov` passes.
- [x] 5.2 Update `README.md`, `docs/getting-started/quickstart.md`, `notebooks/README.md`, and `CHANGELOG.md` with the display API, notebook-wide options, per-call overrides, and hook behavior; verify `hatch test tests/test_readme.py --no-cov` and `hatch run docs:build` pass.

## 6. Integration verification

- [x] 6.1 Run `hatch check --fix` twice and then `hatch test`; verify the second check leaves the tree unchanged and the full suite enforces at least 80% coverage.
- [x] 6.2 Run `scripts/verify --skip-colab`; verify the host quality gates pass and the Colab gate is reported as skipped.
- [x] 6.3 Run `scripts/verify`; verify the pinned Colab smoke gate still passes with Python 3.13 and the dependency floors.
- [x] 6.4 Rebuild the wheel, install it in a clean virtual environment, and render all three object types; verify Markdown and HTML output work without the source tree and `hatch run docs:build` passes with warnings treated as errors.
- [x] 6.5 Inspect the display implementation and public surface; verify no IPython dependency, no public `DEFAULT_*` constants, no bare numeric display thresholds, and unchanged dataclass `repr` output.

## Workflow follow-up

- Archive the change after review and a green full verification run.
- Create the Git commit for the archived change, as required by the project's archive guidance.
