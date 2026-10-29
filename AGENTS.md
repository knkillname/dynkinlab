# AGENTS.md

DynkinLab is a typed Python 3.13 library for Dynkin diagrams, quivers, and mixed bigraphs. The
compatibility target is the latest Google Colab CPU runtime, not just the local interpreter.

## Commands

Hatch drives every quality gate; install it once with `uv tool install hatch`.

- `hatch check` — Ruff lint/format plus strict Pyrefly over `src`, `tests`, and `tools`.
- `hatch check --fix` — apply Ruff fixes before re-running the check.
- `hatch test` — full pytest suite; the 80% coverage gate is enforced by `pyproject.toml`.
- `hatch test tests/test_bigraph.py --no-cov` — focused test run. Without `--no-cov`, the coverage
  gate fails even when the selected tests pass. Node IDs also work:
  `hatch test tests/test_package.py::TestPackage::test_public_surface --no-cov`.
- `hatch run verify` — `check`, `test`, then `colab:smoke`; this is the authoritative local gate
  and what the pre-push hook runs.
- `hatch run docs:build` — strict Sphinx build (`-W --keep-going`); warnings fail the build.
- `hatch run colab:smoke` — only meaningful inside the dev container, where `/usr/bin/python3` is
  the Colab runtime.
- `git config core.hooksPath .githooks` — enable the pre-push hook in a fresh clone.

## Architecture

- `QuasiCartanMatrix` (`src/dynkinlab/quasi_cartan.py`) is the central immutable value. Quadratic
  forms, inertia, flation, bigraphs, and classification all derive from it.
- `QuadraticForm` and `Inertia` depend on the `MatrixLike` protocol (`matrix.py`), not directly on
  `QuasiCartanMatrix`; keep that decoupling.
- Public classification entry points are `QuasiCartanMatrix.classify()` and `Bigraph.classify()`.
  `DynkinClassifier` in `classifier.py` is the implementation.
- `quasi_cartan.py` and `bigraph.py` import `DynkinClassifier` inside their `classify()` methods to
  avoid an import cycle. Do not move that import to module scope.
- The public surface is exactly `src/dynkinlab/__init__.py::__all__`, asserted verbatim by
  `tests/test_package.py`; update both when adding a public name.

## Invariants

- Exact arithmetic is a correctness requirement: inertia uses `Fraction` congruence reduction,
  forms stay integer, and the classifier must not use floating-point eigenvalues. Its bounded
  search uses `np.int64`; the public flation path keeps Python integers for arbitrary precision.
- Do not import or access underscore-prefixed names across modules. The only exception is the
  `_repr_*` notebook hooks. `tests/test_package.py` enforces the import rule.
- No `from __future__` imports; Python 3.13 is the floor. NumPy-style docstrings and 100-character
  lines are enforced. Ruff ignores `ANN` in tests, but Pyrefly still type-checks `src`, `tests`, and
  `tools`, including unannotated definitions.
- README fenced Python blocks are executed by `tests/test_readme.py`; keep every example
  self-contained and runnable. Only README snippets get that coverage, so verify docs examples
  separately; the strict docs build must stay warning-free.
- Notebook rendering must not depend on IPython. `_repr_*` hooks must match `to_markdown()` /
  `to_html()` using canonical `dynkinlab.display.options`, while `repr()` stays dataclass-style.
- The package version is defined only in `src/dynkinlab/__init__.py` (Hatch reads it dynamically).
  CI asserts installed metadata matches it. Record user-visible changes under `CHANGELOG.md` →
  `[Unreleased]`.
- The Colab compatibility checks pin the interpreter and dependency floors; when changing
  `pyproject.toml` dependencies or relying on a newer runtime, update `tools/colab_smoke.py` too.
- Classifier behavior is guarded by fixed golden matrices and a Hypothesis oracle that compares the
  optimized classifier against a naive reference; preserve both when changing search bounds.

## OpenSpec

Substantial changes are planned under `openspec/changes/<slug>/` with proposal, design, and task
files, then archived. Main specs live in `openspec/specs/`. Read `openspec/config.yaml` for extra
constraints, use the `openspec-*` skills, and create a Git commit after each archive.
