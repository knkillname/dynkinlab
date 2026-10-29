# Proposal

## Why

The 2017 DynkinLab prototype is a Python 3.5 REPL with a Tkinter window, random algorithms, and no typed or tested package boundary. The new version is Colab-first: users should import a typed library in a notebook and visualize bigraphs inline. Before adding mathematical behavior, the project needs a reproducible package skeleton whose supported Python and dependency floors match the current Colab runtime.

The current Google Colab CPU image (`release-colab-external-images_20261027-060057_RC00`, digest `sha256:17929077afbae4b805c774450b50ed6945daa122107ac0fb492ce057bc0c3539`) runs Python 3.13.18 and ships NetworkX 3.7 and matplotlib 3.10.0. Pinning those as floors prevents a local-only design that fails in the target environment.

## What Changes

- Create a Hatch-managed, typed `src/dynkinlab` package with `py.typed`, `__about__.py`, and a minimal public `__init__`.
- Add `pyproject.toml` with Hatchling, Python `>=3.13`, `networkx>=3.7`, `matplotlib>=3.10`, Ruff `py313`, Pyrefly `3.13`, pytest, and coverage.
- Make `hatch check --fix` and `hatch test` the only quality gates. Do not add `hatch fmt`, `hatch run check`, a separate type-check environment, or a parallel lint/test workflow.
- Add a Python 3.13 Hatch test matrix and an 80% coverage gate.
- Add a pinned Colab smoke script, `scripts/colab-smoke`, and a `scripts/verify` entry point with `--skip-colab` for fast local runs. The smoke script runs the official `cpu-runtime` image by digest, checks Python 3.13 and the NetworkX/matplotlib floors, and imports the package.
- Add `scripts/verify` for the host gates and a versioned pre-push hook. The full Colab smoke gate belongs in a dedicated cached CI job rather than every unit-test run.
- Add the project baseline: README with a runnable Colab install example, CHANGELOG (Keep a Changelog), LICENSE (MIT), SPDX headers, Sphinx/MyST documentation skeleton, and a notebooks directory.
- Add CI jobs for `hatch check`, `hatch test`, wheel build plus clean install, and the pinned Colab smoke gate.

This change does not add the mathematical API. `QuasiCartanMatrix`, `QuadraticForm`, `Inertia`, `Bigraph`, flation, and exact inertia are the next change.

## Capabilities

### New Capabilities

- `package-foundation`: installable typed package, dependency floors, Hatch quality gates, test and coverage baseline.
- `colab-compatibility`: pinned Colab runtime smoke gate and environment assertions.

### Modified Capabilities

- None.

## Impact

- New package and build files: `pyproject.toml`, `src/dynkinlab/`, `tests/`, `docs/`, `notebooks/`, `scripts/`.
- New runtime dependencies: `networkx>=3.7`, `matplotlib>=3.10`.
- New development dependencies: Hatch 1.18+, Ruff, Pyrefly, pytest, Hypothesis, coverage.
- Minimum Python changes to 3.13.
- No existing code or specs are modified.
- The core mathematical library is deferred to the follow-up change.
