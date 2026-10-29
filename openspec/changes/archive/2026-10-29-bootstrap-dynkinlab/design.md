# Design

## Context

See `proposal.md` for motivation. The repository currently contains only the OpenSpec root and OpenCode command files; there is no Python package, Git history, or CI configuration.

The target environment was verified against the current official Colab CPU image:

- Image: `us-docker.pkg.dev/colab-images/public/cpu-runtime`
- Digest: `sha256:17929077afbae4b805c774450b50ed6945daa122107ac0fb492ce057bc0c3539`
- Release: `release-colab-external-images_20261027-060057_RC00`
- Interpreter: Python 3.13.18
- NetworkX 3.7, matplotlib 3.10.0, NumPy 2.1.3
- Matplotlib backend: `agg`
- Hatch, Hatchling, Ruff, and Pyrefly are not installed in the image.

The host has Hatch 1.18.1 and Docker 29.7.2 on `linux/amd64`. The pulled image occupies 28.3 GB on disk, so it is suitable for a pinned compatibility job, not for every test run.

NetworkX 3.7 does not provide `blockcut_tree`. Future domain changes must assemble the block-cut tree from `biconnected_components` and `articulation_points`; this bootstrap change does not depend on either.

## Goals / Non-Goals

**Goals:**

- A green Hatch package skeleton that installs on Python 3.13.
- The exact Colab dependency floors as runtime dependencies.
- One quality-gate path: `hatch check --fix` and `hatch test`.
- A reproducible, pinned Colab compatibility smoke check.
- A docs, changelog, license, and CI baseline that later changes inherit.

**Non-Goals:**

- `QuasiCartanMatrix`, `QuadraticForm`, `Inertia`, `Bigraph`, flation, or exact inertia.
- NetworkX adapters or visualization code beyond the dependency and smoke checks.
- The Dynkin classifier, certificates, affine types, or notebook tutorials.
- Supporting Python versions below 3.13.
- Making the Colab image the primary development container.

## Decisions

### Hatch is the only workflow

Use Hatch 1.18’s built-in commands:

```text
hatch check --fix   format -> lint -> types
hatch test          pytest in the hatch-test environment
```

Do not add `hatch fmt`, `hatch run check`, a separate type-check environment, or a Makefile that duplicates them. Ruff, Pyrefly, pytest, and coverage configuration live in `pyproject.toml`.

Alternative considered: pre-commit plus separate Ruff/Pyrefly commands. Rejected because it creates a second quality path and the user standard is a single Hatch gate.

### Python 3.13 and dependency floors are hard constraints

`requires-python = ">=3.13"`; the Hatch test matrix contains Python 3.13. The runtime dependency list is:

```toml
dependencies = [
  "networkx>=3.7",
  "matplotlib>=3.10",
]
```

Both are hard dependencies because the first user-facing milestone is a Colab notebook library with inline graph visualization. A lean `[viz]` extra is a possible later split, but it would add conditional-import branches now.

Alternative considered: leave matplotlib optional. Rejected for this bootstrap because the pinned Colab image already provides it and the project is explicitly Colab-first.

### Package skeleton is intentionally small

The package layout is:

```text
src/dynkinlab/__about__.py    version only
src/dynkinlab/__init__.py     exposes __version__
src/dynkinlab/py.typed        typing marker
tests/test_package.py         import and metadata smoke
```

No mathematical API is created. This keeps the first apply run focused on making the toolchain green and gives the next change a clean base. The package uses the house style from the start: SPDX headers, strict type hints, numpydoc docstrings, frozen dataclasses where data objects appear, submodule imports, and no module-level functions.

### Colab smoke uses a pinned CPU image

`scripts/colab-smoke` defines one image reference by digest and runs:

```text
docker run --rm \
  -v "$PWD":/content/dynkinlab:ro \
  -w /content/dynkinlab \
  -e PYTHONPATH=/content/dynkinlab/src \
  --entrypoint /bin/bash \
  <pinned-image> -lc '<checks>'
```

The checks are:

1. `sys.version_info[:2] == (3, 13)`.
2. `networkx.__version__ >= "3.7"` and `matplotlib.__version__ >= "3.10"`.
3. `import dynkinlab` from `/content/dynkinlab/src` succeeds.
4. `dynkinlab.__version__` is a non-empty string.

The mount is read-only. No package is installed into the image’s system Python. If a later change needs a clean-install check, use `uv venv --system-site-packages` inside the container so the Colab packages remain visible.

Alternative considered: `latest` tag. Rejected because the image changes frequently and the smoke result would not be reproducible.

### Verification and CI

`scripts/verify` runs:

1. `hatch check` (no `--fix`; the pre-push path must not rewrite files).
2. `hatch test`.
3. `scripts/colab-smoke` unless `--skip-colab` is passed.

The versioned pre-push hook runs `scripts/verify --skip-colab` so it stays fast. CI has separate jobs:

- `check`: `hatch check`
- `test`: `hatch test`
- `package`: `uv build`, clean-wheel install, verify `py.typed` and `LICENSE`
- `colab`: pull the pinned digest and run `scripts/colab-smoke`
- `docs`: build the Sphinx site with warnings as errors

The Colab job is cached and separate because the image is 28.3 GB on disk and unsuitable for every unit-test run.

### Documentation baseline

Use Sphinx with MyST and furo, mirroring the pygrafito documentation stack. The bootstrap docs contain a landing page, a quickstart with a runnable Colab install example, and a reference placeholder. `myst-nb` is included so later notebooks can be rendered from the same sources.

README Python snippets are executed by the test suite, as in pygrafito. The CHANGELOG follows Keep a Changelog and starts with an `Unreleased` section. The license is MIT and every Python file starts with the SPDX header used by pygrafito.

### Git initialization

The repository is not currently a Git repository. The apply run initializes Git, adds the first commit, and enables the versioned hooks with `git config core.hooksPath .githooks`. The archive step can then follow the existing OpenSpec convention of creating a commit.

## Risks / Trade-offs

- **Image size and CI cost** -> Pin the digest, cache the image, and run the Colab job separately from unit tests.
- **Colab image drift** -> Treat the digest as a versioned dependency; update it deliberately when Colab changes.
- **Hatch test environments are isolated from Colab packages** -> Use the system-site-packages smoke check for parity; Hatch remains the host quality gate.
- **Hard matplotlib dependency** -> Accepted for the Colab-first milestone; revisit a `[viz]` extra if a math-only install becomes a requirement.
- **Root container and host files** -> Mount the project read-only for smoke checks and never mount unrelated host directories.
- **Pyrefly strictness with future NetworkX/matplotlib code** -> Keep third-party types behind small adapter boundaries in later changes; the bootstrap only imports the package itself.
- **`hatch check --fix` can rewrite files** -> Run it before `hatch test`, rerun it to confirm idempotence, and use `hatch check` without `--fix` in verification and CI.

## Migration Plan

This is a greenfield bootstrap; there is no migration. If the first apply run is reverted, remove the generated package, tooling files, hooks, and CI workflow, and leave the OpenSpec change intact.

The follow-up change adds the mathematical core. It is expected to start with `QuasiCartanMatrix`, `QuadraticForm`, `Inertia`, and exact inertia; the flation-versus-inertia priority remains a product decision for that change.

## Open Questions

None for this bootstrap. The choice between exact inertia and flation dynamics belongs to the next change and does not alter this foundation.
