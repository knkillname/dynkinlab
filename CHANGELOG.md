# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- A script-first DynkinLab masterclass (`notebooks/dynkin-masterclass.py` plus the generated Colab
  notebook) that teaches quasi-Cartan matrices, exact inertia, faithful bigraphs, flation, finite
  types, A9, D13, and A9-to-D8 from first principles with a fixed tour and parameter labs.
- `Bigraph.from_paths` for the six-operator path notation, including explicit label order, isolated
  vertices, comments, conflict detection, and configurable arrow multiplicity.
- `QuasiCartanMatrix.has_faithful_bigraph` and display-only `labels` on `to_markdown`/`to_html`,
  with optional collapsed HTML for wide matrices.
- `Bigraph.layout`, `Bigraph.draw`, and `BlockCutTree.draw`, driven by NetworkX layouts, plus the
  `DisplayOptions.bigraph_layout` policy. `Bigraph.to_html` now shares the same layout engine.
- Markdown summaries and notebook hooks for A-type, D-core gluing, and A-cyclization certificates,
  and for flation sequences.
- A VS Code Dev Container based on the latest Google Colab CPU runtime, with Python/Jupyter
  extensions, a notebook-ready Hatch environment, and OpenCode.
- Non-symmetrizable `QuasiCartanMatrix` construction now raises `NotSymmetrizableError` carrying a
  verifiable `SymmetrizerCertificate` for zero entries, opposite signs, or inconsistent cycles.
- NumPy 2.1 or newer is now a declared runtime dependency, used for the classifier's bounded
  integer state.
- The initial typed `dynkinlab` package skeleton, built with Hatchling and targeting Python 3.13.
- Immutable `QuasiCartanMatrix` values for symmetrizable integer matrices with diagonal 2, canonical
  minimal-trace symmetrizers, a strict `symmetric()` factory, and generalized flation.
- Exact integer `QuadraticForm` evaluation and exact `Inertia` computation on the symmetrized matrix
  `S = D A` by rational congruence reduction, including a public `Inertia.from_matrix` entry point.
- Label-generic `Bigraph` views over `QuasiCartanMatrix`, mixed edge/arrow derivation, and
  first-class `FlationStep` / `FlationSequence` witnesses.
- Structural bigraph inspection with `to_networkx`, block-cut forests, A-block partitions, and
  verifiable A-type certificates.
- Structural D-type certificates: `EdgeMultiset`, `Bigraph.sum`, `Bigraph.identify`, D-core gluing
  certificates, and A-cyclization certificates.
- `DynkinType` for the full finite A–G list and an exact cubic `DynkinClassification`
  classifier with exact inertia, normal forms, witnesses, and a documented G2 extension.
- Colab/Jupyter Markdown and HTML rendering for matrices, forms, inertia, mixed bigraphs, and Dynkin
  results, with notebook-wide `dynkinlab.display.options` and per-call `DisplayOptions` overrides.
- Colab compatibility assertions that run against the dev container's Python.
- Sphinx documentation, a versioned pre-push hook, and a CI workflow for checks, tests, packaging,
  and documentation.

### Changed

- The automatic bigraph layout now prefers Kamada-Kawai for connected graphs when SciPy is
  available and otherwise falls back to an unweighted, energy-based spring layout with a configurable
  iteration count, defaulting to 10000 iterations. Planar graphs with a four-vertex clique use a
  planar seed followed by the same spring refinement, avoiding the crossing diagonals that
  force-directed layouts often produce for a K4 block. Planar layout remains available as an
  explicit policy.
- **Breaking:** `Bigraph.edges` now raises `ValueError` when the underlying matrix has no faithful
  bigraph representation, including the `2/4` matrix. Previously it returned a lossy
  multiplicity-four arrow. The classifier is unaffected because it reads entries directly.
- The thesis-companion notebook was replaced by the ground-up masterclass notebook under
  `notebooks/`; the old files and the `thesis-notebook` capability were retired.
- Admissible indefinite components are now rejected by exact inertia with reason `indefinite`, zero
  flation steps, and an empty witness. Non-negative components keep the bounded flation behavior.
- `SymmetrizedMatrix` was removed from `dynkinlab.matrix`; `MatrixLike` remains the read-only
  protocol used by `QuadraticForm` and `Inertia`.
- Developer verification now runs through Hatch (`hatch run verify` and `hatch run colab:smoke`); the
  `scripts/` directory was removed. Colab compatibility is checked inside the dev container, so no
  separate image is downloaded.
- Type checking now runs Pyrefly in strict mode across `src`, `tests`, and `tools`, treats warnings as
  failures, and uses NetworkX type stubs.
