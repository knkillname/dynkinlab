# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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

- Admissible indefinite components are now rejected by exact inertia with reason `indefinite`, zero
  flation steps, and an empty witness. Non-negative components keep the bounded flation behavior.
- `SymmetrizedMatrix` was removed from `dynkinlab.matrix`; `MatrixLike` remains the read-only
  protocol used by `QuadraticForm` and `Inertia`.
- Developer verification now runs through Hatch (`hatch run verify` and `hatch run colab:smoke`); the
  `scripts/` directory was removed. Colab compatibility is checked inside the dev container, so no
  separate image is downloaded.
- Type checking now runs Pyrefly in strict mode across `src`, `tests`, and `tools`, treats warnings as
  failures, and uses NetworkX type stubs.
