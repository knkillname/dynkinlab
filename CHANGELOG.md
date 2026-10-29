# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
- `DynkinType` for the full finite A–G list and a paper-faithful cubic `DynkinClassification`
  classifier with exact inertia, normal forms, witnesses, and a documented G2 extension.
- Colab/Jupyter Markdown and HTML rendering for matrices, forms, inertia, mixed bigraphs, and Dynkin
  results, with notebook-wide `dynkinlab.display.options` and per-call `DisplayOptions` overrides.
- A pinned Colab compatibility smoke gate for the official CPU runtime image.
- Sphinx documentation, a versioned pre-push hook, and a CI workflow for checks, tests, packaging,
  documentation, and Colab compatibility.

### Changed

- Admissible indefinite components are now rejected by exact inertia with reason `indefinite`, zero
  flation steps, and an empty witness. Non-negative components keep the bounded flation behavior.
- `SymmetrizedMatrix` was removed from `dynkinlab.matrix`; `MatrixLike` remains the read-only
  protocol used by `QuadraticForm` and `Inertia`.
