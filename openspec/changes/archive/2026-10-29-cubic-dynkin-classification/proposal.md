# Proposal

## Why

DynkinLab can currently build only symmetric quasi-Cartan matrices, evaluate their exact quadratic forms, and compute exact inertia, but it cannot represent bigraphs, apply flations, or return a Dynkin type. The thesis and the 2018 cubic paper provide a complete exact algorithm for positive definite quasi-Cartan matrices, including non-symmetric ones; this change turns that into the library's first end-to-end bigraph-to-Dynkin-type slice.

## What Changes

- Broaden `QuasiCartanMatrix` to accept any integer matrix with diagonal 2 and a canonical positive diagonal symmetrizer `D` such that `D A` is symmetric. It exposes `A`, `D`, and `S = D A`, computes the exact quadratic form and inertia from `S`, and implements generalized flation. Symmetric inputs keep `D = (1, ..., 1)` and all existing form, inertia, display, equality, and hashing behavior.
- Add `QuasiCartanMatrix.is_symmetric` and a strict `QuasiCartanMatrix.symmetric(entries)` factory for callers that want the symmetric guarantee. Remove the separate symmetrizable matrix type and the public matrix protocol from the planned surface.
- Add a label-generic `Bigraph` view over `QuasiCartanMatrix`. It derives undirected edges and directed arrows with the paper's signed max-magnitude convention, delegates flation and classification to the matrix, and exposes first-class `FlationStep` and `FlationSequence` witness values.
- Add `DynkinType` for the full finite Cartan-Killing list (A, B, C, D, E6/E7/E8, F4, G2) and a rich `DynkinClassification` result carrying status, component labels and types, the flation witness, the final normal form, step/bound information, and exact inertia.
- Implement the paper-faithful cubic classifier: Stage 1 admissibility filtering, distinguished-vertex and solid-arched preprocessing; Stage 2 bounded inflations using the paper's `l(n)` bound and the non-symmetric reduction to `G - x`; Stage 3 graph classification. G2 receives a documented O(1) pre-check because the paper's `|A_ij| < 3` admissible class excludes it.
- Add Colab rendering for `Bigraph`, `DynkinType`, and `DynkinClassification`, plus display options for layout seed, figure size, node labels, multiplicity labels, and arrow styling. The classification display includes the normal form and a witness summary.
- Update the public exports, README, quickstart, API reference, and changelog while keeping the existing Hatch, coverage, packaging, and Colab gates unchanged.

Out of scope for this change: A-block tree and D-core structural decompositions, affine/Euclidean classification, quivers or non-symmetrizable matrices, a `Bigraph` text DSL or CLI, DOT/TikZ export, and performance work beyond the paper's algorithm.

## Capabilities

### New Capabilities

- `bigraph-flation`: label-generic mixed bigraphs over quasi-Cartan matrices, signed edge/arrow construction, flation delegation, and first-class flation witnesses.
- `dynkin-classification`: full finite Dynkin types, the paper-faithful cubic classification algorithm, rich classification results, normal forms, and the G2 exception.

### Modified Capabilities

- `quasi-cartan-inertia`: `QuasiCartanMatrix` becomes the general symmetrizable matrix, with canonical symmetrization, symmetrized form and inertia, generalized flation, and a strict symmetric factory; existing symmetric behavior is preserved.
- `colab-display`: adds Bigraph, DynkinType, and DynkinClassification rendering plus new display options for mixed graph drawing.

## Impact

- New modules under `src/dynkinlab/` for the bigraph, flation witnesses, Dynkin types and results, and the cubic classifier; updates to `quasi_cartan.py`, `matrix.py`, `quadratic_form.py`, `inertia.py`, `display.py`, and `__init__.py`.
- New tests for construction and symmetrization, generalized flation, bigraph conversion, the three classifier stages, the full finite Dynkin list, G2, witness replay, and Colab rendering.
- Documentation updates in `README.md`, `docs/getting-started/quickstart.md`, `docs/reference/index.md`, and `CHANGELOG.md`.
- No new runtime dependencies: `networkx>=3.7` and `matplotlib>=3.10` are already declared. Existing symmetric APIs remain source-compatible; the constructor now also accepts non-symmetric symmetrizable matrices, while `symmetric()` preserves the strict symmetric validation. The classifier is O(n³) for admissible matrices, with the documented G2 extension.
