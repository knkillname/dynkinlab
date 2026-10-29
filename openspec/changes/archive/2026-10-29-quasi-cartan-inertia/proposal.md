# Proposal

## Why

The bootstrap package installs and verifies cleanly but exposes only `__version__`; it has no mathematical behavior. The first mathematical milestone is the quasi-Cartan matrix and its exact quadratic-form inertia, because the later bigraph, flation, and Dynkin-classification work all depend on these invariants. Exact arithmetic is required: floating-point eigenvalues can misclassify degenerate or nearly singular forms.

## What Changes

- Add an immutable `QuasiCartanMatrix` value type for symmetric integer matrices whose diagonal entries are all 2, with construction-time validation.
- Add `QuadraticForm`, evaluating `q(x) = sum(x_i**2) + sum(a_ij * x_i * x_j for i < j)` over integer vectors.
- Add `Inertia` and exact inertia computation using rational congruence reduction with 1×1 and 2×2 pivots, without floating-point eigenvalue approximation.
- Export the new types from `dynkinlab` and document the API and a quickstart example.
- Add unit and property tests covering Dynkin Cartan matrices, affine/degenerate forms, indefinite forms, and validation failures.
- Keep the existing Hatch check/test, coverage, Colab, and packaging gates unchanged.

Out of scope for this change: `Bigraph`, flation, Dynkin classification, non-symmetric symmetrizable matrices, and the flation-versus-inertia priority beyond this symmetric case.

## Capabilities

### New Capabilities

- `quasi-cartan-inertia`: immutable symmetric quasi-Cartan matrices, integer quadratic forms, and exact inertia.

### Modified Capabilities

- None.

## Impact

- New modules under `src/dynkinlab/` and public exports in `src/dynkinlab/__init__.py`.
- New tests under `tests/`.
- Documentation updates in `README.md`, `docs/getting-started/quickstart.md`, and `docs/reference/index.md`.
- No new runtime dependencies; exact arithmetic uses the standard library `fractions` module.
- No changes to package metadata, dependency floors, CI, or the Colab gate.
