# Proposal

## Why

When a matrix has no positive diagonal symmetrizer, `QuasiCartanMatrix` currently raises a generic
`ValueError("matrix is not symmetrizable")`. The caller receives no evidence of why symmetrization
failed, cannot distinguish an input-validation failure from a mathematical obstruction, and cannot
independently verify the negative result. The symmetrizer search already discovers the exact local
obstruction, so that information should be preserved instead of discarded.

## What Changes

- Add `SymmetrizerCertificate`, an immutable, verifiable witness for one of three obstruction kinds:
  a zero entry on one side of an edge, an opposite-sign edge, or an inconsistent ratio cycle.
- Add `SymmetrizerObstruction`, the enumeration naming those three kinds.
- Add `NotSymmetrizableError(ValueError)` carrying the certificate in a public `certificate`
  attribute. Its message remains human-readable and keeps the word `symmetrizable` for
  compatibility with existing callers.
- Extend the canonical symmetrizer search to build and raise that exception at the point where an
  obstruction is found. Existing input-validation failures continue to raise plain `ValueError`.
- Export `NotSymmetrizableError`, `SymmetrizerCertificate`, and `SymmetrizerObstruction` from the
  top-level `dynkinlab` package and list them in `dynkinlab.__all__`.
- Add very concise proof-of-correctness comments beside the obstruction checks: each local witness
  contradicts the positivity equations, and consistency of every ratio on a spanning tree is
  sufficient for existence.
- Add tests for all three certificates, certificate verification, exception compatibility, and the
  existing canonical symmetrizer behavior.
- Update package documentation and the changelog for the new public exception surface.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `quasi-cartan-inertia`: add the non-symmetrizability certificate and exception behavior, and
  extend the public-surface requirement with the new names.

## Impact

- `src/dynkinlab/symmetrizer.py`: certificate value, exception type, obstruction detection, and
  proof comments.
- `src/dynkinlab/quasi_cartan.py`: constructor continues to propagate the new `ValueError` subclass;
  docstring wording is updated.
- `src/dynkinlab/__init__.py`: export the exception and certificate types.
- `tests/test_quasi_cartan_general.py` (or a focused symmetrizer test module): obstruction and
  compatibility coverage.
- `docs/reference/index.md`, `README.md` where construction failures are described, and
  `CHANGELOG.md`.
- No new dependencies and no changes to flation, inertia, or the Dynkin classifier.
- Compatibility: non-symmetrizable construction remains catchable as `ValueError`; the new
  exception adds structured information without removing the existing message contract.
