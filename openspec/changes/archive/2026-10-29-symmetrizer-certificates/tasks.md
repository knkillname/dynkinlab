# Tasks

## 1. Certificate types

- [x] 1.1 Add `SymmetrizerObstruction` with `zero_entry`, `opposite_signs`, and `inconsistent_cycle`, plus a frozen `SymmetrizerCertificate` with `verify(entries)` and `describe(entries)`. Verify with focused unit tests for all three kinds, including rejection of a wrong or tampered matrix.
- [x] 1.2 Add `NotSymmetrizableError(ValueError)` with a public `certificate` attribute, a readable message, and `__reduce__` support. Verify with a pickle round-trip test and an `isinstance(error, ValueError)` assertion.

## 2. Search integration and proof comments

- [x] 2.1 Add parent/depth bookkeeping and a fundamental-cycle helper to the component solver, raising `NotSymmetrizableError` with the matching zero, sign, or cycle certificate. Verify tests cover all three obstructions through `QuasiCartanMatrix` and the existing canonical and disconnected-component tests still pass.
- [x] 2.2 Add the four-line correctness proof comment above the component solver and one-line justification comments at the zero, sign, and cycle branches. Verify by reading the diff that the comments cover all three obstructions, cycle consistency, and primitive normalization.
- [x] 2.3 Preserve `ValueError` compatibility and deterministic failure selection. Verify a test catches the failure as `ValueError`, checks that the message contains `symmetrizable`, and confirms a deterministic first obstruction for a matrix with multiple failing components.
- [x] 2.4 Document the exception and certificate behavior with a short example in `docs/reference/index.md` and the README. Verify `hatch run docs:build` succeeds and the README test still passes.

## 3. Public surface and release notes

- [x] 3.1 Export `NotSymmetrizableError`, `SymmetrizerCertificate`, and `SymmetrizerObstruction` from `dynkinlab`, add them to `__all__`, and update the public-surface expectation in `tests/test_package.py`. Verify the import and `__all__` tests pass.
- [x] 3.2 Update the `QuasiCartanMatrix` constructor docstring and add a `CHANGELOG.md` entry naming the new exception and certificate. Verify the changelog entry is present and the documentation build succeeds.

## 4. Integration verification

- [x] 4.1 Run `hatch check`, `hatch test`, `scripts/verify --skip-colab`, and `hatch run docs:build`. Verify all commands pass and coverage remains at least 80%.
- [x] 4.2 Run `openspec validate --change "symmetrizer-certificates" --strict`. Verify the change reports zero validation issues.
- [x] 4.3 Verify that no file under `src/` or `tests/` contains a `from __future__` import. Run `grep -R "from __future__" src tests` and confirm there are no matches.

## Workflow follow-up

- Archive the change after review requirements are satisfied.
- Verify the archived change and synced main specs.
