# Tasks

## 1. Public result and certificate values

- [x] 1.1 Add `SolidificationStatus`, `NoSolidCertificate`, and `SolidificationResult` in a new `src/dynkinlab/solidification.py`, with frozen dataclasses, validation, `verify()`, `to_markdown()`, and `_repr_markdown_`; verify the new value tests pass.
- [x] 1.2 Add tests for status rendering, sequence verification, certificate verification, and rejection of malformed result/certificate combinations; verify with `hatch test tests/test_solidification.py --no-cov`.

## 2. Exact component search

- [x] 2.1 Implement component splitting, the positive-definite dotted-edge loop, and the bounded BFS for affine and indefinite components using exact Python integers; verify solid, dotted A2, affine, and indefinite-cap tests pass.
- [x] 2.2 Implement frame-isomorphism accounting for small components and the exact bucket/isomorphism fallback for larger components, keeping sign-inverted exact states distinct; verify frame-class counts are consistent and do not exceed exact-state counts.
- [x] 2.3 Add the PSD corank-at-least-two certificate branch and component composition rules; verify the corank-two impossible, disconnected impossible, and disconnected inconclusive tests pass.

## 3. Bigraph integration and public surface

- [x] 3.1 Add `Bigraph.solidify(...)` with cap validation and a lazy import of the new module; verify label-preserving replay and the empty-sequence already-solid case.
- [x] 3.2 Export `SolidificationStatus`, `NoSolidCertificate`, and `SolidificationResult` from `dynkinlab`, update the verbatim `__all__` assertion, and verify `tests/test_package.py::TestPackage::test_public_surface` passes.
- [x] 3.3 Add a test that a relabelled input produces a valid sequence and that repeated identical calls are deterministic; verify with the focused solidification tests.

## 4. Documentation and release notes

- [x] 4.1 Document the three-status decision boundary, caps, certificate scope, and frame accounting in the README and `docs/reference/index.md`; verify every README snippet runs under `hatch test tests/test_readme.py --no-cov`.
- [x] 4.2 Add the user-visible change to `CHANGELOG.md` under `[Unreleased]`; verify the entry names the new public API and the indefinite `INCONCLUSIVE` caveat.
- [x] 4.3 Confirm the dev-container Dockerfile installs the OpenSpec CLI and `post-create.sh` prints its version; verify the Dockerfile is syntactically consistent by inspection and the local `openspec --version` command works.

## 5. Integration verification

- [x] 5.1 Run `hatch check` and fix Ruff or Pyrefly findings in the new module, tests, and docs.
- [x] 5.2 Run `hatch test` and confirm the coverage gate still passes.
- [x] 5.3 Run `hatch run docs:build` and confirm the strict Sphinx build remains warning-free.

## Workflow follow-up

- Archive the change after the project's review requirements are satisfied.
- Create a Git commit after the archive and name the archived change in the commit message.
