# Tasks

## 1. NetworkX conversion

- [x] 1.1 Add `Bigraph.to_networkx()` returning an undirected `nx.Graph` with every label and edge attributes `style`, `directed`, `multiplicity`, and `weight`; verify `hatch test tests/test_structure.py --no-cov` passes the conversion tests.
- [x] 1.2 Add conversion tests for a labeled mixed bigraph and an isolated vertex; verify every label is a node and each edge exposes the four attributes.
- [x] 1.3 Add the `Block` and `BlockCutTree` values with `labels`, `blocks`, `articulation_points`, `to_networkx()`, and `verify()`; verify the A9 structure test finds blocks `{1,2,3}`, `{3,4,5,9}`, `{4,7}`, `{5,6,8}` and articulation labels `3,4,5`.
- [x] 1.4 Add `BlockCutForest` and `Bigraph.block_cut_forest()`; verify tests cover a bridge, a single vertex, a single edge, and a disconnected A2+A3 forest.
- [x] 1.5 Document the NetworkX conversion and block-cut forest in `docs/reference/index.md`; verify `hatch run docs:build` passes.

## 2. A-block partitions

- [x] 2.1 Add `ABlockPartition` and `Bigraph.is_a_block(labels)` using solid-graph bipartition; verify a standard `F({a,b},{c,d})` block returns a partition that verifies.
- [x] 2.2 Add tests for an all-dotted block (Y empty), a single solid edge, a solid triangle rejection, and a disconnected solid graph rejection; verify `hatch test tests/test_structure.py --no-cov` passes.
- [x] 2.3 Document the A-block rule and the all-dotted special case in `docs/reference/index.md`; verify the strict docs build succeeds.

## 3. A-type certificates

- [x] 3.1 Add `ATypeComponentCertificate` and `ATypeCertificate`, plus `Bigraph.is_type_a()` and `Bigraph.a_type_certificate()`; verify A9 returns a certificate whose `verify()` is true.
- [x] 3.2 Add tests for D13 rejection, A1, A2, a disjoint A2+A3 union, and a certificate checked against a different bigraph; verify all cases pass.
- [x] 3.3 Add a compact reference example that displays the A9 certificate and its block-cut forest; verify the example runs and the docs build.
- [x] 3.4 Update `CHANGELOG.md` under `[Unreleased]` with the new structural values and methods; verify the changelog entry is present.

## 4. Public surface and integration

- [x] 4.1 Export `ABlockPartition`, `ATypeCertificate`, `ATypeComponentCertificate`, `Block`, `BlockCutForest`, and `BlockCutTree` from `dynkinlab` and add them to `__all__`; verify `hatch test tests/test_package.py --no-cov` passes.
- [x] 4.2 Add an integration test that the A9 certificate is A-type and the D13 certificate is absent while `D13.classify()` still reports `D_13`; verify the focused classifier and structure tests pass.
- [x] 4.3 Run `hatch check` and confirm Ruff and strict Pyrefly pass across `src`, `tests`, and `tools`.

## 5. Full verification

- [x] 5.1 Run `hatch test` and confirm the full suite passes with the 80% coverage gate.
- [x] 5.2 Run `hatch run docs:build` and confirm the strict Sphinx build is warning-free.
- [x] 5.3 Run `hatch run verify` and confirm `check`, `test`, and the Colab smoke assertions pass.

## Workflow follow-up

- Archive the change with `openspec archive structural-a-type` after review.
- Create a Git commit named after the archived change, per the project archive guidance.
