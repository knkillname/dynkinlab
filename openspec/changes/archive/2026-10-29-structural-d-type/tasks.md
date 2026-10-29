# Tasks

## 1. Edge multiset foundation

- [x] 1.1 Add `src/dynkinlab/edge_multiset.py` with an immutable `EdgeMultiset` that validates labels
  and undirected `BigraphEdge` values, normalizes same-style parallels, and exposes
  `from_bigraph`. Verify `hatch test tests/test_edge_multiset.py --no-cov` passes with tests for
  opposite-style preservation and same-style aggregation.
- [x] 1.2 Add `EdgeMultiset.__add__`, `identify`, and `simplify`; verify focused tests cover
  cancellation of equal opposite pairs, excess-style multiplicity, isolated labels, and rejection of
  directed edges.
- [x] 1.3 Document `EdgeMultiset` in `docs/reference/index.md`; verify `hatch run docs:build` passes.

## 2. Bigraph sum and vertex identification

- [x] 2.1 Add `Bigraph.sum` as a class method with first-appearance and explicit label order; verify
  focused tests cover disjoint union, shared-label gluing, opposite-edge cancellation, incomplete
  explicit order, and rejection of mixed pieces.
- [x] 2.2 Add `Bigraph.identify`; verify focused tests cover non-adjacent merges, adjacent-loop
  removal, opposite-edge cancellation, retained label order, and invalid labels.
- [x] 2.3 Document `Bigraph.sum` and `Bigraph.identify` in `docs/reference/index.md`; verify
  `hatch run docs:build` passes.

## 3. D-core value

- [x] 3.1 Add `src/dynkinlab/structure_d.py` with `DCore`, `DCore.cycle(labels, styles)`, validation
  for ordinary cycles and the two-edge double link, and `cycle_condition()`. Verify focused tests
  cover an odd five-cycle, a one-solid/one-dotted double link, even parity, a chord, duplicate
  labels, and a missing cycle edge.
- [x] 3.2 Document `DCore` in `docs/reference/index.md`; verify `hatch run docs:build` passes.

## 4. D-core gluing certificate

- [x] 4.1 Implement `DCore.gluing(pieces, labels=None)` and `DCoreGluingCertificate`, including
  connected positive-definite A-type pieces, core-edge style and multiplicity, internal endpoints,
  exact consecutive overlaps, explicit label order, and the final D_n type. Verify the D13 and D4
  constructions produce certificates that verify and classify as the expected types.
- [x] 4.2 Add focused tests for non-A pieces, non-positive-definite pieces, mismatched core styles,
  articulation endpoints, unexpected overlaps, wrong piece counts, and tampered core/piece/gluing
  values; verify `hatch test tests/test_structure_d.py --no-cov` passes.
- [x] 4.3 Document the D-core gluing API in `docs/reference/index.md`; verify
  `hatch run docs:build` passes.

## 5. A-cyclization certificate

- [x] 5.1 Implement `Bigraph.a_cyclization(source, target)` and `ACyclizationCertificate` with
  connected positive-definite A-type input, internal endpoints, a unique shortest path of length at
  least two, odd dotted parity, and the identified D_{n-1} result. Verify the A9 1-to-6
  construction produces D8 and its certificate verifies.
- [x] 5.2 Add focused tests for adjacent endpoints, even dotted parity, articulation endpoints,
  non-A sources, disconnected sources, too few vertices, and tampered result/path values; verify
  `hatch test tests/test_structure_d.py --no-cov` passes.
- [x] 5.3 Document A-cyclization in `docs/reference/index.md`; verify `hatch run docs:build` passes.

## 6. Public surface and integration

- [x] 6.1 Export `EdgeMultiset`, `DCore`, `DCoreGluingCertificate`, and `ACyclizationCertificate`
  from `dynkinlab`, update `__all__`, and update the exact surface assertion in
  `tests/test_package.py`; verify `hatch test tests/test_package.py --no-cov` passes.
- [x] 6.2 Add the changelog entry and update the API reference's current/planned surface sections;
  verify `hatch run docs:build` passes.
- [x] 6.3 Run `hatch check`, `hatch test`, `hatch run docs:build`, and `hatch run verify`; verify all
  pass without coverage, lint, type, or documentation warnings.
- [x] 6.4 Run `openspec validate structural-d-type --strict`; verify the change and its delta spec
  are valid.

## Workflow follow-up

- Archive the change with `/opsx-archive` after review and verification.
- Create a Git commit after archiving that names `structural-d-type`.
