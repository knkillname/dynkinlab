# Design

## Context

See `proposal.md` - Why. The current `Bigraph` exposes labels, edges, flation, and classification, but no structural decomposition. The 9-vertex A9 and 13-vertex D13 matrices already exist as golden classifier tests, so they provide fixed targets for the structural layer. NetworkX is already a runtime dependency, and the package's public surface is exactly `dynkinlab.__all__`.

## Goals / Non-Goals

**Goals:**

- Add a stable structural API over `Bigraph` without changing classification, flation, or display behavior.
- Make the A-type theorem inspectable and verifiable, including disconnected bigraphs.
- Keep the values immutable, typed, and consistent with the existing certificate style.
- Make the block-cut structure usable for NetworkX figures without exposing raw NetworkX graphs as the primary value.

**Non-Goals:**

- D-core construction, edge multisets, A-cyclization, or D-type recognition.
- A structural classifier or changes to `DynkinClassifier`.
- Affine Dynkin types or the thesis notebook.
- Rendering changes to the existing bigraph SVG path.

## Decisions

1. **New `structure.py` module with lazy imports from `Bigraph`.** The structural types live in `src/dynkinlab/structure.py`. `Bigraph` methods import the builders inside the method body, following the existing `Bigraph.classify()` pattern, so `structure.py` can import `Bigraph` at module scope without a cycle. Public names are re-exported from `dynkinlab.__init__`.

2. **`to_networkx()` returns an undirected graph with edge attributes.** The graph contains every bigraph label, including isolated vertices. Each edge carries `style` (`solid`/`dotted`), `directed`, `multiplicity`, and signed `weight`. Direction is an attribute rather than a separate `DiGraph` because block-cut analysis and the existing renderer both use the underlying undirected graph. A caller who needs a directed graph can filter `directed=True`.

3. **Block-cut structure is a value object, not a raw NetworkX graph.** `BlockCutForest` contains one `BlockCutTree` per connected component; each tree exposes `labels`, `blocks`, and `articulation_points`. `to_networkx()` is an export for drawing and analysis. This keeps the API stable and lets `verify()` recompute the structure from the bigraph.

4. **Blocks are values with their induced bigraph.** `Block` stores its labels and the induced `Bigraph`, so an A-block partition can be checked without re-indexing the parent. `Block` is frozen and hashable; `BlockCutTree.to_networkx()` uses block values as block nodes and labels as vertex nodes, with a `kind` attribute. If a label collides with a `Block` value, the conversion raises `ValueError` rather than silently merging nodes.

5. **A-block partition uses solid-graph bipartition.** For a candidate block, every pair of vertices must be adjacent. If there are no solid edges, the block is accepted with `X` equal to all vertices and `Y` empty. Otherwise the solid-edge graph must be connected and bipartite; a proper 2-coloring gives `X` and `Y`, and every cross pair must be solid and every within-side pair dotted. A disconnected solid graph with edges is rejected. This is polynomial and handles the all-dotted, single-edge, and standard `F(X,Y)` cases.

6. **Certificates are per component, with a collection value for the whole bigraph.** `ATypeComponentCertificate` stores the component labels, `DynkinType("A", n)`, its `BlockCutTree`, and its partitions. `ATypeCertificate` stores the component certificates. `Bigraph.a_type_certificate()` returns a certificate only when every connected component is A-type; `Bigraph.is_type_a()` is the boolean convenience. A1 is special-cased because it has no blocks. A2 is one block with no articulation points.

7. **Verification recomputes the structure.** `BlockCutForest.verify`, `ABlockPartition.verify`, and `ATypeCertificate.verify` recompute the relevant structure from the supplied `Bigraph` and compare it with the stored value. This is intentionally redundant: a certificate must fail when checked against a different or tampered bigraph, matching the existing `SymmetrizerCertificate` and `DynkinClassification.verify` style.

8. **No classifier changes.** The A-type predicate is a separate structural operation. The notebook's structural algorithm can use it, but `DynkinClassifier` keeps its current bounded-flation behavior. This avoids changing golden results or the exact classifier contract.

## Risks / Trade-offs

- **NetworkX block conventions for bridges and isolated vertices** → Add explicit tests for a bridge, a single vertex, a single edge, and a disconnected forest; normalize the representation in `structure.py`.
- **Solid-graph bipartition edge cases** → Test all-dotted blocks, a solid triangle, a single solid edge, and a disconnected solid graph; reject rather than guess.
- **Block-node collisions in `to_networkx()`** → Detect a label equal to a `Block` value and raise a clear `ValueError`; document that block nodes carry `kind="block"`.
- **Certificate verification cost** → Recomputing is O(n+m) with NetworkX for typical bigraphs and keeps the certificate honest; the 9- and 13-vertex examples are small.
- **Public-surface test churn** → Update `tests/test_package.py` and `__all__` in the same change; add one import scenario to the structural tests.
- **Strict typing with NetworkX stubs** → Use concrete `nx.Graph` annotations and keep builder functions annotated; run `hatch check` before the full test suite.

## Migration Plan

Additive only. No persisted data or API removals. Rollback is removing `structure.py`, the new `Bigraph` methods, the exports, and the tests. Documentation and changelog entries are additive.

## Open Questions

None that change the spec or approach. The D-core layer will decide separately whether it reuses `BlockCutForest` for A-type piece checks.
