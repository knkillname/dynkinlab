# Proposal

## Why

DynkinLab can classify the thesis's 9-vertex A-block example, but it cannot expose or verify the structural theorem that explains the result. The planned thesis companion needs first-class A-blocks, block-cut trees, and an A-type certificate so the theorem is inspectable rather than only a classifier outcome. The same A-type check is the structural gate in the thesis's polynomial algorithm.

## What Changes

- Add `Bigraph.to_networkx()`, returning an undirected NetworkX graph with every vertex label and edge attributes for style, direction, multiplicity, and signed weight.
- Add immutable structural values: `Block`, `BlockCutTree`, `BlockCutForest`, `ABlockPartition`, `ATypeComponentCertificate`, and `ATypeCertificate`.
- Add `Bigraph.block_cut_forest()`, `Bigraph.is_a_block(labels)`, `Bigraph.is_type_a()`, and `Bigraph.a_type_certificate()`.
- Verify the A-block partition rule: cross pairs are solid, within-side pairs are dotted, and the two sides are disjoint.
- Verify the A-type theorem on each connected component: every block is an A-block and every articulation vertex belongs to exactly two blocks.
- Export the new public names from `dynkinlab` and list them in `__all__`.
- Add tests for the 9-vertex A9 tree, the 13-vertex D13 non-example, A1/A2, disjoint unions, bridges, isolated vertices, and certificate rejection.
- Update the reference documentation and changelog.

## Capabilities

### New Capabilities

- `bigraph-structure`: structural inspection of bigraphs, including NetworkX conversion, blocks, block-cut forests, A-block partitions, and verifiable A-type certificates.

### Modified Capabilities

None.

## Impact

- New module `src/dynkinlab/structure.py`; lazy imports from `src/dynkinlab/bigraph.py` to avoid an import cycle.
- `src/dynkinlab/__init__.py` and `tests/test_package.py` for the public surface.
- New focused tests plus additions to `tests/test_bigraph.py`.
- `docs/reference/index.md` and `CHANGELOG.md`.
- No classifier, flation, inertia, or quadratic-form behavior changes.
- No new dependencies; NetworkX is already required.
- Additive public API only.
