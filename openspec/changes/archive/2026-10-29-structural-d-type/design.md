# Design

## Context

See `proposal.md` for the motivation. The existing structural layer in `src/dynkinlab/structure.py`
provides block-cut forests, A-block partitions, and A-type certificates. `Bigraph` already derives
edges from a quasi-Cartan matrix, and `EdgeStyle` / `BigraphEdge` provide the signed edge vocabulary.

The D constructions need two operations that the current model lacks:

- adding edges before simplification, because a two-edge D-core keeps a solid and a dotted edge
  between the same pair;
- summing bigraphs that share vertices, because the D-core construction glues A-type pieces along
  core edges.

The exact classifier remains an independent oracle. These certificates verify structural
constructions supplied by the caller; they do not search for a decomposition.

## Goals / Non-Goals

**Goals:**

- Add a reusable edge-multiset value that models addition and cancellation explicitly.
- Add bigraph sum and vertex identification operations with deterministic label order.
- Add a D-core value that handles both the ordinary cycle and the two-edge double link.
- Add verifiable D-core gluing and A-cyclization certificates.
- Keep the public surface aligned with `__all__` and the existing A-type certificate style.

**Non-Goals:**

- Do not change the classifier or add automatic D-type recognition.
- Do not implement general chordless-cycle enumeration for arbitrary bigraphs.
- Do not support directed edges in structural sums, identification, or D certificates.
- Do not add the thesis notebook in this change.

## Decisions

### Separate `EdgeMultiset` from `Bigraph`

`EdgeMultiset` lives in `src/dynkinlab/edge_multiset.py` and stores a label tuple plus a normalized
tuple of undirected `BigraphEdge` values. Same-style parallel edges combine into one edge with a
larger multiplicity. Opposite-style edges remain separate until `simplify()` cancels them.

A signed-weight map was rejected because it destroys the two-edge D-core: a solid and a dotted edge
between the same pair must both exist before simplification. A raw list of edges was rejected
because duplicate normalization and deterministic ordering would leak into every caller.

`simplify()` returns a `Bigraph`: for each pair it subtracts the smaller style multiplicity from the
larger, builds the signed matrix entry, and keeps diagonal entries at 2. Labels without incident
edges remain isolated vertices.

### `Bigraph.sum` uses the same cancellation semantics

`Bigraph.sum(*pieces, labels=None)` is a class method. It computes the union of piece labels in first
appearance order unless the caller supplies an explicit order. It initializes diagonal entries to 2,
adds each piece's signed off-diagonal weights, and constructs a `Bigraph`. Adding weights is
equivalent to edge-multiset addition followed by simplification, without materializing an
intermediate multiset.

The explicit `labels` argument exists because certificate equality is label-order sensitive and the
thesis examples use one-based labels in a specific order. The certificate verifier recomputes with
the stored gluing's label order.

Pieces containing arrows are rejected. The A/D structural constructions are symmetric; allowing
arrows here would create ambiguous cancellation rules and is outside the thesis's A/D theorems.

### `Bigraph.identify` merges labels before simplification

`Bigraph.identify(source, target)` maps every incident edge from `target` to `source`, drops the
loop that arises when an edge already joins the two labels, then simplifies the resulting edge
multiset. The result removes `target` and keeps `source` at its original position. Unknown or equal
labels raise `ValueError`.

This is the primitive behind A-cyclization. It is public because it is a useful graph operation and
because the certificate needs a deterministic result to store and verify.

### `DCore` is a cycle value, not a general bigraph

`DCore` lives in `src/dynkinlab/structure_d.py` and wraps an `EdgeMultiset`. Its labels are in cyclic
order. For a core of length at least three, the edge set must be exactly the consecutive cycle
edges, one per pair, each with multiplicity one. For length two, the edge set must be exactly two
parallel edges, one solid and one dotted; this is the double link that `Bigraph` alone cannot
represent.

`DCore.cycle(labels, styles)` builds a core from a style per cycle position. For a two-label core,
the two styles are the two parallel edges. The class validates the odd dotted count, duplicate
labels, missing edges, extra chords, and multiplicity.

Only the core's odd parity is checked. Because a D-core is itself a chordless cycle, general
chordless-cycle enumeration would add complexity without changing the certificate's meaning.

### D-core gluing certificate

`DCore.gluing(pieces, labels=None)` requires one piece per core edge, in core order. It validates:

- each piece is connected, positive definite, and has an A-type certificate;
- the piece contains its consecutive core edge with the core's style and multiplicity one;
- both core endpoints are internal (not articulation points) in that piece;
- piece intersections are exactly the consecutive core vertices, with the two-piece case sharing
  both core endpoints and no other vertex;
- the sum is connected, has at least four vertices, and is positive definite.

The result is a `DCoreGluingCertificate` with the core, pieces, gluing, and `DynkinType("D", n)`.
`verify()` recomputes `core.gluing(pieces, labels=self.gluing.labels)` and compares the complete
certificate. A tampered core, piece, label order, or gluing returns false rather than raising.

The A-type certificate is structural and does not itself test positivity, so the D certificate adds
an exact inertia check on every piece and on the final gluing. This prevents a structurally A-type
but affine piece from producing a D certificate.

### A-cyclization certificate

`Bigraph.a_cyclization(source, target)` validates:

- the source is connected, has at least five vertices, and is positive definite with an A-type
  certificate;
- the two labels are distinct internal vertices;
- there is exactly one shortest path between them;
- the path has length at least two and an odd number of dotted edges;
- the identified result is connected, has at least four vertices, and is positive definite.

The result is an `ACyclizationCertificate` storing the source, endpoints, path, identified result,
and `DynkinType("D", n - 1)`. `verify()` recomputes the construction and compares the complete
certificate. NetworkX supplies the shortest-path enumeration; the library's `to_networkx` view is
already the structural graph.

### Modules and imports

- `edge_multiset.py` imports `Bigraph` and the edge vocabulary at module scope.
- `structure_d.py` imports `EdgeMultiset`, `Bigraph`, and `DynkinType` at module scope.
- `bigraph.py` imports the edge-multiset and D-structure modules lazily inside `identify` and
  `a_cyclization`, following the existing `classify` / `block_cut_forest` pattern and avoiding an
  import cycle.
- `__init__.py` exports the four new public values; `tests/test_package.py` asserts the exact
  `__all__` list.

### Error handling

Invalid inputs raise `ValueError` with a concrete reason. Certificate `verify()` methods return
false for any tampered or inconsistent state and do not leak construction errors. This mirrors the
A-type certificate's verify behavior.

## Risks / Trade-offs

- **Label-order sensitivity** → `Bigraph.sum` accepts an explicit order, and certificates verify
  against the stored gluing order. Tests cover both default and explicit orders.
- **Two-edge core special case** → `DCore` has dedicated length-two validation and a D4 regression
  test; the same code path also supports ordinary cycles such as D13.
- **Structural A-type does not imply positive definite** → D certificates require exact inertia on
  every A-type piece and on the final result.
- **No general D recognition** → users must supply the core/pieces or the cyclization endpoints;
  the classifier remains the only automatic D path.
- **Repeated verification cost** → certificates recompute the construction for integrity. The
  examples are small, and the cost is acceptable for a verification API.
- **Public surface growth** → four new names require synchronized `__all__`, package tests, API
  docs, and changelog entries.

## Migration Plan

No migration is required. Existing values, methods, and classifier behavior remain unchanged; the
new API is additive. The only compatibility-sensitive check is the exact `__all__` assertion, which
must be updated alongside the exports.
