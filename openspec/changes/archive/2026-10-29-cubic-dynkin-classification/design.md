# Design

## Context

See `proposal.md` for motivation. The current package exposes an immutable `QuasiCartanMatrix` that validates symmetry, exact `QuadraticForm`, exact `Inertia`, and Colab/Markdown rendering. There is no bigraph, no flation, and no Dynkin type. `networkx>=3.7` and `matplotlib>=3.10` are already runtime dependencies, but no module uses them yet.

The mathematical source is the thesis and Pérez, Abarca & Rivera, *Cubic Algorithm to Compute the Dynkin Type of a Positive Definite Quasi-Cartan Matrix*, Fundamenta Informaticae 158 (2018) 369–384. The paper supplies the admissible-class filter, the three-stage O(n^3) classifier, the `l(n)` bound, the non-symmetric preprocessing, and the final graph classification. The thesis supplies the bigraph/flation viewpoint and the structural A/D results that are deliberately deferred here.

House constraints that shape the design: frozen dataclasses; strict typing; numpydoc; SPDX headers; no module-level functions; no imports of underscore-prefixed private names (the `_repr_*` hooks are the exception); the existing Hatch, coverage, packaging, and Colab gates stay unchanged.

## Goals / Non-Goals

**Goals:**

- Broaden `QuasiCartanMatrix` to all symmetrizable quasi-Cartan matrices while preserving the behavior of existing symmetric inputs.
- Make a label-generic mixed bigraph a view over `QuasiCartanMatrix`, so there is one algebraic source of truth.
- Keep a strict symmetric factory for callers that want the old symmetry guarantee.
- Implement the paper's classifier faithfully, including the Stage 1/2 optimizations, and return a first-class certificate.
- Return the full finite Dynkin list, with a documented G2 extension beyond the paper's admissible class.
- Keep the classifier exact and bounded; never loop indefinitely.
- Make the result readable in Colab through the existing display pattern.

**Non-Goals:**

- A-block tree, D-core, and affine `A~` structural decompositions.
- Affine/Euclidean classification.
- Quivers, directed multigraphs beyond the paper's mixed bigraphs, or non-symmetrizable matrices.
- A text DSL/CLI, DOT/TikZ export, or animation of witnesses.
- Renaming `QuasiCartanMatrix` or introducing a second public symmetrizable matrix type.
- Changing the form, inertia, display, equality, hashing, or repr behavior of symmetric inputs.

## Decisions

### One general `QuasiCartanMatrix` plus a strict symmetric factory

Keep the existing `QuasiCartanMatrix` name and broaden it to the general quasi-Cartan matrix: integer entries, diagonal 2, and a positive diagonal integer symmetrizer `D` such that `D A` is symmetric. Symmetric matrices are the special case `D = (1, ..., 1)`.

The class stores `A` and the canonical `D`; it exposes:

```text
dimension
entries             (A)
symmetrizer         (D)
symmetrized_entries (S = D A)
is_symmetric
inertia
flation(source, target)
```

The constructor first checks whether `A` is symmetric. If it is, the canonical symmetrizer is all ones and no graph traversal is needed; existing symmetric inputs keep exactly their current form, inertia, display, equality, hashing, and repr behavior. If it is not symmetric, the constructor computes the canonical minimal-trace symmetrizer described below and rejects the matrix if none exists. `QuasiCartanMatrix.symmetric(entries)` is a strict factory that rejects non-symmetric input, preserving the old validation contract for callers that want it.

Alternative considered: keep the symmetric class and add a sibling named `SymmetrizableQuasiCartanMatrix`. Rejected because the long name is a symptom of letting the special case own the general name; the literature calls the symmetrizable object a quasi-Cartan matrix. Shorter sibling names such as `SymmetrizableMatrix` lose the quasi-Cartan meaning, and `MixedQuasiCartanMatrix` describes the bigraph rather than the matrix.

Alternative considered: add a public `QuasiCartanLike` protocol. Rejected because there is now one concrete matrix type; the Bigraph and classifier can accept `QuasiCartanMatrix` directly. The existing `MatrixLike` protocol remains for the form/inertia adapter.

### Canonical minimal-trace symmetrizer

The paper defines the symmetrizer as the unique positive diagonal integer `D` of minimal trace for which `D A` is symmetric. For a connected matrix it is unique up to positive scalar; for a disconnected matrix each component is normalized independently.

Implementation:

1. Validate shape, integer entries, diagonal 2.
2. Traverse each connected component of the non-zero off-diagonal graph.
3. Choose the smallest-index vertex in the component as root and set its value to 1.
4. Propagate rational ratios `d_j = d_i * a_ij / a_ji` with `Fraction`; reject a zero/non-zero mismatch, a sign mismatch, or an inconsistent cycle.
5. Scale the component to positive integers by the least common multiple of denominators, divide by the gcd, and keep the root normalized.
6. Reject if the resulting `D A` is not symmetric or not integral.

A symmetric input short-circuits to `D = (1, ..., 1)` and skips the component traversal; this preserves the existing construction cost and behavior for the common symmetric path.

Alternative considered: require the caller to supply `D`. Rejected because the value would not be canonical, equality/hash would depend on scaling, and the Bigraph/classifier would need to renormalize anyway.

### Form and inertia use `S = D A`

The form is

```text
q_A(x) = 1/2 x^T S x
       = sum_i d_i x_i^2 + sum_{i<j} d_i a_ij x_i x_j
```

The existing `QuadraticForm` is generalized from `sum x_i^2` to `sum (S_ii / 2) x_i^2`; for symmetric input `S_ii = 2`, so behavior is unchanged. `QuasiCartanMatrix.quadratic_form` passes an immutable `SymmetrizedMatrix` adapter with `entries = S`; the existing `Inertia` reduction consumes the same adapter. This keeps the exact form/inertia algorithms and avoids duplicating them.

Alternative considered: teach `Inertia` to symmetrize internally. Rejected because `Inertia.from_matrix` already accepts a structural matrix, and an explicit adapter keeps the algebra clear and testable.

### Flation is D-independent

The paper's entry formula is independent of `D`:

```text
A'_ij = A_ij                         if i != r and j != r
A'_rj = A_rj - A_rs * A_sj          if i = r and j != r
A'_ir = A_ir - A_sr * A_is          if i != r and j = r
A'_rr = A_rr
```

For `s = r`, the flation is the sign inversion `D_r A D_r` with `D_r` the diagonal matrix with `-1` at `r` and `1` elsewhere. The operation returns a new value; the implementation updates from a copied row/column using the original entries, never sequentially reusing a mutated entry. This is the trap called out in the thesis appendix.

`D` is carried or canonicalized only when constructing the result. The classifier's inner loop updates `A` in a mutable local state; it does not reconstruct a frozen matrix (and therefore does not recompute `D`) per flation.

Alternative considered: implement the graph-local flation algorithm directly. Rejected as the source of truth: the matrix formula is easier to verify against the paper's equation (1), and the bigraph is a view. The graph interpretation is still tested through the edge/arrow view.

### Bigraph is a label-generic view, not a second store

`Bigraph` holds a `QuasiCartanMatrix` and a tuple of unique hashable labels, defaulting to `0..n-1`. It stores no independent adjacency. Edges are derived on access from `A`:

```text
A_ij = A_ji          -> undirected edge, signed weight A_ij
|A_ij| < |A_ji|      -> arrow i -> j, signed weight
                        sgn(A_ij) * max(|A_ij|, |A_ji|)
```

The style is solid for a negative weight and dotted for a positive weight; multiplicity is the absolute weight. The paper prints the weight magnitude; the style bullets use the sign. Figure 1 confirms the intended signed weight, so the library uses the signed value consistently and records the convention in the docs.

This view is lossless for the positive definite/admissible case: the larger magnitude is stored, the smaller magnitude is 1, and the common sign is unambiguous. For a general symmetrizable matrix whose smaller magnitude is greater than 1, the view follows the paper's mixed-bigraph projection and the matrix remains the exact source of truth. The docs state this limitation.

Alternative considered: an independent graph value with an edge list. Rejected because it duplicates validation and invites drift between `A`, `B(A)`, and `D`. The signed matrix is canonical.

`Bigraph` delegates `flation` to the matrix by mapping labels to indices, and delegates `classify` to the classifier. Both preserve the labels. Unknown labels raise `ValueError`.

### First-class flation witnesses

`FlationStep(source, target)` identifies labels. `FlationSequence(steps)` is immutable and supports:

```text
apply(bigraph)          replay on a bigraph
inverse()               reverse the step order, because T is an involution
verify(bigraph, final)  replay and compare
```

A step with equal source and target is the sign inversion used by Stage 1. A step with distinct labels is an inflation/deflation. The classifier returns the full sequence, including preprocessing sign inversions and Stage 2 inflations. The sequence is the first-class certificate, not a debug string.

Alternative considered: return a list of index pairs. Rejected because the Bigraph is label-generic and the result is rendered/verified by users; a value object is the right boundary.

### Classifier architecture

`DynkinClassifier` operates on a `QuasiCartanMatrix` and returns a `DynkinClassification`. It first splits the matrix into connected components of the non-zero off-diagonal graph and classifies each component independently. The paper assumes connected input; component splitting is the library's extension from the thesis's earlier algorithm.

For each component:

**G2 pre-check.** If the component is 2-by-2 and matches the `G2` Cartan matrix pattern (entries `-1` and `-3` with the correct diagonal), classify it as `G2` immediately with an empty sign/flation witness. This is an O(1) extension: the paper's `|A_ij| < 3` admissible class and Stage 3 omit G2 even though Figure 2 includes it. Both orientations are accepted.

**Stage 1: filter and preprocess.**

- Check admissibility: diagonal 2; off-diagonal magnitudes below 3 (G2 handled first); no undirected pair with magnitude greater than 1; global symmetrizability is already guaranteed by the matrix type; local sign/ratio conditions are checked where the matrix type is bypassed by a component view.
- Determine symmetry. If non-symmetric and `n != 4`, find a distinguished vertex: a source or sink with all arrows incident on it. If none, return `NOT_POSITIVE_DEFINITE` with reason `no distinguished vertex` (Lemma 4.1; F4 is the only positive definite exception and it is n = 4).
- Build the dotted-edge index `g`: `g_i = 1` if at least one dotted edge is incident to vertex `i`. This supports O(n) dotted-edge lookup and O(n) maintenance after a flation.
- Make the bigraph solid-arched. For each dotted arrow incident on the distinguished vertex, apply the sign inversion at its non-distinguished endpoint (`T_{u,u}`). Record each inversion in the witness. The paper's Stage 1 guarantees the result is solid-arched; the reference oracle checks this invariant.

**Stage 2: bounded inflations.**

- Symmetric: while a dotted edge exists, choose one via `g` and apply the flation. The number of iterations is bounded by

```text
l(n) = n(n+1)/2                if n < 4
       n(15n - 168) + 504      if 6 <= n <= 8
       n(n-1)                  otherwise
```

- Non-symmetric with `n > 4`: after Stage 1, remove the distinguished vertex and run the symmetric inflation procedure on `G - x` with bound `l(n-1)`. Apply the resulting flations to the full matrix. The distinguished vertex and its solid arrows remain untouched; the paper's Stage 1/2 construction is what makes the non-symmetric case O(n^3) rather than the naive O(n^6).
- Non-symmetric with `n <= 4`: run the inflation loop with the paper's bound of 24 iterations; if it has not finished, return `NOT_POSITIVE_DEFINITE`.
- For a non-symmetric dotted edge, choose the flation direction that preserves the solid-arched invariant. The implementation follows Lemma 4.2's case split; the test oracle also tries both directions and verifies the invariant.
- If the bound is exhausted, return `NOT_POSITIVE_DEFINITE` with reason `bound exhausted`, the partial witness, and the current state.
- After every flation, update `A` in O(n) using the entry formula and update the `g` flags by scanning the source and target rows/columns in O(n).

**Stage 3: classify the normal form.**

The final graph is a mixed bigraph with no dotted edges or arrows (all arrows are solid). Traverse it:

1. If the graph has a cycle, a parallel edge, a vertex of degree greater than 3, or more than one arrow, it is not a Dynkin diagram.
2. If exactly one vertex has degree 3: no arrow is allowed; BFS distances from that vertex decide `D_n` (at most one vertex at distance 2) or `E_n` for `6 <= n <= 8` (exactly two at distance 2 and at most one at distance 3).
3. Otherwise the graph is a path: no arrow means `A_n`; one arrow `u -> v` means `B_n` if `v` is a leaf, `C_n` if `u` is a leaf, and `F_4` if `n = 4` and both are internal; any other position is not a Dynkin diagram.
4. G2 was already handled before Stage 1.

If Stage 3 does not match a Dynkin diagram, return `NOT_POSITIVE_DEFINITE` with reason `normal form is not a Dynkin diagram` and retain the normal form as a diagnostic.

The classifier is O(n^3) for admissible input: Stage 1 is O(n^2), Stage 2 is O(n^3), Stage 3 is O(n^2), and witness recording is O(n^2) after the paper's preprocessing. The G2 pre-check is O(1).

### Classification result

`DynkinClassification` is immutable and exposes:

```text
status              CLASSIFIED | NOT_ADMISSIBLE | NOT_POSITIVE_DEFINITE
reason              human-readable diagnostic
inertia             exact inertia of the symmetrized matrix
components          tuple of (labels, DynkinType) for each component
witness             FlationSequence, possibly partial
normal_form         Bigraph | None
steps               number of flation/sign-inversion steps
bound               applied paper bound, or the n <= 4 bound
input               retained Bigraph used by verification
```

`CLASSIFIED` means a finite Dynkin type was found. `NOT_ADMISSIBLE` means the paper's filter rejected the matrix (except G2). `NOT_POSITIVE_DEFINITE` covers no distinguished vertex, bound exhaustion, and a non-Dynkin normal form; the `reason` distinguishes them. `verify()` replays the witness on the retained input and compares with the normal form.

Alternative considered: a separate `NORMAL_FORM_NOT_FINITE` status. Rejected because the specs distinguish admissibility from positivity and a diagnostic reason is enough; it keeps the status enum small and stable.

The result retains the input Bigraph (or a default-label view for a matrix input), which makes `verify()` a true certificate check without asking the caller to reconstruct the input. The retained value is a reference to an immutable matrix/bigraph, so the memory cost is a handle, not a copy of every intermediate state.

### Components, labels, and determinism

The classifier splits components by matrix index order before constructing the result. Components are returned in ascending order of their smallest matrix index; duplicate types are preserved. At the matrix level labels are integer index tuples; at the Bigraph level the classifier maps component indices to the original labels. The witness uses the same labels.

### Display architecture

The display layer follows the existing pattern: value objects own their `to_markdown`/`to_html`/`_repr_*` methods; display options resolve explicit options over `dynkinlab.display.options`.

- `QuasiCartanMatrix` uses the same numeric table renderer for symmetric and non-symmetric inputs, showing `A` and the configured index base.
- `Bigraph.to_markdown` renders a compact edge/arrow table, with labels, direction, style, and multiplicity.
- `Bigraph.to_html`/`_repr_html_` render a self-contained inline SVG. `networkx` computes a deterministic layout with the configured seed; `matplotlib` draws solid/dotted lines, arrows, and multiplicity labels into an SVG buffer. No IPython dependency is used.
- Parallel edges are collapsed into one line with a multiplicity label, matching the paper/prototype convention. The `bigraph_show_multiplicity` option controls the label.
- `DynkinType.to_markdown` returns the canonical name; `DynkinClassification.to_markdown` includes status, inertia, component table, step count, bound, and a normal-form summary.
- New `DisplayOptions` fields are validated on assignment and reset: `bigraph_layout_seed` (non-negative integer, default 0), `bigraph_figure_size` (positive pair, default `(4.0, 4.0)`), `bigraph_show_labels` (bool, default true), `bigraph_show_multiplicity` (bool, default true), and `bigraph_arrow_style` (`"arrow"` or `"line"`, default `"arrow"`).

Alternative considered: return a matplotlib `Figure` or use `IPython.display`. Rejected because the package must remain importable without IPython and the string-returning hooks are the established contract.

### Module layout

```text
src/dynkinlab/matrix.py              MatrixLike and SymmetrizedMatrix adapter
src/dynkinlab/quasi_cartan.py        general matrix, symmetric factory, flation
src/dynkinlab/symmetrizer.py         canonical minimal-trace symmetrizer helper
src/dynkinlab/quadratic_form.py      generalized diagonal coefficient, form display
src/dynkinlab/inertia.py             exact inertia (unchanged algorithm)
src/dynkinlab/bigraph.py             Bigraph, BigraphEdge, EdgeStyle
src/dynkinlab/flation.py             FlationStep, FlationSequence
src/dynkinlab/dynkin.py              DynkinType, DynkinClassification, status
src/dynkinlab/classifier.py          DynkinClassifier and Stage 1/2/3 helpers
src/dynkinlab/display.py             DisplayOptions and shared rendering helpers
src/dynkinlab/bigraph_display.py     SVG renderer used by Bigraph display methods
```

`Bigraph.classify` uses a deferred import of `DynkinClassifier` to avoid the `bigraph -> classifier -> bigraph` module cycle. `FlationSequence` accepts `Bigraph` or `QuasiCartanMatrix` values and uses deferred imports for replay. All new public names are exported through `__init__.py` and `__all__`; the source/test import check must continue to pass.

## Risks / Trade-offs

- **The paper's Stage 1/2 pseudocode has ambiguous details** -> The design pins the interpretation (distinguished vertex, sign inversion at the non-distinguished endpoint, Lemma 4.2 direction choice, `G - x` reduction, n <= 4 bound 24). A test-only naive inflations oracle property-tests the optimized classifier on small random matrices; the paper's Figure 1, Example 3.1, and the thesis examples are golden tests.
- **Broadening `QuasiCartanMatrix` changes its validation contract** -> Existing symmetric inputs are unchanged and stay on the fast path; `symmetric()` preserves strict rejection; tests cover non-symmetric acceptance and non-symmetrizable rejection; docs state that the general constructor is symmetrizable, not symmetric-only.
- **G2 is outside the paper's admissible class** -> Treat it as a documented O(1) extension; test both orientations; the rest of the algorithm and the O(n^3) bound are unchanged. The classifier's status and docs state the extension.
- **The printed bigraph weight is unsigned while the style test uses the sign** -> Use the signed max magnitude, matching Figure 1; document the convention and test arrows whose raw entries are both negative.
- **The canonical symmetrizer could leak into the flation hot loop** -> The classifier updates a mutable `A` state and carries the symmetrizer; it constructs a canonical matrix/result only at the boundaries. Public flation may canonicalize on each call, but it is not the classifier's inner loop.
- **The Bigraph view is not lossless for general non-positive-definite symmetrizable matrices** -> The matrix remains the exact source; the view follows the paper's projection and the docs state the limitation. Positive definite/admissible inputs are complete because one magnitude is 1.
- **Exact inertia uses rational arithmetic and may have larger constants than the cubic integer classifier** -> The core classification remains O(n^3); inertia is exact and is computed once at the result boundary. No floating-point eigenvalues are used.
- **Full A-G canonical diagrams have arrow-direction conventions (B/C, F4, G2)** -> Build canonical matrices explicitly, test both orientations, and make Stage 3 and canonical display share the same convention.
- **SVG layout is not deterministic across environments** -> Set a layout seed, fix the figure size, and assert on labels/styles/arrow markers rather than pixel geometry. Use the `Agg` backend and capture to a buffer; never call `pyplot.show`.
- **Label-generic values accept arbitrary hashables** -> Validate uniqueness and hashability; raise `ValueError` for duplicate, unhashable, or wrong-length labels. Component order is based on matrix index, not label ordering, so it stays deterministic.
- **Existing display options are mutable global state** -> Extend validation/reset and use isolated `DisplayOptions` instances in tests; the canonical object is reset where needed.

## Migration Plan

This is a source-compatible broadening for valid symmetric inputs. Existing `QuasiCartanMatrix` form, inertia, display, equality, hashing, and repr remain unchanged; the constructor now also accepts non-symmetric symmetrizable matrices, and `symmetric()` preserves the strict symmetric validation. The class gains symmetrizer, symmetrized entries, and flation. New types and exports are added; no data migration is involved.

Rollback: remove the new modules, new `__all__` entries, and the new display requirements; the symmetric path remains functional. Update README, quickstart, reference docs, and CHANGELOG in the same change. The package version is not bumped here; the unreleased changelog records the new surface. The Colab smoke gate continues to check that the package imports under the pinned runtime; no new dependency is added.

## Open Questions

None that affect the specs, the approach, or the task breakdown. DOT/TikZ export, witness animation, A-block/D-core decomposition, and affine classification are explicitly deferred.
