# Design

## Context

See `proposal.md` for motivation and the four delta specs for the behavioral contract. The current
notebook is 39 cells with 713 code lines; five cells exceed 40 lines, including a 226-line cell of
A3/A9/D8/D13 coordinate dictionaries and drawing helpers. The package already has the mathematics,
the structural certificates, and the classifier. It also already depends on NumPy, NetworkX, and
matplotlib, so the missing layer is reusable notebook ergonomics, not a new runtime stack.

The archived `thesis-companion-notebook` design deliberately kept the path parser, matrix table,
drawability predicate, and drawing helpers local, and deliberately rejected a public
`Bigraph.from_paths`. This change reverses those four decisions because the notebook is being
replaced, not merely cleaned up. The replacement is a new capability, not another revision of the
thesis-companion contract.

The current library keeps `QuasiCartanMatrix` as the central value and `Bigraph` as a labeled view.
`Bigraph.edges` currently accepts the `2/4` matrix and emits a multiplicity-four arrow, which is the
bigraph of a different matrix. The notebook worked around that with a local predicate. The redesign
makes the view truthful instead of preserving a second, notebook-local definition of drawability.

## Goals / Non-Goals

**Goals:**

- A thin masterclass source whose code cells are declarations, computations, and checks, not a
  shadow library.
- Public methods on existing values for notation, faithful-view detection, layout, drawing, labeled
  matrix display, and certificate summaries.
- NetworkX-owned layouts for every figure; no stored coordinates and no manual position tables.
- A notebook that teaches the material from linear algebra and graph theory upward, without
  thesis-specific framing or algorithm comparisons.
- A fixed canonical tour plus parameter labs that readers can change safely.
- Mechanical enforcement of the 40-line cell cap, no local definitions, and no coordinate tables.
- No new top-level exports, runtime dependencies, or version pinning.

**Non-Goals:**

- Implementing the thesis's polynomial classification algorithm.
- Translating the full thesis or reproducing full proofs.
- Adding a new `LabeledMatrix` value type or changing `QuasiCartanMatrix` equality to include
  labels.
- Adding interactive widgets in the first implementation; plain parameter cells are the primary
  interaction.
- Publishing the notebook through Sphinx or changing the wheel's notebook contents.
- Adding a general Graphviz/DOT parser or per-edge multiplicity syntax.
- Keeping the old `thesis-companion` path as an alias or redirect.

## Decisions

### 1. Replacement and capability retirement

The repository gets `notebooks/dynkin-masterclass.py` and its generated `.ipynb`. The
`thesis-companion` files are deleted. The `thesis-notebook` capability is retired in full, and a new
`dynkin-masterclass` capability owns the notebook contract. The old archive remains historical
context, not an active contract.

This is a clean replacement rather than an in-place rewrite because the filename, title, audience,
and code budget all change. Keeping the old name would preserve the wrong mental model in the first
thing a reader sees. The cost is that old Colab links break; the README and quickstart point to the
new artifact, and the change is recorded in the changelog.

Alternative considered: keep `thesis-companion` as a thin redirect. Rejected because a stub notebook
is another artifact to maintain and still advertises the retired contract.

### 2. Public API shape: methods, not new top-level values

No new class or top-level export is introduced. The new behavior is reachable through existing
public values:

| Value | New behavior |
|---|---|
| `Bigraph` | `from_paths`, `layout`, `draw` |
| `QuasiCartanMatrix` | `has_faithful_bigraph`, labeled `to_markdown`/`to_html`, `collapse` |
| `BlockCutTree` | `draw` |
| `ATypeCertificate`, `DCoreGluingCertificate`, `ACyclizationCertificate` | `to_markdown`, `_repr_markdown_` |
| `FlationSequence` | `to_markdown`, `_repr_markdown_` |
| `DisplayOptions` | `bigraph_layout` |

`dynkinlab.__all__` stays unchanged. This keeps the package-surface test and the user-facing import
story stable while still moving reusable notebook behavior into the library.

Alternative considered: a `dynkinlab.notebook` module or a `LabeledMatrix` class. Rejected because
the behavior is general graph/matrix behavior, and `Bigraph` already pairs a matrix with labels.

### 3. Path notation grammar

`Bigraph.from_paths(text, labels=None, *, arrow_multiplicity=2)` implements:

```text
graph  ::= line*
line   ::= path | comment
path   ::= vertex (edgeop vertex)*
vertex ::= integer | identifier
edgeop ::= "--" | ".." | "->" | "<-" | ".>" | "<."
```

Whitespace and commas separate tokens; `#` starts a comment; a line with one vertex declares an
isolated label. With arrow multiplicity `m`, the operators map as follows:

| operator | forward entry | reverse entry |
|---|---:|---:|
| `--` | -1 | -1 |
| `..` | +1 | +1 |
| `->` | -1 | -m |
| `<-` | -m | -1 |
| `.>` | +1 | +m |
| `<.` | +m | +1 |

Explicit labels pin matrix order and reject unknown path labels. Declaring the same unordered pair
twice with different entries raises. Invalid syntax, empty input without pinned labels, and an
invalid multiplicity raise with a line-aware message. The parser builds a `QuasiCartanMatrix` and
wraps it in `Bigraph`, so the matrix remains the source of truth.

This is a general DynkinLab notation, not a thesis parser. The fixed arrow multiplicity per string
covers B/C/F/G examples through `arrow_multiplicity`; graphs with mixed arrow multiplicities can use
direct `Bigraph` construction.

Alternatives considered: keeping the parser local (rejected by the code budget), a full DOT parser
(new grammar and potential dependency), and per-edge multiplicity syntax (unneeded for the
masterclass examples).

### 4. Faithful-view predicate and truthful edges

`QuasiCartanMatrix.has_faithful_bigraph` is a property computed directly from entries. For each
pair `i < j`:

- both zero is no edge;
- equal entries are an undirected edge;
- unequal entries with `min(abs(a), abs(b)) == 1` are an arrow;
- everything else is not faithful.

`Bigraph.edges` starts by checking the predicate and raises `ValueError("matrix has no faithful
bigraph representation")` when it is false. The `2/4` matrix now raises instead of producing a
multiplicity-four arrow. The classifier reads entries directly, so its `NOT_ADMISSIBLE` result is
unaffected.

The predicate lives on the matrix because it depends only on entries and because the notebook is
matrix-first. The masterclass never constructs a drawing for a non-faithful matrix; it displays the
labeled matrix, inertia, and classifier status only.

Alternative considered: putting `is_faithful` only on `Bigraph`. That would make the 2/4 matrix
require a wrapper that is not actually a faithful bigraph, which muddies the concept.

### 5. Labeled matrix rendering

`QuasiCartanMatrix.to_markdown(options=None, *, labels=None)` and
`to_html(options=None, *, labels=None, collapse=False)` accept display-only labels. Supplied labels
must be unique, hashable, and match the dimension. When present, they replace the index-base
headers; when absent, the current index-base behavior is unchanged. HTML labels are escaped and
Markdown labels escape `|`. The HTML table remains horizontally scrollable; `collapse=True` wraps it
in a `<details>` element whose summary names the dimension.

Labels are display metadata, not part of matrix identity. The D8 result uses the non-contiguous
labels `(1, 2, 3, 4, 5, 7, 8, 9)`, so this is not an optional convenience: index-based rendering
would silently mislabel it. The `_repr_html_` hook continues to use the canonical index base and
does not collapse, so existing notebook display behavior stays predictable.

Alternatives considered: a first-class `LabeledMatrix` (duplicates `Bigraph`), a
`Bigraph.to_matrix_html` method (makes the wrapper primary), and local HTML in the notebook
(rejected by the code budget).

### 6. Layout and drawing architecture

`bigraph_display.py` becomes the single rendering engine for both inline SVG and notebook figures:

- `compute_layout(bigraph, layout, seed, iterations)` returns a coordinate mapping for every label.
- `BigraphRenderer` draws onto an existing matplotlib axes and also produces the inline SVG.
- `BlockCutTreeRenderer` draws the bipartite or centroid block-cut tree onto an axes.
- `Bigraph.layout` delegates to `compute_layout`.
- `Bigraph.draw` delegates to the renderer and returns the axes.
- `BlockCutTree.draw` delegates to the tree renderer and returns the axes.

`bigraph.py` and `structure.py` use lazy imports of the renderers at the call site, matching the
existing `Bigraph.to_html` pattern and avoiding an import cycle. Return annotations use fully
qualified string forms such as `"matplotlib.axes.Axes"`; matplotlib is already a runtime dependency,
but package import stays light.

The layout policy is:

- explicit positions, when supplied, win;
- otherwise the named layout, or `DisplayOptions.bigraph_layout` when omitted;
- `auto`: Kamada-Kawai for a connected graph when SciPy is available, unweighted energy-based spring
  with the configured iteration count otherwise; a planar graph containing a four-vertex clique
  starts from a planar embedding and receives the same spring refinement, because force-directed
  layouts tend to draw the clique with crossing diagonals;
- `planar`, `kamada_kawai`, `spring`, `circular`: the corresponding NetworkX layout;
- a single vertex is placed at the origin;
- an unknown name raises `ValueError`;
- `spring` uses `DisplayOptions.bigraph_layout_seed` and
  `DisplayOptions.bigraph_layout_iterations` unless explicit values are supplied;
- layout distances use unweighted adjacency, because the public edge `weight` is a signed
  multiplicity rather than a drawing distance.

`planar` on a non-planar graph raises rather than silently switching policies; `auto` is the safe
choice for arbitrary input. The default is force-directed because the worked examples are highly
symmetric and the planar embedding, while crossing-free, looked rigid. `DisplayOptions.bigraph_layout`
defaults to `"auto"`, `bigraph_layout_iterations` defaults to 10000, both are validated on assignment,
participate in `reset()`, and are honored by both `to_html` and `draw`.

The masterclass shares positions only when they are computed. A3 before/after uses one layout; the
D13 core uses the gluing layout; D8 computes its own layout so the identified graph is not merely a
subgraph of the A9 picture. No coordinate literal is stored.

Alternatives considered: `nx.draw_networkx` (loses arrow direction and mixed-style control),
keeping the local matplotlib helper (rejected by the code budget), and manual coordinates (explicitly
rejected by the requested standard).

### 7. Block-cut tree rendering

`BlockCutTree.draw` has two layout modes. Without positions it uses `nx.bipartite_layout` on
`tree.to_networkx()`, with `tree.blocks` as the top partition and `tree.labels` as the bottom
partition. When the caller supplies the original graph's coordinates, it keeps articulation points
fixed, removes non-cut vertices from the figure, places each block node at the centroid of its
block's vertices, and connects blocks only to their articulation points. The notebook uses the
second mode with the A9 positions, so the two panels share the same cut-vertex geometry.

In both modes block nodes are squares, vertex nodes are circles, articulation points use the
highlight fill, and each block carries a `B1`, `B2`, ... label plus its label set. The centroid is
computed from the supplied positions, not stored. Alternative considered: spring layout for a more
organic tree; rejected because the tree structure and shared cut-vertex geometry are easier to read.

### 8. Certificate and witness Markdown

The structural certificates and flation sequences get `to_markdown()` and `_repr_markdown_` hooks,
following the existing display pattern:

- A-type: type, one row per block with labels and both partition sides, plus articulation points.
- D-core gluing: type, core order and styles, one row per piece with core edge, labels, and A-type.
- A-cyclization: endpoints, path, result labels, and type.
- Flation sequence: one row per step in application order, with equal endpoints called out as sign
  inversions.

The methods return plain strings and do not import IPython. They also leave `repr()` dataclass-style.
This removes the notebook's hand-built table loops and gives the certificate objects the same
display treatment as classification results.

Alternative considered: a generic table builder in the notebook. Rejected because it recreates the
shadow library and makes certificates less useful outside the notebook.

### 9. Masterclass narrative and code budget

The notebook is an eight-part lesson:

1. matrices, symmetrizers, quadratic forms, and exact inertia;
2. labeled matrices, faithful bigraphs, and the drawable range;
3. flation as a form-preserving change of basis;
4. finite Dynkin types and the classifier as a replayable oracle;
5. A-type structure through block-cut trees, with A9;
6. D-type gluing through an odd D-core, with D13;
7. D-type cyclization through an internal path, with A9 to D8;
8. the boundary table, algorithm caveat, exercises, and labs.

The voice is pedagogical, not paper-companion. It assumes linear algebra and graph theory, defines
quasi-Cartan matrices, symmetrizers, forms, inertia, flation, and Dynkin types before using them,
and gives proof ideas rather than full proofs. The path notation is introduced as a DynkinLab input
notation; the classifier is introduced as an independent bounded-flation oracle; no thesis-specific
algorithm is named or compared.

Code cells obey a hard budget:

- at most 40 physical lines;
- no `def` or `class`;
- preceded by Markdown that states the claim or construction;
- reusable behavior in the public API;
- no coordinate tables or literal position dictionaries;
- roughly 200 code lines in total.

The canonical tour is fixed and always runs. Labs are separate cells exposing `PATHS`,
`SOURCE`/`TARGET`, `X`/`Y`, and `LAYOUT`; they verify invariants such as preserved inertia,
certificate validity, or result dimension rather than hardcoding the canonical type. This keeps the
notebook editable without making the committed execution brittle.

Widgets are deliberately deferred. Plain parameter cells work in Colab and local Jupyter, need no
new test dependency, and read as code rather than UI configuration. A future change can add one
optional widget lab without changing the library.

Alternatives considered: keeping the paper format (rejected by the masterclass goal), adding
Colab `#@param` forms (Colab-specific), and putting all examples in one playground (loses the guided
tour).

### 10. Test strategy

Library tests:

- `from_paths`: A9 parse, explicit labels, isolated vertices, conflicts, invalid syntax, arrow
  multiplicity, and labels outside the pinned order.
- Faithful predicate: A2/A3 true, 2/4 false, 2/4 `edges` raises, unit arrow true, symmetric dotted
  true.
- Labeled matrix display: HTML/Markdown labels, non-contiguous labels, invalid labels, escaping,
  collapse default and collapsed output.
- Layout: auto Kamada-Kawai for connected graphs, planar-seeded relaxation for planar graphs with a
  four-vertex clique, spring fallback without SciPy or for disconnected graphs, configured spring
  iterations, named layouts, unknown layout, seeded repeatability, single vertex.
- Drawing: axes returned without IPython, mixed styles, highlights, shared positions, non-faithful
  rejection, block-cut tree nodes and articulation styling.
- Certificate/witness Markdown: tables, hooks, repr unchanged.

Notebook tests in `tests/test_notebook.py`:

- source/notebook cell types and normalized sources match;
- clean-kernel execution reaches the final version/certificate message;
- every code cell is at most 40 lines;
- AST inspection finds no `FunctionDef`, `AsyncFunctionDef`, or `ClassDef`;
- AST inspection finds no dictionary literal that looks like a coordinate table (two or more
  entries whose values are two-number sequences), while allowing computed `DictComp` positions;
- every code cell is preceded by a Markdown cell;
- the source contains no thesis-companion framing.

The structural guards are intentionally source-level. They catch the failure modes that caused the
redesign without trying to grade prose quality.

## Risks / Trade-offs

- **NetworkX layout algorithms can change across versions** → Layouts are deterministic for a given
  environment, the smoke test re-executes, and committed outputs are snapshots. The notebook offers
  named layouts so readers can choose a different view without editing coordinates.
- **Force-directed layouts are not guaranteed crossing-free** → `auto` prefers Kamada-Kawai for
  connected graphs and falls back to seeded spring with configurable iterations; `planar` remains
  an explicit policy for readers who need a crossing-free view.
- **`Bigraph.edges` now raises for the 2/4 matrix** → This is a deliberate breaking behavior for an
  invalid view. It is documented in the changelog; the classifier is unaffected; the matrix-only
  case uses matrix display.
- **Labels on matrix display can be confused with matrix identity** → Documented as display-only;
  equality and hashing remain matrix-based; invalid label sequences raise.
- **Certificate Markdown duplicates some formatting logic** → Each value owns its small table; the
  alternative is a new shared public display type that would enlarge the API more.
- **The 40-line cap can force awkward splits** → The API removes the helpers, and the boundary table
  is the only remaining loop-heavy cell; it stays within the cap by using a compact case list.
- **Removing the old notebook breaks existing Colab links** → The README, quickstart, and changelog
  point to the new file; no redirect stub is kept.
- **Auto layout changes existing inline SVG placement** → Tests assert visual structure, not exact
  coordinates; the canonical options still allow a per-call override.
- **The change is large** → It is sequenced into library affordances first and notebook rewrite
  second, with the full suite and docs gate before archive.

## Migration Plan

1. Add the library methods and display options with unit tests, keeping `__all__` unchanged.
2. Update the classifier-adjacent edge tests for the non-faithful rejection.
3. Create the masterclass source, generate the notebook with executed outputs, and add the
   structural notebook guards.
4. Remove the thesis-companion files and update the notebook guide, quickstart, API reference, and
   changelog.
5. Run `hatch check`, `hatch test`, `hatch run docs:build`, and `hatch run verify`.
6. Push the implementation to `main` so Colab can install the new API and open the new notebook.

Rollback is a normal git revert: the old notebook and its capability spec return with the previous
commit. There is no data or runtime migration.
