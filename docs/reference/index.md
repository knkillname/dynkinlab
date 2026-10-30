# API reference

:::{admonition} tl;dr
:class: tldr

DynkinLab exposes the package version and the immutable values used for
quasi-Cartan matrix work and structural A/D certificates. The mathematical
surface grows with each change.
:::

## Current surface

- `dynkinlab.__version__` — the installed package version as a non-empty string.
- `dynkinlab.QuasiCartanMatrix` — an immutable symmetrizable integer matrix with diagonal entries 2.
- `dynkinlab.QuadraticForm` — the exact integer quadratic form of a quasi-Cartan matrix.
- `dynkinlab.Inertia` — the exact positive, negative, and zero eigenvalue counts.
- `dynkinlab.Bigraph`, `FlationStep`, `FlationSequence` — labeled mixed bigraphs and witnesses.
- `dynkinlab.BlockCutForest`, `ATypeCertificate`, `ABlockPartition` — structural A-type inspection
  and certificates.
- `dynkinlab.EdgeMultiset`, `DCore`, `DCoreGluingCertificate`, `ACyclizationCertificate` — structural
  D-type gluing and cyclization certificates.
- `dynkinlab.DynkinType`, `DynkinClassification` — finite types and classifier results.
- `dynkinlab.display.options` — the notebook-wide mutable display settings.

## Display options

`dynkinlab.display.options` is the canonical `DisplayOptions` instance. It controls the
notebook-wide defaults for rich representations:

- `matrix_index_base` — first index used for matrix row and column headers. The default is `0`.
- `expanded_form_dimension_limit` — largest form dimension for which a quadratic form is expanded
  in Markdown. The default is `3`.
- `bigraph_layout` — layout policy: `"auto"`, `"planar"`, `"kamada_kawai"`, `"spring"`, or
  `"circular"`. The default is `"auto"`. `kamada_kawai` requires SciPy.
- `bigraph_layout_iterations` — spring-layout iteration count. The default is `10000`.
- `bigraph_layout_seed` — deterministic layout seed. The default is `0`.
- `bigraph_figure_size` — positive `(width, height)` pair. The default is `(4.0, 4.0)`.
- `bigraph_show_labels` — whether vertex labels are drawn. The default is `True`.
- `bigraph_show_multiplicity` — whether multiplicity labels are shown. The default is `True`.
- `bigraph_arrow_style` — `"arrow"` or `"line"`. The default is `"arrow"`.

```python
import dynkinlab

dynkinlab.display.options.matrix_index_base = 1
dynkinlab.display.options.expanded_form_dimension_limit = 4
dynkinlab.display.options.bigraph_layout = "spring"
dynkinlab.display.options.bigraph_layout_iterations = 10000
dynkinlab.display.options.bigraph_layout_seed = 7
dynkinlab.display.options.bigraph_show_multiplicity = False

dynkinlab.display.options.reset()
```

Rendering methods accept an optional `DisplayOptions` object for a per-call override. The override
takes precedence over `dynkinlab.display.options` for that call only; the canonical object remains
unchanged.

```python
from dynkinlab.display import DisplayOptions

options = DisplayOptions(matrix_index_base=0)
```

## QuasiCartanMatrix

`QuasiCartanMatrix(entries)` copies and validates a square matrix of integers. The matrix must be
non-empty, square, and have 2 on every diagonal entry. It must also be symmetrizable: there must be
a positive diagonal integer symmetrizer `D` such that `D A` is symmetric. Symmetric matrices are the
special case $D = (1,\dots,1)$.

```python
from dynkinlab import QuasiCartanMatrix

matrix = QuasiCartanMatrix(((2, -1), (-1, 2)))
print(matrix.dimension)  # 2
print(matrix.entries)  # ((2, -1), (-1, 2))
print(matrix.symmetrizer)  # (1, 1)
print(matrix.is_symmetric)  # True

non_symmetric = QuasiCartanMatrix(((2, -1), (-2, 2)))
print(non_symmetric.symmetrizer)  # (2, 1)
print(non_symmetric.symmetrized_entries)  # ((4, -2), (-2, 2))
```

`QuasiCartanMatrix.symmetric(entries)` is a strict factory that rejects non-symmetric matrices,
while the general constructor accepts both symmetric and non-symmetric symmetrizable matrices.
Invalid input raises `ValueError` during construction. A valid matrix with no positive diagonal
symmetrizer raises `NotSymmetrizableError`, a `ValueError` subclass carrying a verifiable
`SymmetrizerCertificate`:

```python
from dynkinlab import NotSymmetrizableError, QuasiCartanMatrix

entries = ((2, -1, -2), (-2, 2, -1), (-1, -2, 2))
try:
    QuasiCartanMatrix(entries)
except NotSymmetrizableError as error:
    print(error)
    assert error.certificate.verify(entries)
```

The certificate names the obstruction: a zero entry on one side of an edge, opposite signs on an
edge, or an inconsistent cycle of edge ratios. Use `certificate.kind` and `certificate.vertices` to
inspect it programmatically.

`matrix.flation(source, target)` applies the generalized flation and returns a new matrix. Equal
source and target indices perform the sign inversion used by the classifier.

```python
from dynkinlab import QuasiCartanMatrix

matrix = QuasiCartanMatrix(((2, 1), (1, 2)))
print(matrix.flation(0, 1).entries)  # ((2, -1), (-1, 2))
```

`matrix.has_faithful_bigraph` reports whether every off-diagonal pair is zero, symmetric, or an
arrow with one side of magnitude one. The predicate agrees with edge derivation: requesting the
edges of a non-faithful view raises `ValueError` instead of inventing a lossy arrow.

```python
from dynkinlab import Bigraph, QuasiCartanMatrix

a2 = QuasiCartanMatrix(((2, -1), (-1, 2)))
assert a2.has_faithful_bigraph
assert len(Bigraph(a2, ("a", "b")).edges) == 1

not_faithful = QuasiCartanMatrix(((2, 2), (4, 2)))
assert not not_faithful.has_faithful_bigraph
try:
    Bigraph(not_faithful, (1, 2)).edges
except ValueError:
    pass
else:
    raise AssertionError("a non-faithful matrix must not produce edges")
```

## Matrix display

`matrix.to_markdown()` returns a Markdown table and `matrix.to_html()` returns an HTML table. Both
use `dynkinlab.display.options.matrix_index_base` for row and column headers unless a per-call
`DisplayOptions` object is supplied. In Colab, the object's `_repr_html_` hook displays the HTML
table automatically.

The optional `labels` argument supplies display-only headers for the rows and columns. Labels must
match the dimension, be unique, and be hashable; they do not change matrix equality or hashing. The
HTML renderer also accepts `collapse=True` to wrap a wide table in a collapsed disclosure.

```python
from dynkinlab import QuasiCartanMatrix

matrix = QuasiCartanMatrix(((2, -1), (-1, 2)))
print(matrix.to_markdown(labels=("a", "b")))
print(matrix.to_html(labels=("a", "b")))
print(matrix.to_html(labels=("a", "b"), collapse=True))
```

## QuadraticForm

`matrix.quadratic_form` returns a `QuadraticForm`. For a matrix with symmetrizer
$D = \operatorname{diag}(d_i)$ and $S = DA$, its `evaluate(vector)` method computes

```text
q(x) = sum(d_i * x_i**2) + sum(s_ij * x_i * x_j for i < j)
```

exactly with Python integers.

```python
from dynkinlab import QuasiCartanMatrix

form = QuasiCartanMatrix(((2, -1), (-1, 2))).quadratic_form
print(form.evaluate((1, 1)))  # 1

form = QuasiCartanMatrix(((2, -1), (-2, 2))).quadratic_form
print(form.evaluate((1, 1)))  # 1
```

A vector with the wrong length or a non-integer entry raises `ValueError`.

## Form display

`form.to_markdown()` returns the general quadratic-form formula and, when the form is small enough,
the expanded polynomial. The expansion limit comes from
`dynkinlab.display.options.expanded_form_dimension_limit` or a per-call `DisplayOptions` object. In
Colab, `QuadraticForm` displays through `_repr_markdown_`.

```python
from dynkinlab import QuasiCartanMatrix

form = QuasiCartanMatrix(((2, -1), (-1, 2))).quadratic_form
print(form.to_markdown())
```

## Inertia

`matrix.inertia` and `Inertia.from_matrix(matrix)` compute the exact inertia of the symmetrized
matrix $S = DA$. The result exposes `positive`, `negative`, and `zero` counts, plus `rank`,
`signature`, `is_positive_definite`, `is_negative_definite`, and `is_degenerate`.

```python
from dynkinlab import Inertia, QuasiCartanMatrix

matrix = QuasiCartanMatrix(((2, -1), (-1, 2)))
inertia = matrix.inertia
assert inertia == Inertia.from_matrix(matrix) == Inertia(2, 0, 0)
print(inertia.rank, inertia.signature, inertia.is_positive_definite)  # 2 2 True
```

Inertia is computed by exact rational congruence reduction, so zero eigenvalues are reported as
zero rather than as a floating-point tolerance.

## Inertia display

`inertia.to_markdown()` returns a compact table with the positive, negative, and zero counts, rank,
and signature, followed by a definiteness or degeneracy label. In Colab, `Inertia` displays through
`_repr_markdown_`.

## Bigraph and flation

`Bigraph.from_paths(text, labels=None, *, arrow_multiplicity=2)` builds a labeled bigraph from the
DynkinLab path notation. Each line is a path, `#` starts a comment, and whitespace or commas separate
tokens. A line with one vertex declares an isolated label. The six operators are:

| operator | entries |
|---|---|
| `--` | `-1, -1` |
| `..` | `+1, +1` |
| `->` | `-1, -m` |
| `<-` | `-m, -1` |
| `.>` | `+1, +m` |
| `<.` | `+m, +1` |

$m$ is the `arrow_multiplicity`. Explicit labels pin matrix order and reject path labels outside the
order; conflicting declarations of the same pair raise `ValueError`.

```python
from dynkinlab import Bigraph

a3 = Bigraph.from_paths("1--2--3")
print(a3.labels)  # (1, 2, 3)
print(a3.matrix.entries)

pinned = Bigraph.from_paths("b--a", labels=("a", "b"))
print(pinned.labels)  # ('a', 'b')
```

`Bigraph(matrix, labels=None)` is a label-generic view over a `QuasiCartanMatrix`. It derives
undirected edges for equal entries and arrows for unequal magnitudes. The signed weight follows the
library's convention: the direction points from the smaller magnitude to the larger, and the style is
solid for a negative weight and dotted for a positive one. The matrix must pass
`has_faithful_bigraph`; otherwise `Bigraph.edges` raises `ValueError` rather than emitting a lossy
multiplicity arrow.

```python
from dynkinlab import Bigraph, FlationSequence, FlationStep, QuasiCartanMatrix

matrix = QuasiCartanMatrix(((2, -1), (-2, 2)))
bigraph = Bigraph(matrix, ("a", "b"))
print(bigraph.edges)  # (BigraphEdge(source='a', target='b', directed=True, ...),)
print(bigraph.flation("a", "b").edges)

sequence = FlationSequence((FlationStep("a", "b"), FlationStep("a", "a")))
final = sequence.apply(bigraph)
assert sequence.verify(bigraph, final)
assert sequence.inverse().apply(final) == bigraph
```

`Bigraph.flation(source, target)` delegates to the underlying matrix and returns a new bigraph with
the same labels. `FlationStep` and `FlationSequence` are first-class witnesses; a step with equal
source and target is the sign inversion used by the classifier. `FlationSequence.to_markdown()` lists
the steps in application order and labels equal endpoints as a sign inversion.

## Bigraph display

`Bigraph.to_markdown()` returns a compact table of directions, styles, and multiplicities.
`Bigraph.to_html()` and `_repr_html_` return a self-contained inline SVG with labeled vertices,
solid/dotted edges, arrows, and multiplicity labels. The rendering is deterministic for a given
layout seed and uses no IPython dependency.

```python
from dynkinlab import Bigraph, QuasiCartanMatrix

bigraph = Bigraph(QuasiCartanMatrix(((2, -1), (-2, 2))), ("a", "b"))
print(bigraph.to_markdown())
print(bigraph.to_html())
```

`Bigraph.layout(layout=None, *, seed=None, iterations=None, options=None)` returns a coordinate
mapping for every label. The policy comes from the `layout` argument, or from
`dynkinlab.display.options.bigraph_layout` when omitted. `"auto"` prefers Kamada-Kawai for connected
graphs when SciPy is available and otherwise uses an unweighted, energy-based `nx.spring_layout`
with the configured iteration count. Layout distances ignore the signed solid/dotted edge weights.
A planar graph containing a four-vertex clique starts from `nx.planar_layout` and is
refined by the energy-based spring layout, which avoids the crossing diagonals that force-directed
layouts often give a $K_4$. The named policies are `"planar"`, `"kamada_kawai"`, `"spring"`, and
`"circular"`; `"kamada_kawai"` requires SciPy. An unknown policy raises `ValueError`, and
`"planar"` on a non-planar graph propagates the NetworkX error rather than silently switching
policies.

```python
from dynkinlab import Bigraph

bigraph = Bigraph.from_paths("1--2--3")
positions = bigraph.layout("circular")
assert set(positions) == set(bigraph.labels)
assert bigraph.layout("spring", seed=7, iterations=10000) == bigraph.layout(
    "spring", seed=7, iterations=10000
)
```

`Bigraph.draw(*, ax=None, layout=None, seed=None, iterations=None, positions=None,
options=None, title=None, highlight_edges=(), highlight_nodes=(),
display_labels=None)` renders the bigraph on a matplotlib axes and returns it. It preserves solid and dotted styles, arrow direction, multiplicity, and labels;
optional highlights and display labels are supported. Supplying `positions` reuses a computed
mapping. The method does not import IPython.

```python
from dynkinlab import Bigraph

bigraph = Bigraph.from_paths("1--2--3")
positions = bigraph.layout()
axes = bigraph.draw(positions=positions, title="A3")
assert axes.figure is not None
```

## Edge multisets

`EdgeMultiset(labels, edges)` represents undirected solid and dotted edges before simplification.
Same-style parallel edges combine into one edge with a larger multiplicity, while opposite-style
parallel edges remain separate. This is what allows a two-edge D-core to keep a solid and a dotted
edge between the same pair.

```python
from dynkinlab import BigraphEdge, EdgeMultiset, EdgeStyle

multiset = EdgeMultiset(
    ("a", "b"),
    (
        BigraphEdge("a", "b", False, EdgeStyle.SOLID, 1),
        BigraphEdge("a", "b", False, EdgeStyle.DOTTED, 1),
    ),
)
assert len(multiset.edges) == 2
assert multiset.simplify().edges == ()
```

`EdgeMultiset.from_bigraph(bigraph)` copies a symmetric bigraph's labels and edges. `+` unions
labels in first-appearance order and adds edge multiplicities. `identify(source, target)` merges two
labels and drops the resulting loop. `simplify()` cancels opposite-style pairs and returns a
`Bigraph`; labels without incident edges remain isolated vertices.

## Bigraph sums and vertex identification

`Bigraph.sum(*pieces, labels=None)` identifies shared labels, adds off-diagonal weights, keeps
diagonal entries at 2, and simplifies opposite parallel edges. The default label order is first
appearance; an explicit order must cover exactly the union of the piece labels.

```python
from dynkinlab import Bigraph, QuasiCartanMatrix

a2 = Bigraph(QuasiCartanMatrix(((2, -1), (-1, 2))), ("a", "b"))
a3 = Bigraph(QuasiCartanMatrix(((2, -1, 0), (-1, 2, -1), (0, -1, 2))), ("b", "c", "d"))
summed = Bigraph.sum(a2, a3)
assert summed.labels == ("a", "b", "c", "d")
assert len(summed.edges) == 3

cancelled = Bigraph.sum(
    Bigraph(QuasiCartanMatrix(((2, -1), (-1, 2))), ("a", "b")),
    Bigraph(QuasiCartanMatrix(((2, 1), (1, 2))), ("a", "b")),
)
assert cancelled.edges == ()
```

`Bigraph.identify(source, target)` merges the target label into the source label, drops any loop
created by the merge, and simplifies opposite parallel edges. The retained label keeps its original
position.

```python
merged = summed.identify("a", "c")
assert merged.labels == ("a", "b", "d")
```

## Structural bigraphs

`Bigraph.to_networkx()` returns an undirected `networkx.Graph` with every label as a node and each
derived edge carrying `style`, `directed`, `multiplicity`, and `weight`. This is the underlying graph
used by the structural operations and is useful for custom figures.

```python
from dynkinlab import Bigraph, QuasiCartanMatrix

bigraph = Bigraph(QuasiCartanMatrix(((2, -1), (-1, 2))), ("a", "b"))
graph = bigraph.to_networkx()
assert graph.edges["a", "b"]["style"] == "solid"
```

`Bigraph.block_cut_forest()` returns a `BlockCutForest` with one `BlockCutTree` per connected
component. Each tree exposes `labels`, `blocks`, and `articulation_points`; every `Block` carries its
labels and induced bigraph. `BlockCutTree.to_networkx()` returns a bipartite graph with block nodes
marked `kind="block"` and vertex nodes marked `kind="vertex"`.

```python
forest = bigraph.block_cut_forest()
tree = forest.trees[0]
assert tree.verify(bigraph)
print(tree.articulation_points)
```

`BlockCutTree.draw(*, ax=None, options=None, title=None, positions=None)` renders the tree on a
matplotlib axes. Block nodes are squares, vertex nodes are circles, articulation points are
highlighted, and every block shows its label set. By default the tree uses
`nx.bipartite_layout`. When `positions` supplies coordinates from the original graph, articulation
points keep those coordinates, non-cut vertices are removed, and each block node is placed at the
centroid of its block's vertices. The method returns the axes.

```python
from dynkinlab import Bigraph

bigraph = Bigraph.from_paths("1--2--3")
tree = bigraph.block_cut_forest().trees[0]
axes = tree.draw(title="A3 block-cut tree")
assert axes.figure is not None

positions = bigraph.layout()
axes = tree.draw(positions=positions, title="A3 tree at graph coordinates")
assert axes.figure is not None
```

`Bigraph.is_a_block(labels)` returns an `ABlockPartition` when the induced vertex set is an A-block,
and `None` otherwise. The two sides X and Y are disjoint: every cross pair is solid and every pair
within a side is dotted. A block with no solid edges is accepted with X equal to all labels and Y
empty.

```python
partition = bigraph.is_a_block(("a", "b"))
assert partition is not None and partition.verify(bigraph, ("a", "b"))
print(partition.x, partition.y)
```

`Bigraph.a_type_certificate()` returns an `ATypeCertificate` when every connected component is
A-type, and `None` otherwise. The certificate contains one `ATypeComponentCertificate` per
component, with the component's $A_n$ type, block-cut tree, and A-block partitions. For the 9-vertex
A-block tree, the certificate has one $A_9$ component and four block partitions:

```python
from dynkinlab import Bigraph, QuasiCartanMatrix

a9_edges = {
    (1, 2): -1, (1, 3): -1, (2, 3): 1,
    (3, 4): 1, (3, 5): -1, (3, 9): -1,
    (4, 5): -1, (4, 7): -1, (4, 9): -1,
    (5, 6): 1, (5, 8): 1, (5, 9): 1, (6, 8): 1,
}
rows = [[2 if row == column else 0 for column in range(9)] for row in range(9)]
for (source, target), weight in a9_edges.items():
    rows[source - 1][target - 1] = weight
    rows[target - 1][source - 1] = weight
a9 = Bigraph(QuasiCartanMatrix(rows), tuple(range(1, 10)))

certificate = a9.a_type_certificate()
assert certificate is not None and certificate.verify(a9)
print([component.dynkin_type.name for component in certificate.components])
print([block.labels for block in a9.block_cut_forest().trees[0].blocks])
```

`ATypeCertificate.to_markdown()` renders the component types, one row per block with both partition
sides, and the articulation points. `_repr_markdown_` returns the same string for notebook display;
`repr()` stays dataclass-style.

`Bigraph.is_type_a()` is the boolean convenience. The certificate is structural: it does not change
`Bigraph.classify()`, so the 13-vertex D-core example still classifies as $D_{13}$ while its A-type
certificate is absent.

## D-core cycles

`DCore.cycle(labels, styles)` builds a cyclic D-core from labels in cyclic order and one style per
position. A core of three or more labels is a simple cycle. A two-label core is the two-edge double
link with one solid and one dotted edge. `cycle_condition()` reports whether the number of dotted
core edges is odd; construction rejects even parity, chords, duplicate labels, missing cycle edges,
and non-simple multiplicities.

```python
from dynkinlab import DCore, EdgeStyle

core = DCore.cycle(
    (1, 2, 3, 4, 5),
    (EdgeStyle.DOTTED, EdgeStyle.SOLID, EdgeStyle.DOTTED, EdgeStyle.DOTTED, EdgeStyle.SOLID),
)
assert core.cycle_condition()

double_link = DCore.cycle((1, 2), (EdgeStyle.SOLID, EdgeStyle.DOTTED))
assert len(double_link.edges.edges) == 2
```

### D-core gluing

`DCore.gluing(pieces, labels=None)` takes one connected positive definite A-type piece per core
edge. It checks that each piece contains its core edge with the same style, that both endpoints are
internal vertices, and that consecutive pieces overlap only at the shared core vertex. It returns a
`DCoreGluingCertificate` whose `verify()` recomputes the construction.

```python
from dynkinlab import Bigraph, DCore, EdgeStyle, QuasiCartanMatrix

first = Bigraph(QuasiCartanMatrix(((2, -1), (-1, 2))), (1, 2))
second = Bigraph(
    QuasiCartanMatrix(
        ((2, 1, -1, 0), (1, 2, -1, 0), (-1, -1, 2, -1), (0, 0, -1, 2)),
    ),
    (1, 2, 3, 4),
)
core = DCore.cycle((1, 2), (EdgeStyle.SOLID, EdgeStyle.DOTTED))
certificate = core.gluing((first, second), labels=(1, 2, 3, 4))
assert certificate.verify()
assert certificate.dynkin_type.name == "D_4"
```

The certificate exposes `core`, `pieces`, `gluing`, `dynkin_type`, and `verify()`.
`DCoreGluingCertificate.to_markdown()` lists the type, core order, and one row per piece with its core
edge, vertex labels, and A-type. The D13 example uses the five pieces A3, A4, A2, A7, and A2 along the
odd five-cycle and produces the 13-vertex D13 bigraph.

### A-cyclization

`Bigraph.a_cyclization(source, target)` requires a connected positive definite A-type bigraph with at
least five vertices. The two labels must be internal vertices joined by a unique shortest path of
length at least two with an odd number of dotted edges. Identifying them produces D_{n-1}, returned
as an `ACyclizationCertificate` with `source`, `x`, `y`, `path`, `result`, `dynkin_type`, and
`verify()`.

```python
from dynkinlab import Bigraph, QuasiCartanMatrix

source = Bigraph(
    QuasiCartanMatrix(
        (
            (2, -1, 0, 0, 0),
            (-1, 2, -1, 0, 0),
            (0, -1, 2, 1, 0),
            (0, 0, 1, 2, -1),
            (0, 0, 0, -1, 2),
        ),
    ),
    (1, 2, 3, 4, 5),
)
certificate = source.a_cyclization(1, 5)
assert certificate.verify()
assert certificate.dynkin_type.name == "D_4"
```

`ACyclizationCertificate.to_markdown()` reports the source labels, identified endpoints, shortest
path, result labels, and Dynkin type. The A9 example uses internal vertices 1 and 6 along the path
`1-3-5-6` to produce D8.

## Dynkin types

`DynkinType(family, rank)` represents a finite Cartan-Killing type. The supported families and
ranks are $A_n$ for $n \ge 1$, $B_n$ for $n \ge 2$, $C_n$ for $n \ge 3$, $D_n$ for $n \ge 4$,
$E_6$, $E_7$, $E_8$, $F_4$, and $G_2$.

```python
from dynkinlab import Bigraph, DynkinType

dynkin_type = DynkinType("A", 3)
print(dynkin_type.name)  # A_3
matrix = dynkin_type.cartan_matrix()
bigraph = dynkin_type.bigraph()
assert DynkinType.from_matrix(matrix) == dynkin_type
assert DynkinType.from_bigraph(Bigraph(matrix)) == dynkin_type
```

`DynkinType.from_matrix(matrix)` and `DynkinType.from_bigraph(bigraph)` apply the final
solid-bigraph classification stage to a connected solid mixed bigraph. They return `None` when the
graph is not a finite Dynkin diagram.

## Classification algorithm

The classifier works on connected components of the nonzero off-diagonal graph.

**Admissibility.** A component is admissible when every off-diagonal entry has magnitude at most 2,
each nonzero pair is symmetrizable with consistent signs, and no pair has both magnitudes greater
than 1. The $2 \times 2$ $G_2$ matrix is the documented exception: its magnitude-3 entries bypass the
admissibility filter and are classified directly.

**Flation search.** A flation is the integer row/column operation that replaces one row and column
using another; setting source and target equal performs a sign inversion. Flations preserve the
Dynkin type. The classifier applies them only to dotted edges and maintains a dotted-edge index so
the symmetric search stays within its O(n^3) bound. Symmetric components run until no dotted edge
remains, bounded by the dimension-dependent symmetric inflation bound. Non-symmetric components
first choose a distinguished vertex (a source or sink carrying every arrow) and make the bigraph
solid-arched; for n > 4 they run the symmetric search on the graph without that vertex, and for
n <= 4 they use a bounded direct search.

**Normal form.** When the search ends with a solid bigraph, the component is read as its connected
Dynkin diagram. A search that exhausts its bound, cannot complete preprocessing, or reaches a
non-Dynkin normal form returns `NOT_POSITIVE_DEFINITE` with a reason. An admissible component with a
negative eigenvalue is rejected by exact inertia before any flation.

The classifier uses exact integer arithmetic and performs O(n^3) operations for admissible input; it
does not rely on floating-point eigenvalues. `DynkinType.from_matrix` and
`DynkinType.from_bigraph` expose the final normal-form classification directly.

## Dynkin classification

`matrix.classify()` returns a `DynkinClassification` result. `Bigraph.classify()` does the same
while preserving the bigraph labels. The result exposes `status`, `reason`, exact `inertia`,
`components`, the first-class flation `witness`, the `normal_form`, `steps`, `bound`, and `verify()`.

```python
from dynkinlab import Bigraph, ClassificationStatus, DynkinType, QuasiCartanMatrix

matrix = DynkinType("B", 5).cartan_matrix()
result = matrix.classify()
assert result.status is ClassificationStatus.CLASSIFIED
assert result.components[0].dynkin_type == DynkinType("B", 5)
assert result.verify()

labeled = Bigraph(matrix, ("a", "b", "c", "d", "e")).classify()
assert labeled.components[0].labels == ("a", "b", "c", "d", "e")
```

The classifier is exact and bounded. `NOT_ADMISSIBLE` reports a filter failure. Admissible indefinite
components are rejected by exact inertia before any flation, so the result reports
`NOT_POSITIVE_DEFINITE` with reason `indefinite`, zero steps, and an empty witness. Non-negative
components keep the bounded behavior, where `NOT_POSITIVE_DEFINITE` reports bound exhaustion or a
non-Dynkin normal form. G2 is handled as a documented O(1) extension to the admissibility
conditions.

## Classification display

`DynkinType.to_markdown()` returns the canonical type name and `_repr_markdown_` is available for
notebook display. `DynkinClassification.to_markdown()` includes the status, exact inertia, component
labels and types, step count, applied bound, and a normal-form summary.

```python
from dynkinlab import DynkinType

result = DynkinType("A", 3).cartan_matrix().classify()
print(result.to_markdown())
```

## Planned surface

Later changes add affine classification.
