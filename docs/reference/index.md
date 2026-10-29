# API reference

:::{admonition} tl;dr
:class: tldr

DynkinLab exposes the package version and the immutable values used for
quasi-Cartan matrix work. The mathematical surface grows with each change.
:::

## Current surface

- `dynkinlab.__version__` — the installed package version as a non-empty string.
- `dynkinlab.QuasiCartanMatrix` — an immutable symmetrizable integer matrix with diagonal entries 2.
- `dynkinlab.QuadraticForm` — the exact integer quadratic form of a quasi-Cartan matrix.
- `dynkinlab.Inertia` — the exact positive, negative, and zero eigenvalue counts.
- `dynkinlab.Bigraph`, `FlationStep`, `FlationSequence` — labeled mixed bigraphs and witnesses.
- `dynkinlab.DynkinType`, `DynkinClassification` — finite types and classifier results.
- `dynkinlab.display.options` — the notebook-wide mutable display settings.

## Display options

`dynkinlab.display.options` is the canonical `DisplayOptions` instance. It controls the
notebook-wide defaults for rich representations:

- `matrix_index_base` — first index used for matrix row and column headers. The default is `0`.
- `expanded_form_dimension_limit` — largest form dimension for which a quadratic form is expanded
  in Markdown. The default is `3`.
- `bigraph_layout_seed` — deterministic layout seed. The default is `0`.
- `bigraph_figure_size` — positive `(width, height)` pair. The default is `(4.0, 4.0)`.
- `bigraph_show_labels` — whether vertex labels are drawn. The default is `True`.
- `bigraph_show_multiplicity` — whether multiplicity labels are shown. The default is `True`.
- `bigraph_arrow_style` — `"arrow"` or `"line"`. The default is `"arrow"`.

```python
import dynkinlab

dynkinlab.display.options.matrix_index_base = 1
dynkinlab.display.options.expanded_form_dimension_limit = 4
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
special case `D = (1, ..., 1)`.

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

## Matrix display

`matrix.to_markdown()` returns a Markdown table and `matrix.to_html()` returns an HTML table. Both
use `dynkinlab.display.options.matrix_index_base` for row and column headers unless a per-call
`DisplayOptions` object is supplied. In Colab, the object's `_repr_html_` hook displays the HTML
table automatically.

```python
from dynkinlab import QuasiCartanMatrix

matrix = QuasiCartanMatrix(((2, -1), (-1, 2)))
print(matrix.to_markdown())
print(matrix.to_html())
```

## QuadraticForm

`matrix.quadratic_form` returns a `QuadraticForm`. For a matrix with symmetrizer `D = diag(d_i)`
and `S = D A`, its `evaluate(vector)` method computes

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
matrix `S = D A`. The result exposes `positive`, `negative`, and `zero` counts, plus `rank`,
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

`Bigraph(matrix, labels=None)` is a label-generic view over a `QuasiCartanMatrix`. It derives
undirected edges for equal entries and arrows for unequal magnitudes. The signed weight follows the
library's convention: the direction points from the smaller magnitude to the larger, and the style is
solid for a negative weight and dotted for a positive one.

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
source and target is the sign inversion used by the classifier.

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

## Dynkin types

`DynkinType(family, rank)` represents a finite Cartan-Killing type. The supported families and
ranks are `A_n` for `n >= 1`, `B_n` for `n >= 2`, `C_n` for `n >= 3`, `D_n` for `n >= 4`,
`E_6`, `E_7`, `E_8`, `F_4`, and `G_2`.

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
than 1. The 2-by-2 `G_2` matrix is the documented exception: its magnitude-3 entries bypass the
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

Later changes add the structural A/D decompositions and affine classification.
