# API reference

:::{admonition} tl;dr
:class: tldr

DynkinLab exposes the package version and the immutable values used for
quasi-Cartan matrix work. The mathematical surface grows with each change.
:::

## Current surface

- `dynkinlab.__version__` — the installed package version as a non-empty string.
- `dynkinlab.QuasiCartanMatrix` — an immutable symmetric integer matrix with diagonal entries 2.
- `dynkinlab.QuadraticForm` — the exact integer quadratic form of a quasi-Cartan matrix.
- `dynkinlab.Inertia` — the exact positive, negative, and zero eigenvalue counts.
- `dynkinlab.display.options` — the notebook-wide mutable display settings.

## Display options

`dynkinlab.display.options` is the canonical `DisplayOptions` instance. It controls the
notebook-wide defaults for rich representations:

- `matrix_index_base` — first index used for matrix row and column headers. The default is `0`.
- `expanded_form_dimension_limit` — largest form dimension for which a quadratic form is expanded
  in Markdown. The default is `3`.

```python
import dynkinlab

dynkinlab.display.options.matrix_index_base = 1
dynkinlab.display.options.expanded_form_dimension_limit = 4

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
non-empty, symmetric, and have 2 on every diagonal entry.

```python
from dynkinlab import QuasiCartanMatrix

matrix = QuasiCartanMatrix(((2, -1), (-1, 2)))
print(matrix.dimension)  # 2
print(matrix.entries)  # ((2, -1), (-1, 2))
```

Invalid input raises `ValueError` during construction.

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

`matrix.quadratic_form` returns a `QuadraticForm`. Its `evaluate(vector)` method computes

```text
q(x) = sum(x_i**2) + sum(a_ij * x_i * x_j for i < j)
```

exactly with Python integers.

```python
from dynkinlab import QuasiCartanMatrix

form = QuasiCartanMatrix(((2, -1), (-1, 2))).quadratic_form
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

`matrix.inertia` and `Inertia.from_matrix(matrix)` compute the exact inertia of a quasi-Cartan
matrix. The result exposes `positive`, `negative`, and `zero` counts, plus `rank`,
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

## Planned surface

Later changes add `Bigraph`, flation, and the Dynkin classifier.
