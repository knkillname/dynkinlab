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

## Planned surface

Later changes add `Bigraph`, flation, and the Dynkin classifier.
