# Design

## Context

See `proposal.md` for motivation and `specs/quasi-cartan-inertia/spec.md` for the behavior contract. The bootstrap change left a typed `src/dynkinlab` package whose public surface is only `__version__`. The project uses Hatch, Ruff, Pyrefly in strict mode, pytest with an 80% coverage gate, and the house style of frozen dataclasses, numpydoc docstrings, submodule imports, and no module-level functions.

This change is the first mathematical layer. It must stay small enough to validate the domain model before bigraphs and flation arrive.

## Goals / Non-Goals

**Goals:**

- A small, immutable value model for symmetric quasi-Cartan matrices.
- Exact integer quadratic-form evaluation with no floating-point arithmetic.
- Exact inertia using rational congruence reduction, including degenerate forms.
- A public API that later bigraph and classification changes can build on.

**Non-Goals:**

- Non-symmetric symmetrizable matrices and symmetrizer discovery.
- `Bigraph`, flation, Dynkin classification, or visualization.
- Performance work for large matrices beyond a clear O(n^3) rational algorithm.
- A general-purpose linear algebra package or a replacement for NumPy/SymPy.

## Decisions

### Symmetric quasi-Cartan matrices only

A quasi-Cartan matrix in the literature is a symmetrizable integer matrix with diagonal entries all equal to 2. This change implements the symmetric case: `a_ij == a_ji` for every pair and `a_ii == 2`. That is the case used by bigraphs and by the flation operation planned for the next change, and it gives an unambiguous quadratic form and inertia without choosing a symmetrizer.

Alternative considered: implement general symmetrizable matrices now. Rejected because different symmetrizers must be normalized, the inertia is then block-sensitive, and it would expand the first mathematical milestone before the bigraph model exists. A later change can add a separate symmetrizable type without breaking this one.

### Immutable tuple-backed value types

`QuasiCartanMatrix`, `QuadraticForm`, and `Inertia` are frozen dataclasses. Matrices store a tuple of tuples of `int`, not a list or a NumPy array, so values are immutable, hashable, comparable, and free of floating-point representation.

Construction copies the input, validates it once, and raises `ValueError` for an empty matrix, ragged or non-square input, non-integer entries, asymmetry, or a diagonal entry other than 2. Booleans are rejected even though `bool` subclasses `int`; they are not matrix integers.

Alternative considered: validate lazily on first use. Rejected because invalid values would exist as objects and fail far from the caller.

### Public API shape

The top-level package exports three names:

```python
from dynkinlab import Inertia, QuadraticForm, QuasiCartanMatrix
```

- `QuasiCartanMatrix(entries)` exposes `dimension`, `entries`, `quadratic_form`, and `inertia`.
- `QuadraticForm(matrix)` exposes `evaluate(vector) -> int`.
- `Inertia(positive, negative, zero)` exposes `rank`, `signature`, `is_positive_definite`, `is_negative_definite`, and `is_degenerate`; `Inertia.from_matrix(matrix)` computes the exact result.

`QuasiCartanMatrix.inertia` is the public convenience path for computing inertia. `Inertia.from_matrix(matrix)` is the public exact-reduction entry point; the result type is public so callers can compare and inspect it. No new exception hierarchy is introduced: invalid matrix or vector input raises `ValueError`.

Alternative considered: expose a module-level `exact_inertia(matrix)` function. Rejected because the house style forbids module-level functions and the matrix property is the natural object-oriented entry point. The word "exact" is a guarantee of the implementation, not a second method.

### Exact inertia by rational congruence reduction

Inertia is invariant under congruence. The implementation performs symmetric Gaussian elimination over `fractions.Fraction` on a mutable copy of the matrix:

1. Use a nonzero diagonal entry as a 1×1 pivot, eliminate the row and column, and count its sign.
2. If the current diagonal pivot is zero but a nonzero off-diagonal entry remains, permute it into a 2×2 block. A block `[[0, b], [b, c]]` with `b != 0` has one positive and one negative eigenvalue, so count one of each and eliminate the remaining rows and columns using the block's inverse.
3. If the remaining block is entirely zero, count every remaining dimension as a zero eigenvalue and stop.

The result is a triple of integer counts. The algorithm never calls `float`, `math.sqrt`, or a numerical eigenvalue routine.

Alternative considered: `numpy.linalg.eigvalsh` with a tolerance. Rejected because exactness is the point of the change and tolerance choices misclassify degenerate forms. Alternative considered: exact characteristic polynomial plus Sturm sequences. Rejected as more code and more edge cases than congruence reduction for the expected small matrices.

The algorithm is O(n^3) rational operations and is intended for the small matrices used in Dynkin and bigraph work. If profiling later shows a problem, fraction-free Bareiss elimination can replace the inner loop without changing the public API.

### Quadratic form convention

For a symmetric matrix `A`, the form is

```text
q(x) = sum_i x_i^2 + sum_{i < j} a_ij x_i x_j
```

which is the integer-valued half of `x^T A x`. The factor of one half is why the diagonal contributes `x_i^2` once rather than twice. Evaluation validates the vector length and integer entries and returns a Python `int`; it never converts to a float.

Alternative considered: expose `x^T A x` directly. Rejected because the integer-valued form is the one used in the quasi-Cartan/root-system literature and avoids a constant factor in examples.

### Module and test layout

```text
src/dynkinlab/_matrix.py         MatrixLike protocol used by forms and inertia
src/dynkinlab/quasi_cartan.py    QuasiCartanMatrix
src/dynkinlab/quadratic_form.py  QuadraticForm
src/dynkinlab/inertia.py         Inertia and exact reduction
src/dynkinlab/__init__.py        public exports and __all__
tests/test_quasi_cartan.py       construction and validation
tests/test_quadratic_form.py     formula and vector validation
tests/test_inertia.py            known cases, pivots, properties
tests/test_package.py            updated public surface
```

The form and inertia modules accept the small `MatrixLike` structural protocol rather than importing the concrete matrix class. This keeps the modules acyclic and avoids type-checking-only imports.

Documentation updates go in `README.md`, `docs/getting-started/quickstart.md`, and `docs/reference/index.md`. The README keeps exactly one executable Python block so `tests/test_readme.py` continues to exercise it.

### Testing strategy

Known matrices cover the contract:

- `A_2`, `A_3`, `D_4`, and `E_8` Cartan matrices are positive definite.
- The affine `A_2^(1)` matrix has one zero eigenvalue.
- `((2, 3), (3, 2))` is indefinite.
- A matrix with a zero pivot after elimination exercises the zero-pivot path.

Hypothesis property tests cover:

- Permuting rows and columns together preserves inertia.
- Congruence by a diagonal matrix of entries ±1 preserves inertia.
- Inertia of a block diagonal of two quasi-Cartan matrices is the componentwise sum of their inertias.
- Rejected inputs always raise `ValueError` and never produce a partially constructed value.

The 80% coverage gate is unchanged. No new runtime dependency is needed.

## Risks / Trade-offs

- **Symmetric-only scope may be narrower than the literature** → Document the restriction in the API and defer general symmetrizable matrices to a separate change.
- **Rational arithmetic can grow for large matrices** → Keep the algorithm simple and O(n^3); the expected matrices are small, and a fraction-free optimization can follow without an API change.
- **Zero-pivot handling is easy to get wrong** → Use explicit 2×2 pivots, cover degenerate and zero-pivot examples, and add property tests for invariance.
- **Public API growth is hard to undo** → Export only the three value types and keep helper methods on those types; no classifier or bigraph names leak into this change.
- **Documentation drift** → The README snippet is executable, and the docs build treats warnings as errors.

## Migration Plan

This is an additive API change. Existing imports of `__version__` continue to work. If the change is reverted, remove the three new modules, their tests, the documentation additions, and the new `__all__` entries; no data or package metadata migration is involved.

## Open Questions

None that affect the specs, approach, or task breakdown. The exact algorithm can be optimized later without changing observable behavior.
