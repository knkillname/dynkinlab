# Design

## Context

See `proposal.md` for motivation. The current `Symmetrizer.canonical` discovers connected components with NetworkX, propagates exact rational ratios over a spanning tree, and raises a generic `ValueError("matrix is not symmetrizable")` in three places: a zero entry on one side of an edge, opposite signs on an edge, and inconsistent ratios around a cycle. `QuasiCartanMatrix` calls this method during construction, so the exception is the natural failure boundary.

The implementation must preserve the existing canonical result: a primitive positive integer vector per connected component, with each component normalized independently. It must also preserve compatibility with callers that catch `ValueError`.

## Goals / Non-Goals

**Goals:**

- Attach a small, structured, independently verifiable certificate to every symmetrizability failure.
- Keep the constructor failure path exception-based; do not introduce a result object.
- Preserve the existing canonical symmetrizer algorithm and exact arithmetic.
- Put a very concise correctness proof in source comments.
- Use native Python 3.13 syntax; do not add `from __future__` imports.
- Expose the exception and certificate types through the top-level package.

**Non-Goals:**

- General non-diagonal symmetrizers.
- Rational or floating-point matrix inputs beyond the existing integer constructor.
- A Stiemke/Farkas dual-vector API.
- Changes to flation, inertia, the Dynkin classifier, or the supplied-symmetrizer validation path.
- Collecting multiple failures from one matrix; construction reports the first obstruction found in deterministic component traversal order.

## Decisions

### Native Python 3.13 syntax

Do not use `from __future__` imports. The package floor is Python 3.13, so the implementation, tests, and documentation examples use native language features instead. A final grep check enforces this during verification.

### Dedicated exception instead of a result object

Use `NotSymmetrizableError(ValueError)` with a public `certificate` attribute. The constructor already has failure semantics and callers already catch `ValueError`, so an exception subclass adds information without changing control flow. A `Result` union would force every existing caller to branch on success and would complicate `QuasiCartanMatrix`.

The exception carries only the message and the certificate. It does not retain the matrix, avoiding a large exception payload and unnecessary traceback retention. Implement `__reduce__` so the extra attribute survives pickling.

### Certificate shape

Use a frozen dataclass:

```python
@dataclass(frozen=True)
class SymmetrizerCertificate:
    kind: SymmetrizerObstruction
    vertices: tuple[int, ...]
```

`SymmetrizerObstruction` is a `StrEnum` with `zero_entry`, `opposite_signs`, and `inconsistent_cycle`. For the first two kinds, `vertices` is the edge `(i, j)` with a canonical orientation: for `zero_entry`, `a_ij == 0` and `a_ji != 0`; for `opposite_signs`, `i < j`. For `inconsistent_cycle`, `vertices` is the simple cycle in traversal order, with the closing edge implied.

`verify(entries)` recomputes the witness from the matrix:

- `zero_entry`: exactly one side of the named edge is zero, with the zero on the first side.
- `opposite_signs`: the product of the two edge entries is negative.
- `inconsistent_cycle`: every consecutive edge, including the closing edge, is present with nonzero same-sign entries, and the ratio product is not `1`.

A separate `describe(entries)` method formats the human-readable message. The exception message is `"matrix is not symmetrizable: " + certificate.describe(entries)`, preserving the existing `"symmetrizable"` match while adding the precise reason.

### Obstruction detection

Keep the existing component and ratio traversal, adding parent/depth bookkeeping only for cycle reconstruction:

- Zero side: `a_ij == 0` and `a_ji != 0` forces `d_j = 0`.
- Opposite signs: `a_ij * a_ji < 0` cannot balance with positive `d_i`, `d_j`.
- Ratio conflict: when a visited vertex is reached with a different value, reconstruct the fundamental cycle from the BFS tree plus the closing edge and emit an `inconsistent_cycle` certificate.

The first obstruction encountered is raised. Components are processed in the existing sorted order, so failure selection is deterministic.

### Concise proof comments

Place a four-line comment above the component solver:

```python
# Correctness: a zero side forces the opposite d to 0; opposite signs cannot
# balance positive d; otherwise every edge fixes d_j/d_i = a_ij/a_ji > 0.
# A connected graph is consistent iff every fundamental cycle product is 1;
# clearing denominators and dividing by gcd yields the unique primitive solution.
```

Add one short comment at each obstruction branch: `0 = d_j*a_ji forces d_j = 0`, `positive d cannot balance opposite signs`, and `a positive solution forces every cycle product to 1`. These comments are the complete correctness argument; no long docstring is added.

### Public surface

Export `NotSymmetrizableError`, `SymmetrizerCertificate`, and `SymmetrizerObstruction` from `dynkinlab`. Keep `Symmetrizer` itself internal, consistent with the current package surface. Update `__all__` and the public-surface test.

### Documentation

Update the reference page and README with a short failure example showing `error.certificate.verify(entries)`. Add a changelog entry under the unreleased section.

## Risks / Trade-offs

- **Exception attribute is extra API surface** → Keep it small and immutable; document it; test it; provide pickling support.
- **Cycle certificate can contain up to `n` vertices** → Still O(n) and much smaller than the matrix; only the first obstruction is stored.
- **Changing the exception type could break exact-type checks** → `NotSymmetrizableError` subclasses `ValueError`, so existing `except ValueError` and `pytest.raises(ValueError)` behavior remains valid.
- **Wrong certificate orientation for zero entries** → Construct the edge in canonical `(zero_side, nonzero_side)` order and test `verify` directly.
- **Existing generic error message might be asserted elsewhere** → Keep the word `symmetrizable` and the existing input-validation messages unchanged.
- **Proof comments can drift from code** → Keep the proof at the algorithm boundary and branch comments next to the checks they justify; tests cover all three obstructions.

## Migration Plan

No data or API migration is required. Construction of non-symmetrizable matrices continues to raise a `ValueError`; callers that want the certificate catch `NotSymmetrizableError` or inspect the narrower type. Rollback is a normal revert of the source, exports, tests, and documentation.

## Open Questions

None.
