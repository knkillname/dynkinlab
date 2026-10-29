# Design

## Context

See `proposal.md` for motivation and the spec deltas for the behavioral contract. The current classifier keeps mutable component state as `list[list[int]]` and relies on Python's arbitrary-precision integers because some admissible indefinite inputs produce enormous intermediate entries. The public value objects are frozen dataclasses backed by tuples of Python integers so they remain immutable, hashable, and exactly comparable. Exact inertia is computed with `fractions.Fraction` and must stay free of floating-point eigenvalue approximations; the classifier's stated complexity is O(n^3).

NumPy is already present in every supported environment through matplotlib, but it is not declared directly. NetworkX is already a declared runtime dependency and is currently used only for bigraph display. The project's OpenSpec context forbids importing underscore-prefixed names from other modules; any helper shared across modules must be public.

Measurements behind the design:

- A 6x6 admissible symmetric unit form with entries in `{-1, 0, 1}` reaches a 24-digit entry after 36 flations; its inertia is `(5, 1, 0)`, so it is indefinite.
- Positive definite flation-equivalent matrices up to rank 13 stayed at `max |entry| = 3` across hundreds of thousands of flations.
- Affine `A_2` through `A_13` cycles, which are non-negative and degenerate, stayed at `max |entry| = 2` after 100,000 random flations each.
- The paper's non-positive-definite bound example has inertia `(3, 0, 1)` and stays at `max |entry| = 2` while exhausting its bound.

## Goals / Non-Goals

**Goals:**

- Reject indefinite components with exact inertia before they enter the flation loop.
- Use NumPy `int64` arrays for the classifier's post-admissibility state and vectorize flation and edge masks.
- Preserve exact Python-integer semantics at the public `QuasiCartanMatrix` boundary.
- Remove the duplicated flation formula and use NetworkX for graph structure.
- Remove the unused `SymmetrizedMatrix` adapter.
- Declare and verify a NumPy dependency floor.

**Non-Goals:**

- Classifying indefinite forms, or producing a partial witness for them.
- Replacing exact inertia with SymPy or any floating-point linear algebra.
- Adding SciPy.
- Changing the public value objects from tuples to arrays, or exposing `int64` at the API boundary.
- Changing the admissibility filter, the paper's flation stages, or the Dynkin normal-form rules.
- Adding affine or Euclidean classification.

## Decisions

### Component-level exact-inertia gate

`DynkinClassifier._classify_component` first handles the size-1, size-2 G2, and admissibility cases, then computes exact inertia for the component. If `negative > 0`, it returns a `_ComponentOutcome` with status `NOT_POSITIVE_DEFINITE`, reason `indefinite`, no steps, no bound applied, and no normal form. Non-negative components continue to the existing stages.

Placing the gate after `_admissible` keeps the current precedence: a matrix with an entry of magnitude 3 remains `NOT_ADMISSIBLE`, not `NOT_POSITIVE_DEFINITE`. Placing it at component level preserves the classified components before an indefinite component in a disconnected input. Restricting it to `negative > 0` preserves the existing bounded behavior for degenerate non-negative forms, including the paper's bound-exhaustion example and affine `A_2^(1)`.

Alternatives considered:

- **Whole-matrix gate before component splitting.** Simpler, but it would drop previously classified components from a mixed input and would run before the admissibility check unless duplicated there.
- **Reject every non-positive-definite component.** Would make the paper's PSD bound-exhaustion example return immediately, changing its reason and step count.
- **No gate, keep Python big integers.** Safe but keeps the hot path in Python objects and prevents meaningful NumPy use.

The component gate adds a second exact-inertia computation for connected inputs: the top-level `classify` still computes the overall result inertia, and each component computes its own gate inertia. Inertia is O(n^3) and currently sub-millisecond at Dynkin ranks, so the duplicate is accepted for clarity. If profiling shows it matters, compute component inertias once and sum them for the result.

### `int64` state only after admissibility and the gate

After the gate, the component's entries are converted to `np.ndarray(dtype=np.int64)`. All mutable stage helpers accept this array; the final normal form is converted back with `.tolist()` and tuple construction. The public `QuasiCartanMatrix` continues to store tuples of Python integers.

The admissibility filter guarantees off-diagonal magnitudes below 3 (and the G2 pre-check handles the magnitude-3 case), so the initial conversion cannot overflow. The non-negative gate limits the flation loop to forms whose entries stayed at 2-3 in the measurements. Object dtype would preserve exactness for arbitrary integers but is about five times slower for flation than Python lists; it is not used in the hot path.

Alternatives considered:

- **Object-dtype arrays everywhere.** Exact but slower than the current lists and not a performance win.
- **`int64` storage in the public value objects.** Breaks arbitrary-precision semantics and makes hashing/equality depend on a bounded representation.
- **SymPy matrices.** Exact but adds a 1.29-second import in the pinned Colab image and does not simplify the classifier's control flow.

### Explicit overflow guard

The shared flation helper checks the int64 result against `+/-2**31` after each flation and raises `ValueError` when the bound is exceeded. The threshold leaves enough headroom that the next flation's products and differences fit in int64. For the intended non-negative domain the maximum observed entry is 3, so the guard is a safety net rather than an expected branch. It prevents silent wrapping if a future input or algorithm change escapes the measured domain.

Alternatives considered:

- **Rely on the mathematical expectation that non-negative forms stay small.** No proof is attached, and silent corruption is unacceptable.
- **Fall back to Python integers on overflow.** Correct but adds a second classifier path and more code.
- **Raise only after the whole loop.** Would allow an intermediate wrap before detection.

### One flation helper with an explicit dtype

A public static method on `QuasiCartanMatrix` takes entries, source, target, and an explicit dtype, applies the paper's row/column formula from the original entries, and returns an array. `QuasiCartanMatrix.flation` passes `object` and converts the result to tuples, preserving exact arbitrary-precision behavior. The classifier passes `numpy.int64` and receives an array directly. The sign-inversion case (`source == target`) is handled inside the same helper.

This removes the duplicate formula in `QuasiCartanMatrix.flation` and `DynkinClassifier._flate_entries` without violating the rule against importing private names across modules.

Alternatives considered:

- **Keep the duplication.** Two formulas can drift, especially around the original-entry trap in the paper's appendix.
- **A private cross-module helper.** Forbidden by the project context.
- **Object dtype in the classifier as well.** Slower and defeats the purpose of the migration.

### NetworkX for graph structure

`Symmetrizer.components` builds an `nx.Graph` from nonzero off-diagonal entries, calls `nx.connected_components`, sorts each component's indices, and sorts the components by their smallest index. `DynkinType.from_bigraph` builds a graph of vertex indices, uses `nx.is_tree` to replace the manual edge-count/connectivity checks, `graph.degree()` for the degree logic, and `nx.single_source_shortest_path_length` for the branch distances. The arrow, leaf, and normal-form rules stay unchanged.

Alternatives considered:

- **`scipy.sparse.csgraph`.** Exact for graph traversal, but adds a dependency and duplicates NetworkX, which is already required.
- **Keep the manual component walk and BFS.** Works, but is more code and more places to get ordering wrong.

### Remove `SymmetrizedMatrix`

The adapter is not used by production paths; it duplicates validation and does not enforce the symmetry/even-diagonal assumptions that `QuadraticForm` and `Inertia` rely on. It is removed together with its tests. `MatrixLike` remains as the read-only protocol for `QuadraticForm` and `Inertia`, and the existing optional `symmetrized_entries` lookup continues to serve `QuasiCartanMatrix`.

Alternatives considered:

- **Use the adapter in `QuasiCartanMatrix.quadratic_form` and `inertia`.** Would remove some duck-typing but add wrapper allocation and require tightening the adapter's contract.
- **Keep it as dead code.** Contradicts the goal of reducing code.

### No type-checking-only imports

The project forbids `if TYPE_CHECKING` blocks. `QuasiCartanMatrix` and `Bigraph` need the `DynkinClassification` return type, but importing `dynkinlab.dynkin` at runtime would close the `quasi_cartan -> dynkin -> bigraph -> quasi_cartan` cycle. Instead, each module imports the package name (`import dynkinlab`) and uses a fully qualified string annotation such as `"dynkinlab.dynkin.DynkinClassification"`. The annotation is never evaluated at runtime, Pyrefly resolves it statically, and no conditional or private-name import is used. The obsolete `if TYPE_CHECKING:` coverage exclusion is removed from `pyproject.toml`.

### NumPy dependency floor

`pyproject.toml` declares `numpy>=2.1`, the first NumPy line supporting Python 3.13. The wheel smoke test in CI asserts the metadata requirement, and the pinned Colab smoke gate imports NumPy and asserts a version of at least 2.1. The pinned image currently ships NumPy 2.1.3.

Alternatives considered:

- **`numpy>=2.0`.** Does not support Python 3.13.
- **No direct declaration, rely on matplotlib.** Works accidentally but hides a direct import and breaks if matplotlib's dependency changes.

### Keep exact inertia in `Fraction`

The inertia reduction stays as-is. It is exact, O(n^3), and avoids the SymPy import cost. The new gate makes its result part of the classifier's control flow, but not a performance bottleneck at the expected matrix sizes. Replacing it with SymPy's charpoly and Descartes sign counting would save roughly 100 lines but would change the complexity promise and add a heavy import; that is deliberately left to a separate change if line count becomes the priority.

Alternatives considered:

- **SymPy `DomainMatrix.charpoly` plus Descartes.** Exact and correct, but not O(n^3) and 1.29 seconds to import in Colab.
- **SciPy `ldl`.** Floating point; forbidden by the exactness requirement.

## Risks / Trade-offs

- **Unexpected int64 growth** -> the `+/-2**31` guard raises `ValueError`; tests include the 6x6 indefinite example, random non-negative forms, and affine cycles. If the guard ever fires on a valid non-negative form, a fallback to Python integers is the next design step.
- **Changed indefinite diagnostics** -> marked breaking in the proposal; tests and docs will assert reason `indefinite`, zero steps, and an empty witness.
- **Duplicate exact-inertia work** -> accepted for simplicity; component-inertia summation is the optimization if profiles justify it.
- **NetworkX ordering changes** -> components are explicitly sorted; the classifier's edge choice and witness tests are re-run to confirm deterministic behavior.
- **Removal of `SymmetrizedMatrix`** -> not in `dynkinlab.__all__`, but importable from `dynkinlab.matrix`; changelog and docs call out the removal.
- **NumPy/CI drift** -> the dependency floor is checked in wheel metadata, the Colab smoke gate, and the existing Hatch environment.
- **Public `flation` with object dtype** -> exact for arbitrary integers, but not the fastest path for huge matrices; it is not the classifier's hot loop.

## Migration Plan

1. Add `numpy>=2.1` to `pyproject.toml`; update the wheel metadata smoke test and the pinned Colab smoke gate.
2. Add the shared dtype-aware flation helper; route `QuasiCartanMatrix.flation` through it with `object`; verify exactness and involution.
3. Add the component-level exact-inertia gate; add the indefinite-rejection tests and confirm the PSD bound examples are unchanged.
4. Convert the classifier's post-gate state to `int64`; vectorize flation and dotted-edge maintenance; add the overflow guard; compare classification results and witnesses against the current implementation on non-negative and positive definite inputs.
5. Replace `Symmetrizer.components` and the `DynkinType` adjacency/BFS code with NetworkX; re-run the existing classifier, oracle, and golden tests.
6. Remove `SymmetrizedMatrix` and its tests; update docs, `CHANGELOG.md`, and the README requirements.
7. Run `hatch check`, `hatch test`, `scripts/verify --skip-colab`, and the pinned Colab smoke gate.

Rollback is straightforward: the public classifier API is unchanged except for the indefinite diagnostics, and the dependency floor can stay because NumPy is already present through matplotlib. Reverting the gate and the int64 helpers restores the previous list/Fraction path.

## Open Questions

None that affect the specs, approach, or task breakdown.
