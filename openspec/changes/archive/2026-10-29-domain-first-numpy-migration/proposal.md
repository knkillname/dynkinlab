# Proposal

## Why

The classifier's mutable state uses Python lists and arbitrary-precision integers. That was necessary because admissible but indefinite inputs can blow up during the bounded flation loop: a 6x6 admissible symmetric unit form with entries in `{-1, 0, 1}` reaches a 24-digit entry after 36 flations. Those blow-ups are confined to indefinite forms, which are outside the non-negative unit-form domain the classifier is meant to classify. Exact inertia already identifies them, so the classifier can reject them before entering the flation loop instead of carrying big integers through the hot path.

## What Changes

- Add an exact-inertia gate in component classification. After the admissibility filter, a component with a negative eigenvalue is returned as `NOT_POSITIVE_DEFINITE` with reason `"indefinite"` and no flation steps. Non-negative components, including degenerate affine forms, keep the existing bounded flation behavior.
- Use NumPy `int64` arrays for the classifier's mutable component state after admissibility and the non-negative gate. Flation and dotted-edge masks become vectorized operations. Add an explicit overflow guard that raises `ValueError` instead of silently wrapping.
- Keep exact Python-integer semantics at the public `QuasiCartanMatrix` boundary. The public `flation` path uses exact object arithmetic; only the classifier's post-admissibility state is `int64`.
- Extract one shared flation implementation used by both `QuasiCartanMatrix.flation` and the classifier, removing the duplicated row/column formula.
- Use NetworkX for connected components and graph traversal in `Symmetrizer` and `DynkinType`, replacing the hand-written component walk and BFS distance helper.
- Remove the unused `SymmetrizedMatrix` adapter and its tests. `MatrixLike` remains as the read-only protocol used by `QuadraticForm` and `Inertia`.
- Declare NumPy as a direct runtime dependency and assert its floor in the pinned Colab smoke gate.
- **BREAKING**: indefinite admissible inputs now report reason `"indefinite"` with zero steps and an empty witness instead of a partial bound-exhaustion witness. The overall status remains `NOT_POSITIVE_DEFINITE`.
- **BREAKING**: `dynkinlab.matrix.SymmetrizedMatrix` is removed. It is not in `dynkinlab.__all__`, but it was importable from its submodule.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `dynkin-classification`: Add early rejection of indefinite components by exact inertia; keep the bounded flation loop for non-negative components and preserve the existing PSD bound-exhaustion behavior.
- `package-foundation`: Add NumPy to the declared runtime dependency floors.
- `colab-compatibility`: Assert NumPy at or above the declared floor in the pinned Colab smoke gate.

## Impact

- `src/dynkinlab/classifier.py`: exact-inertia gate, NumPy `int64` component state, vectorized flation/masks, overflow guard.
- `src/dynkinlab/quasi_cartan.py`: shared flation helper and exact public path.
- `src/dynkinlab/symmetrizer.py`, `src/dynkinlab/dynkin.py`: NetworkX-based components and graph traversal.
- `src/dynkinlab/matrix.py`: remove `SymmetrizedMatrix`; keep the `MatrixLike` protocol.
- `pyproject.toml`, `scripts/colab-smoke`, `.github/workflows/ci.yml`: NumPy dependency floor and metadata/runtime assertions.
- Tests: early-indefinite rejection, classifier equivalence on non-negative inputs, shared flation, NetworkX behavior, NumPy metadata, and removal of the deleted adapter tests.
- Docs: state the non-negative domain and the early indefinite rejection behavior.
