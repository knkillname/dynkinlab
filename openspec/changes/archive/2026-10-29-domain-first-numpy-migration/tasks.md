# Tasks

## 1. Dependency and packaging metadata

- [x] 1.1 Add `numpy>=2.1` to `[project].dependencies` in `pyproject.toml`; verify `hatch run python -c "import numpy"` succeeds and the file lists the floor.
- [x] 1.2 Update the wheel metadata smoke in `.github/workflows/ci.yml` to require `numpy>=2.1`; verify the workflow YAML parses and the metadata assertion matches `pyproject.toml`.
- [x] 1.3 Update `scripts/colab-smoke` to import NumPy and assert version `>=2.1`; verify `scripts/colab-smoke` passes against the pinned image.
- [x] 1.4 Update the requirements text in `README.md`, `docs/index.md`, and `docs/getting-started/quickstart.md` to include NumPy 2.1; verify `hatch test tests/test_readme.py` passes and the docs mention the new floor.
- [x] 1.5 Add an `Unreleased` changelog entry for the NumPy dependency and the classifier domain behavior; verify the entry names the added dependency and the early indefinite rejection.

## 2. Shared flation helper

- [x] 2.1 Add a public dtype-aware `QuasiCartanMatrix.flate_entries(entries, source, target, dtype)` helper that applies the paper's row/column formula from the original entries and handles `source == target`; verify unit tests cover both sign inversion and distinct source/target with `object` and `int64` dtypes.
- [x] 2.2 Route `QuasiCartanMatrix.flation` through the helper with `object` dtype and convert the result back to tuples; verify `tests/test_quasi_cartan_general.py::TestFlation` passes and a new test with integers beyond int64 proves exactness and involution.
- [x] 2.3 Add a randomized reference test that compares the helper output with a straightforward independent formula for small matrices; verify the test passes for symmetric and non-symmetric inputs, including invalid indices rejected by the public method.
- [x] 2.4 Update the helper and `flation` docstrings to state the dtype contract; verify `hatch check` passes on the touched modules.

## 3. Exact-inertia gate

- [x] 3.1 Add the component-level gate after the size-1, G2, and admissibility checks: if exact inertia has `negative > 0`, return `NOT_POSITIVE_DEFINITE` with reason `indefinite`, zero steps, and an empty witness; verify a new test using the 6x6 indefinite example asserts all four fields.
- [x] 3.2 Add regression tests that the paper's bound-exhaustion example still reports `bound exhausted` with `steps == bound == 12`, affine `A_2^(1)` still reports `NOT_POSITIVE_DEFINITE`, and the no-distinguished-vertex PSD case still reaches Stage 1; verify `tests/test_classifier_golden.py` and `tests/test_classifier.py` pass.
- [x] 3.3 Add a disconnected test with a classified positive definite component and an indefinite component; verify the classified component and labels are retained while the overall status is `NOT_POSITIVE_DEFINITE` with reason `indefinite`.
- [x] 3.4 Document the early indefinite rejection in `docs/reference/index.md` and the changelog; verify the docs text names reason `indefinite` and the zero-step/empty-witness behavior.

## 4. int64 classifier state

- [x] 4.1 Convert the post-gate component state to `np.ndarray(dtype=np.int64)` and update `_run_symmetric`, `_run_naive`, `_run_distinguished`, `_stage_three`, and `_SymmetricState` to accept arrays; convert final normal forms back to tuples; verify the existing classifier, golden, and oracle tests pass.
- [x] 4.2 Route the classifier's flation through the shared helper with `np.int64` and add the `+/-2**31` overflow guard that raises `ValueError`; verify a direct helper test with an oversized entry raises, while normal non-negative inputs do not.
- [x] 4.3 Vectorize dotted-edge maintenance, `_first_positive`, and `_solid_arched` while preserving the existing edge-choice order and O(n) dotted-index updates; verify `test_example_three_one_exhausts_the_bound`, `test_b5_witness_contains_sign_inversion`, and `test_dotted_bigraph_uses_labels_in_witness` pass.
- [x] 4.4 Extend the oracle/equivalence tests with canonical, flated, permuted, and affine-cycle inputs; verify statuses, component types, step counts, bounds, and witness verification match the expected results.
- [x] 4.5 Confirm the int64 path does not rebuild a full O(n^2) dotted mask after every flation; verify by inspection and by running the classifier on `D_13` and `A_13` golden cases without test timeouts.

## 5. NetworkX graph layer

- [x] 5.1 Replace `Symmetrizer.components` with a NetworkX graph and `nx.connected_components`, sorting each component and the component list deterministically; verify disconnected symmetrizer and classifier tests pass.
- [x] 5.2 Replace `DynkinType`'s manual adjacency/connectivity checks and `_distances` BFS with `nx.Graph`, `nx.is_tree`, `graph.degree`, and `nx.single_source_shortest_path_length`, keeping the branch/arrow rules unchanged; verify `tests/test_dynkin.py` and the canonical round-trip tests pass.
- [x] 5.3 Add explicit ordering/determinism tests for components and graph classification; verify repeated runs return the same component order and labels.
- [x] 5.4 Update any reference text that describes the hand-written graph walk or BFS; verify the docs build still passes.

## 6. Remove the unused adapter

- [x] 6.1 Delete `SymmetrizedMatrix` from `src/dynkinlab/matrix.py` and remove its tests from `tests/test_quasi_cartan_general.py`, keeping `MatrixLike`; verify `grep -R "SymmetrizedMatrix" src tests` returns no matches.
- [x] 6.2 Update `CHANGELOG.md` and any reference text to note the removal; verify the changelog names `SymmetrizedMatrix` and the docs build still passes.
- [x] 6.3 Re-run `hatch test` and verify the coverage gate remains at least 80% after the deleted tests.

## 7. Integration verification

- [x] 7.1 Run `hatch check`; verify formatting, linting, and strict type checking pass.
- [x] 7.2 Run `hatch test`; verify all tests pass and coverage is at least 80%.
- [x] 7.3 Run `scripts/verify --skip-colab`; verify the host gates pass and the Colab gate is reported as skipped.
- [x] 7.4 Run `scripts/colab-smoke`; verify the pinned Colab gate passes with NumPy at or above 2.1.
- [x] 7.5 Build the wheel, install it in a clean environment, and inspect its metadata; verify `numpy>=2.1` is declared and `import dynkinlab` succeeds.
- [x] 7.6 Run `hatch run docs:build`; verify the Sphinx build passes with warnings treated as errors.
- [x] 7.7 Run `openspec validate "domain-first-numpy-migration" --strict`; verify the change remains valid.

## Workflow follow-up

- Archive the change after the project's review requirements are satisfied.
- Create a Git commit after archiving, naming the archived change in the commit message.
- Verify the archived result.
