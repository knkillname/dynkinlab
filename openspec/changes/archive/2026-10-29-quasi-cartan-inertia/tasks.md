# Tasks

## 1. Quasi-Cartan matrix model

- [x] 1.1 Create `src/dynkinlab/quasi_cartan.py` with a frozen `QuasiCartanMatrix` that stores a tuple of tuples, validates non-empty square symmetric integer input with diagonal 2, rejects booleans, and exposes `dimension` and `entries`; add `tests/test_quasi_cartan.py` covering construction, copying, equality, and every rejection case; verify `hatch test tests/test_quasi_cartan.py` passes.
- [x] 1.2 Export `QuasiCartanMatrix` from `dynkinlab`, update `tests/test_package.py` to import it and include it in the `__all__` assertion, and add a `QuasiCartanMatrix` section to `docs/reference/index.md`; verify `hatch test tests/test_package.py` and `hatch run docs:build` pass.

## 2. Integer quadratic form

- [x] 2.1 Create `src/dynkinlab/quadratic_form.py` with a frozen `QuadraticForm` that validates vector length and integer entries and evaluates `q(x) = sum(x_i**2) + sum(a_ij * x_i * x_j for i < j)` using Python integers; add `tests/test_quadratic_form.py` covering known values, zero and negative coordinates, wrong-length vectors, and non-integer entries; verify `hatch test tests/test_quadratic_form.py` passes.
- [x] 2.2 Add `QuasiCartanMatrix.quadratic_form` returning the form, export `QuadraticForm` from `dynkinlab`, update `tests/test_package.py`, and add a `QuadraticForm` section to `docs/reference/index.md`; verify the package tests and `hatch run docs:build` pass.

## 3. Exact inertia

- [x] 3.1 Create `src/dynkinlab/inertia.py` with a frozen `Inertia(positive, negative, zero)` result exposing `rank`, `signature`, `is_positive_definite`, `is_negative_definite`, and `is_degenerate`, plus an exact rational congruence reduction using 1×1 and 2×2 pivots and no floating-point arithmetic; add `tests/test_inertia.py` covering the `A_2`, `A_3`, `D_4`, and `E_8` Cartan matrices, affine `A_2^(1)`, an indefinite matrix, a zero-pivot case, and the result invariants; verify `hatch test tests/test_inertia.py` passes.
- [x] 3.2 Add Hypothesis property tests to `tests/test_inertia.py` for simultaneous row/column permutation, congruence by a diagonal of ±1 entries, block-diagonal additivity, and invalid-input rejection; verify `hatch test tests/test_inertia.py` passes.
- [x] 3.3 Add `QuasiCartanMatrix.inertia`, export `Inertia` from `dynkinlab`, and update `tests/test_package.py` to assert the complete public surface of `__version__`, `QuasiCartanMatrix`, `QuadraticForm`, and `Inertia`; verify `hatch test tests/test_package.py` passes.
- [x] 3.4 Replace the README quickstart with one executable matrix/form/inertia example, add the inertia section to `docs/reference/index.md`, extend `docs/getting-started/quickstart.md`, and add an `Unreleased` entry to `CHANGELOG.md`; verify `hatch test tests/test_readme.py` and `hatch run docs:build` pass.

## 4. Integration verification

- [x] 4.1 Run `hatch check --fix` twice and then `hatch test`; verify the second check leaves the tree unchanged and the test run enforces at least 80% coverage.
- [x] 4.2 Run `scripts/verify --skip-colab`; verify the host quality gates pass and the Colab gate is reported as skipped.
- [x] 4.3 Run `scripts/verify`; verify the pinned Colab smoke gate still passes with Python 3.13 and the dependency floors.
- [x] 4.4 Rebuild the wheel, install it in a clean virtual environment, and import `QuasiCartanMatrix`, `QuadraticForm`, and `Inertia`; verify the installed wheel works without the source tree and `hatch run docs:build` still passes with warnings treated as errors.
- [x] 4.5 Inspect the source and public surface; verify no `Bigraph`, flation, Dynkin classifier, or non-symmetric symmetrizer code was added and the top-level package exports only the documented names.

## Workflow follow-up

- Archive the change after review and a green full verification run.
- Create the Git commit for the archived change, as required by the project's archive guidance.
- Start the next change for `Bigraph`, flation, and the Dynkin classifier after this inertia milestone is archived.
