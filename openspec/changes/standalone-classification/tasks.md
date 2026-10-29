# Tasks

## 1. Neutralize source documentation

- [x] 1.1 Update the module, class, and method docstrings in `src/dynkinlab/classifier.py` to describe admissibility, the bounded flation search, the solid-arched invariant, and normal-form classification without source-relative wording; rename `_stage_three` to `_classify_normal_form`; verify `rg -n -i '\bpaper\b|\bthesis\b|Lemma|Stage [0-9]' src/dynkinlab/classifier.py` returns nothing and `hatch check` passes.
- [x] 1.2 Update the bound description in `src/dynkinlab/dynkin.py` and the flation docstring in `src/dynkinlab/quasi_cartan.py`; verify `rg -n -i '\bpaper\b|\bthesis\b' src/dynkinlab/dynkin.py src/dynkinlab/quasi_cartan.py` returns nothing and `hatch check` passes.
- [x] 1.3 Run the classifier tests after the docstring and helper changes; verify `hatch test tests/test_classifier.py tests/test_classifier_golden.py tests/test_classifier_oracle.py --no-cov` passes.

## 2. Rename test descriptions and examples

- [x] 2.1 Rename the paper- and thesis-named classes, methods, and docstrings in `tests/test_classifier_golden.py` and `tests/test_classifier.py` to describe the mathematical examples directly; keep every matrix and assertion unchanged; verify `hatch test tests/test_classifier_golden.py tests/test_classifier.py --no-cov` passes and `hatch test tests/test_classifier_golden.py tests/test_classifier.py --collect-only --no-cov` shows the new names.
- [x] 2.2 Rename the paper-specific test method and comment in `tests/test_bigraph.py` and the reference-flation docstring in `tests/test_quasi_cartan_general.py`; verify `hatch test tests/test_bigraph.py tests/test_quasi_cartan_general.py --no-cov` passes.
- [x] 2.3 Confirm no live test file mentions the source publication; verify `rg -n -i '\bpaper\b|\bthesis\b' tests` returns nothing.

## 3. Make user documentation self-contained

- [x] 3.1 Update `README.md` and `CHANGELOG.md` to use neutral classifier wording and describe the classifier as exact and self-contained; verify `hatch test tests/test_readme.py --no-cov` passes.
- [x] 3.2 Update `docs/getting-started/quickstart.md` and `docs/reference/index.md` to remove source-relative phrases; verify `rg -n -i '\bpaper\b|\bthesis\b' docs/getting-started/quickstart.md docs/reference/index.md` returns nothing.
- [x] 3.3 Add a concise "Classification algorithm" section to `docs/reference/index.md` covering admissibility conditions, the flation operation, the bounded search, the solid normal form, the G2 exception, and O(n^3) complexity; verify `hatch run docs:build` succeeds.
- [x] 3.4 Review the new explanation against the implementation and tests; verify each documented condition and outcome is exercised by an existing test or explicitly noted as a documented extension.

## 4. Validate the change

- [x] 4.1 Verify all live source, test, README, changelog, and documentation paths are free of source-publication references: run `rg -n -i '\bpaper\b|\bthesis\b' src tests README.md CHANGELOG.md docs/getting-started docs/reference` and confirm no matches.
- [x] 4.2 Run the full verification sequence and confirm the same classification outcomes: `hatch run verify` passes with the existing test count and coverage threshold.
- [x] 4.3 Validate the OpenSpec change and its delta: `openspec validate standalone-classification --type change --strict` passes.

## Workflow follow-up

- Archive the change after review so the `dynkin-classification` delta replaces the source-relative wording in the active spec.
- Verify the active spec has no source-publication references after archive.
