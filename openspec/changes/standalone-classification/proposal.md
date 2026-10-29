# Proposal

## Why

DynkinLab's classifier documentation, tests, and active specification currently describe the
algorithm by referring to an external paper and thesis. A reader cannot understand terms such as
"the paper's bound" or "Stage 2" without finding that source, which makes the library feel like a
companion to a publication rather than a standalone package. The library should explain and name its
own behavior.

## What Changes

- Replace paper and thesis references in live source docstrings, test names and docstrings, README,
  contributor-facing docs, and the active `dynkin-classification` specification with self-contained
  terminology.
- Rename the private `_stage_three` helper to `_classify_normal_form` so the implementation no longer
  uses source-relative stage numbering.
- Add a concise "Classification algorithm" section to the reference docs that defines admissibility,
  flation, the bounded search, the solid normal form, and the G2 exception.
- Keep archived OpenSpec changes unchanged as historical records.
- No public API, runtime behavior, dependency, or compatibility changes.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `dynkin-classification`: reword the admissibility, bound, G2, and non-finite outcome requirements
  so they describe the algorithm directly instead of referring to an external paper or thesis.

## Impact

- Source docstrings: `src/dynkinlab/classifier.py`, `src/dynkinlab/dynkin.py`, and
  `src/dynkinlab/quasi_cartan.py`; private helper rename in `classifier.py`.
- Tests: `tests/test_classifier.py`, `tests/test_classifier_golden.py`, `tests/test_bigraph.py`, and
  `tests/test_quasi_cartan_general.py` (names and docstrings only).
- Documentation: `README.md`, `CHANGELOG.md`, `docs/getting-started/quickstart.md`, and
  `docs/reference/index.md`.
- Active spec: `openspec/specs/dynkin-classification/spec.md` via a delta.
- Archived changes under `openspec/changes/archive/` are intentionally out of scope.
