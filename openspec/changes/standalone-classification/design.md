# Design

## Context

See `proposal.md` — Why. The classifier already implements the behavior described by the active
specification; this change only rewrites how live surfaces describe and name that behavior.
Archived OpenSpec changes remain historical records.

## Goals / Non-Goals

**Goals:**

- Make live source docstrings, tests, user documentation, and the active specification self-contained.
- Preserve the algorithm, public API, runtime outputs, dependencies, and compatibility.
- Explain admissibility, flation, the bounded search, the normal form, and the G2 exception in
  DynkinLab's own terms.

**Non-Goals:**

- Renaming `flation`, `FlationStep`, `FlationSequence`, or other public API vocabulary.
- Removing or rewriting archived OpenSpec changes.
- Changing bounds, classification outcomes, witnesses, or complexity.
- Adding a separate attribution or references page; provenance stays in the archived changes.

## Decisions

1. **Restrict edits to live surfaces.** Source, tests, README, changelog, docs, and the active
   `dynkin-classification` spec are updated. Archived changes are left untouched because they record
   what was planned at the time and rewriting them would remove useful provenance.

2. **Use behavior-descriptive vocabulary.** Replace source-relative phrases with terms the library
   defines itself: “admissibility conditions,” “symmetric inflation bound,” “bounded flation
   search,” and “normal-form classification.” Avoid stage numbers where a phase name is clearer.

3. **Keep public API names.** `flation`, `FlationStep`, `FlationSequence`, and “cubic” remain. They
   are either established library vocabulary or a complexity description, and renaming them would
   be a breaking API change outside this proposal's scope.

4. **Put the self-contained explanation in `docs/reference/index.md`.** The reference page already
   explains matrices, forms, bigraphs, and classification. Adding a short algorithm section there
   keeps navigation stable; a dedicated page can be split out later if the material grows.

5. **Rename one private helper.** `_stage_three` becomes `_classify_normal_form`. Other internal
   names (`_admissible`, `_bound`, `_run_symmetric`, `_run_distinguished`) are already descriptive
   and do not need changes.

6. **Keep test matrices and assertions unchanged.** Rename test classes, methods, and docstrings to
   describe the mathematical examples rather than their source. This preserves coverage and makes
   the examples self-explanatory.

7. **Replace the requirement whose scenario is renamed.** OpenSpec matches scenarios by name and
   rejects a `MODIFIED` block that appears to drop an existing scenario; it has no scenario-level
   rename. The "Cubic classification of connected positive definite matrices" requirement is
   therefore removed and re-added as "Exact classification of connected positive definite matrices"
   with identical behavior and all five scenarios, one of which now names the symmetric inflation
   bound. The two requirements whose scenario names do not change use `MODIFIED`.

## Risks / Trade-offs

- **Removing live citations reduces immediate provenance** → The archived change keeps the full
  citation, while the live docs explain the algorithm directly.
- **The remove/add delta loses requirement identity** → All behavior and scenarios are preserved
  verbatim under the new name; validation and tests confirm equivalent outcomes.
- **A wording-only spec delta could appear to change behavior** → The delta keeps every scenario
  outcome; implementation tests verify the same results.
- **Self-contained docs could duplicate implementation detail** → Describe invariants and observable
  behavior, not private helpers or line-by-line formulas.
- **Renaming a private helper could break an overlooked reference** → Search all source and tests
  before renaming; the helper is private and not part of the public surface.
