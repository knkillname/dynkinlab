# Tasks

## 1. General quasi-Cartan matrix, symmetrizer, and flation

- [x] 1.1 Add the `SymmetrizedMatrix` matrix-like adapter and keep the existing `MatrixLike` protocol for form and inertia consumers; verify `hatch check` accepts the adapter and existing matrix-like imports still work.
- [x] 1.2 Generalize `QuadraticForm` to use the diagonal coefficient `S_ii / 2`; verify the existing `A_2` form tests still pass and a diagonal-4 adapter evaluates as `2*x_0^2`.
- [x] 1.3 Broaden `QuasiCartanMatrix` construction and validation to shape, integers, diagonal 2, and symmetrizability, with an optional explicit symmetrizer and a symmetric fast path; verify symmetric `A_2`, non-symmetric `B_2`, and the paper's directed-cycle no-symmetrizer matrix.
- [x] 1.4 Implement the canonical minimal-trace symmetrizer helper by connected component, including disconnected normalization; verify the `B_2` symmetrizer is `(2, 1)`, symmetric `A_2` is `(1, 1)`, and a disconnected example is normalized per block.
- [x] 1.5 Add `is_symmetric` and the strict `QuasiCartanMatrix.symmetric(entries)` factory; verify true/false reporting, symmetric acceptance, and rejection of the non-symmetric `B_2` matrix by the factory.
- [x] 1.6 Add `symmetrizer`, `symmetrized_entries`, `quadratic_form`, and `inertia` to `QuasiCartanMatrix` using `S = D A`; verify the symmetric form value, the non-symmetric `B_2` value at `(1, 1)`, positive definiteness, and the affine zero count.
- [x] 1.7 Implement generalized flation on `QuasiCartanMatrix`, including sign inversion and the original-entry row/column update; verify involution, inertia preservation, sign inversion, and that entries outside the target row/column are unchanged.
- [x] 1.8 Update the matrix reference documentation for the general constructor, symmetrizer, strict symmetric factory, and flation; verify the public-import test and `hatch run docs:build` pass.

## 2. Bigraph view and flation witnesses

- [x] 2.1 Implement the public edge style and immutable edge value with source, target, direction, style, and multiplicity; verify valid values are hashable and invalid endpoints, styles, or multiplicities raise `ValueError`.
- [x] 2.2 Implement the label-generic `Bigraph` view over `QuasiCartanMatrix`, including label defaults, uniqueness/length validation, and matrix preservation; verify string labels, default labels, duplicate labels, and wrong-length labels.
- [x] 2.3 Implement mixed edge and arrow derivation with the paper's signed max-magnitude convention; verify symmetric, negative asymmetric, positive asymmetric, zero, and paper Figure 1 cases.
- [x] 2.4 Implement `Bigraph.flation` delegation and label mapping, including source equal to target; verify dotted-to-solid flips, sign inversion, unknown-label rejection, and inertia preservation.
- [x] 2.5 Implement `FlationStep` and `FlationSequence` with apply, inverse, and verify; verify two-step replay, inverse recovery, successful verification, and rejection of a wrong final graph.
- [x] 2.6 Export the bigraph and flation names and document their reference surface; verify the public-import test and `hatch run docs:build` pass.

## 3. Dynkin types and Stage 3 classification

- [x] 3.1 Implement the immutable `DynkinType` value with family, rank, canonical name, equality, and hashing; verify valid A–G types and rejection of `A_0`, `B_1`, `C_2`, `D_3`, `E_5`, `F_3`, and `G_3`.
- [x] 3.2 Implement canonical Cartan matrices and bigraphs for A, B, C, D, E6/E7/E8, F4, and G2, including B/C arrow orientation and the G2 weight-3 arrow; verify each round-trips through the matrix and bigraph views.
- [x] 3.3 Implement Stage 3 graph classification, including cycle, parallel-edge, degree, arrow-count, D/E distance, path A/B/C/F4, and G2 branches; verify every canonical type and a non-Dynkin tree.
- [x] 3.4 Export `DynkinType` and document its reference surface; verify the public-import test and `hatch run docs:build` pass.

## 4. Classifier Stage 1 and G2

- [x] 4.1 Implement connected-component splitting for a `QuasiCartanMatrix` in deterministic index order; verify mixed A2/E6, isolated A1, and duplicate A2 components.
- [x] 4.2 Implement the G2 pre-check for both 2-by-2 orientations and its trivial witness/normal form; verify both orientations classify as G2 before the admissibility filter.
- [x] 4.3 Implement the admissibility filter for off-diagonal magnitude bounds, parallel undirected edges, and local sign/ratio requirements; verify non-admissible matrices return the specified status.
- [x] 4.4 Implement distinguished-vertex detection for non-symmetric components, including the n = 4 F4 exception; verify source, sink, no-distinguished, and F4 cases.
- [x] 4.5 Implement the dotted-edge index and the solid-arched repair by sign inversion at the non-distinguished endpoint; verify the invariant holds, the sign inversions are recorded, and the paper's mixed example is handled.
- [x] 4.6 Document the Stage 1 decisions and add focused tests for the filter, distinguished vertex, and solid-arched repair; verify the new tests and the existing matrix/bigraph tests pass.

## 5. Classifier Stage 2

- [x] 5.1 Implement the symmetric bounded inflation loop with the paper's `l(n)` bound and the dotted-edge index; verify canonical A/D/E inputs, step counts within the bound, and dotted-edge removal.
- [x] 5.2 Implement the non-symmetric `n > 4` reduction to `G - x`, including the Lemma 4.2 direction choice that preserves solid-arched; verify B/C/F4 inputs and the invariant after every step.
- [x] 5.3 Implement the non-symmetric `n <= 4` branch with the 24-iteration bound; verify B2/C2/F4 and a non-positive-definite n <= 4 input.
- [x] 5.4 Maintain the dotted-edge index after each flation in O(n) and update the matrix with the paper's entry formula; verify no full-matrix rescan is required and the step count remains within the bound.
- [x] 5.5 Implement bound-exhaustion handling with a partial witness and a `NOT_POSITIVE_DEFINITE` outcome; verify the paper's Example 3.1 and the affine `A_2^(1)` matrix terminate.

## 6. Classification result and integration

- [x] 6.1 Implement the immutable `DynkinClassification` result with status, reason, inertia, components, witness, normal form, steps, bound, input, and `verify()`; verify each field and witness replay.
- [x] 6.2 Wire the classifier stages into the result, including component labels, normal-form bigraphs, exact inertia, and failure reasons; verify classified and non-finite results.
- [x] 6.3 Implement `Bigraph.classify` delegation with label-preserving components and witness steps; verify a labeled disconnected bigraph and a labeled classified bigraph.
- [x] 6.4 Add a test-only naive generalized-inflations oracle with a conservative root-count bound and property-test the optimized classifier against it on small random matrices; verify agreement on type or non-finite status.
- [x] 6.5 Add golden tests for every canonical type, G2, paper Figure 1, paper Example 3.1, the thesis 13-vertex D-core, and the thesis 9-vertex A-block; verify types, statuses, normal forms, and witness replay.
- [x] 6.6 Add randomized positive definite generation by flation/sign-inversion/transposition and verify the classifier recovers the starting Dynkin type.
- [x] 6.7 Export `DynkinClassification` and the status value, document the classifier result surface, and verify the public-import test and `hatch run docs:build` pass.

## 7. Colab display

- [x] 7.1 Extend `DisplayOptions` with bigraph layout seed, figure size, label visibility, multiplicity visibility, and arrow style; verify defaults, assignment validation, reset, and per-call overrides.
- [x] 7.2 Extend the `QuasiCartanMatrix` display tests to non-symmetric entries, verifying that Markdown, HTML, and the notebook hook show the original `A` entries and the configured index base.
- [x] 7.3 Add `Bigraph.to_markdown` with labels, direction, style, and multiplicity; verify mixed edge/arrow summaries, dotted styles, and string labels.
- [x] 7.4 Add the deterministic SVG/HTML renderer and `_repr_html_` for `Bigraph`; verify labels, solid/dotted styles, arrow markers, multiplicity labels, and no IPython dependency.
- [x] 7.5 Add Markdown rendering and hooks for `DynkinType` and `DynkinClassification`, including status, inertia, component table, steps, bound, and normal-form summary; verify classified and non-finite outputs.
- [x] 7.6 Verify plain `repr` for `Bigraph`, `DynkinType`, `DynkinClassification`, and `FlationSequence`, and update the display reference documentation; verify the tests and `hatch run docs:build`.

## 8. Documentation, quality gates, and packaging

- [x] 8.1 Update `README.md` with one executable quickstart covering a mixed bigraph, flation, and `classify()`; verify `hatch test tests/test_readme.py` passes.
- [x] 8.2 Update `docs/getting-started/quickstart.md` and `CHANGELOG.md` with the general `QuasiCartanMatrix`, bigraph, classifier, and display surface; verify the documentation builds with warnings treated as errors.
- [x] 8.3 Run `hatch check --fix` twice and `hatch test`; verify the second check leaves the tree unchanged and the coverage gate passes.
- [x] 8.4 Build the wheel and install it in a clean environment; verify import, `py.typed`, `LICENSE`, and package metadata.
- [x] 8.5 Run `scripts/verify --skip-colab` and the full `scripts/verify`; verify the host gates and the pinned Colab smoke gate pass.
- [x] 8.6 Inspect the public surface and source imports; verify all specified names are exported and no underscore-prefixed private names are imported.

## Workflow follow-up

- Review the completed change against the paper's Stage 1/2/3 and the specs.
- Archive the change after review and a green full verification run.
- Create the Git commit for the archived change and name the archived change in the commit message.
- Defer A-block/D-core, affine classification, DOT/TikZ export, and witness animation to later changes.
