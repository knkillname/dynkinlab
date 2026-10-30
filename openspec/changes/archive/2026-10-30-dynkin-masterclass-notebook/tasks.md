# Tasks

## 1. Path notation and faithful views

- [x] 1.1 Add `Bigraph.from_paths` and its line/token parser to `src/dynkinlab/bigraph.py`, including explicit labels, isolated vertices, comments, conflict detection, and `arrow_multiplicity`; verify with focused tests for A9, labels, conflicts, invalid text, and multiplicity 3.
- [x] 1.2 Add `QuasiCartanMatrix.has_faithful_bigraph` and make `Bigraph.edges` raise for non-faithful matrices; verify with tests for A2/A3, `2/4`, a unit arrow, and a symmetric dotted pair.
- [x] 1.3 Document the path grammar and faithful predicate in `docs/reference/index.md`; verify the new examples and the strict docs build.
- [x] 1.4 Run `hatch check` and the focused bigraph test files; verify no new top-level export or runtime dependency.

## 2. Labeled matrix tables

- [x] 2.1 Extend `QuasiCartanMatrix.to_markdown` and `to_html` with display-only labels and extend `to_html` with `collapse`; verify labels, non-contiguous D8 headers, escaping, invalid labels, and the expanded default.
- [x] 2.2 Document labeled and collapsed matrix rendering in `docs/reference/index.md`; verify the strict docs build.
- [x] 2.3 Run `hatch check` and the display tests; verify the existing index-base and `_repr_html_` behavior is unchanged.

## 3. NetworkX layout and drawing

- [x] 3.1 Add `DisplayOptions.bigraph_layout` and `bigraph_layout_iterations` with validation and reset coverage; verify defaults, invalid policy and iteration rejection, and per-call overrides.
- [x] 3.2 Refactor `src/dynkinlab/bigraph_display.py` into a shared layout/drawing engine and add `Bigraph.layout`; verify auto Kamada-Kawai for connected graphs, planar-seeded unweighted energy spring for planar K4 graphs such as A9, seeded energy spring with 10000 iterations for fallback and disconnected graphs, force fallback without SciPy, named layouts, unknown names, single-vertex handling, and repeatability.
- [x] 3.3 Add `Bigraph.draw` with highlights, display labels, shared positions, and returned axes; verify mixed styles, arrow direction, non-faithful rejection, and no IPython import.
- [x] 3.4 Add `BlockCutTree.draw` with a default NetworkX bipartite layout and an optional original-graph position mode that fixes articulation points and places blocks at centroids; verify A9 block and vertex nodes, articulation highlighting, centroid placement, and repeatability.
- [x] 3.5 Document the layout policy, `Bigraph.draw`, and `BlockCutTree.draw` in `docs/reference/index.md`; verify the strict docs build.
- [x] 3.6 Run `hatch check` and the drawing/display test files; verify the inline SVG renderer and the notebook figures share the same layout engine.

## 4. Certificate and witness summaries

- [x] 4.1 Add `ATypeCertificate.to_markdown` and its hook; verify the A9 block, partition, and articulation table.
- [x] 4.2 Add `DCoreGluingCertificate.to_markdown` and `ACyclizationCertificate.to_markdown` with hooks; verify the D13 piece table and the A9-to-D8 summary.
- [x] 4.3 Add `FlationSequence.to_markdown` and its hook, including sign-inversion labeling; verify sequence order and the plain `repr()`.
- [x] 4.4 Document certificate and witness rendering in `docs/reference/index.md`; verify the strict docs build.
- [x] 4.5 Run `hatch check` and the structure/display test files; verify every new hook returns the same string as its `to_markdown()`.

## 5. Masterclass notebook

- [x] 5.1 Create `notebooks/dynkin-masterclass.py` with the setup, matrix/form/inertia, and faithful-bigraph sections using only public API calls; verify Jupytext parses the source and the cells stay within 40 lines.
- [x] 5.2 Add the flation, finite-type, and A9 block-cut sections with fixed-tour checks and parameter cells; verify the source remains free of local function/class definitions.
- [x] 5.3 Add the D13 gluing, A9-to-D8 cyclization, boundary table, exercises, labs, and final certificate checks; verify every canonical certificate and table is produced without hardcoded coordinates.
- [x] 5.4 Generate `notebooks/dynkin-masterclass.ipynb` with executed outputs; verify the cell types/sources match the source and the figures/tables are present.
- [x] 5.5 Update `tests/test_notebook.py` for the new paths and add guards for the 40-line limit, no definitions, no coordinate tables, Markdown-before-code, and no thesis framing; verify the focused notebook tests pass.
- [x] 5.6 Remove `notebooks/thesis-companion.py` and `.ipynb`; verify no active code or test path still references them.
- [x] 5.7 Rewrite `notebooks/README.md` for the masterclass source, generation, Colab install, labs, and smoke test; verify every documented command runs.
- [x] 5.8 Update the quickstart notebook pointer to the masterclass; verify the strict docs build.

## 6. Changelog and release integration

- [x] 6.1 Record the masterclass, notebook-facing API, and `2/4` edge behavior change under `[Unreleased]` in `CHANGELOG.md`; verify the entries distinguish additions from the breaking behavior.
- [x] 6.2 Run `hatch check` and the full `hatch test`; verify the coverage gate and all notebook guards pass.
- [x] 6.3 Run `hatch run docs:build` and `hatch run verify`; verify the strict docs build, package tests, and Colab smoke check pass.
- [x] 6.4 Validate the change with `npx --yes @fission-ai/openspec validate "dynkin-masterclass-notebook" --type change --strict`; verify it is valid.
- [x] 6.5 Inspect `src/dynkinlab/__init__.py`, `pyproject.toml`, and `git diff --stat`; verify `__all__`, runtime dependency floors, and distribution metadata are unchanged apart from the intended modules.

## Workflow follow-up

- Review the generated notebook and its committed outputs before archiving.
- Archive the change after review, sync the new `dynkin-masterclass` capability, retire `thesis-notebook`, and commit the archive with the change name.
- Push to `origin/main` so Colab can install the new API and open the masterclass from the default branch.
