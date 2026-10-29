# Tasks

## 1. Notebook foundation and local helpers

- [ ] 1.1 Create `notebooks/thesis-companion.py` with `# %%` code-cell and `# %% [markdown]`
  Markdown-cell markers, title/abstract, the seven section headings, and a setup cell that installs
  the latest DynkinLab from `main` in Colab (with no commit, tag, or version pin), imports the
  library, and sets the display defaults; generate `notebooks/thesis-companion.ipynb` with
  Jupytext. Verify the source parses into cells, the generated notebook exists, every heading is
  present, and the setup cell has no pin constant or IPython magic.
- [ ] 1.2 Implement the local `graph(text, labels=None)` parser with the six edge operators,
  comments, first-appearance or explicit labels, and conflict rejection; verify a source cell
  reconstructs the A9, D13, and mixed B6 matrices and compares them to the validated entries in
  `design.md`.
- [ ] 1.3 Implement the local labeled matrix display helper with a horizontally scrollable HTML
  table and a collapsed D13 form; verify the D8 header is `(1, 2, 3, 4, 5, 7, 8, 9)` and the D13
  matrix remains present without breaking the page.
- [ ] 1.4 Implement `has_faithful_bigraph(matrix)` and the Markdown explanation of the `2 x 2`
  minor and drawable range; verify A9, D13, and D8 are drawable and the `2/4` matrix is not.
- [ ] 1.5 Implement the local drawing helper for undirected and directed edges, solid and dotted
  styles, multiplicity labels, highlights, and fixed positions for A3, A9, and D8; verify the three
  figures render with identical placement on repeated execution.
- [ ] 1.6 Add the D13 fixed positions, core-edge highlight, and deterministic block-cut tree
  drawing; verify the D13 figure highlights all five core edges and the A9 tree labels its four
  blocks and articulation points.

## 2. Matrix-first worked examples

- [ ] 2.1 Write section 2 with the A2/A3 labeled matrices, the drawable-range table, and the `2/4`
  matrix-only case; verify the inertia and classifier assertions and that the table distinguishes
  matrix and bigraph cases.
- [ ] 2.2 Write section 3 with the A3 flation before/after matrices and figures; verify inertia and
  the classified type remain A3.
- [ ] 2.3 Write section 4 with the A9 matrix, block-cut tree figure, and A-type certificate; verify
  the certificate and the exact blocks and articulation points.
- [ ] 2.4 Write section 5 with the D13 core, five-piece table, gluing certificate, and highlighted
  D13 figure; verify the certificate, the D13 type, and the piece types.
- [ ] 2.5 Write section 6 with the A9-to-D8 cyclization, highlighted path figure, D8 matrix and
  figure, and certificate; verify the path `(1, 3, 5, 6)`, the D8 type, and the `1=6` label.
- [ ] 2.6 Write section 7 with the thesis algorithm sketch, the bounded-flation oracle caveat, the
  generated boundary table, and the final assertions; verify every table status and certificate
  label and the final "all certificates verified" output.
- [ ] 2.7 Regenerate `notebooks/thesis-companion.ipynb` from the Python source, execute it with the
  `nbclient` client, and write the executed notebook back to disk; verify the committed figures and
  tables are present and the final cell output reports the package version and the verification
  message.

## 3. Notebook smoke execution

- [ ] 3.1 Add `jupytext>=1.16`, `nbclient>=0.10`, and `nbformat>=5.10` to the `hatch-test` extra
  dependencies; verify the test environment imports all three and the `python3` kernel spec is
  available.
- [ ] 3.2 Add `tests/test_notebook.py` that reads `notebooks/thesis-companion.py` with the
  percent-format reader, compares its cell types and normalized sources with the generated
  notebook, executes the source-derived notebook with `NotebookClient`, and uses the notebook
  directory as the working directory; verify `hatch test tests/test_notebook.py --no-cov` passes.
- [ ] 3.3 Verify the smoke test fails for a temporary source copy with a raising cell, then remove
  the copy; verify `hatch check` passes, adding only a minimal import ignore if Pyrefly cannot type
  the third-party notebook libraries.

## 4. Documentation and changelog

- [ ] 4.1 Rewrite `notebooks/README.md` with the `.py` source workflow, Jupytext regeneration, latest
  git install, Colab workflow, local smoke test, and the paper-companion scope; verify no PyPI
  install instruction or version pin remains in the notebook guide.
- [ ] 4.2 Update `docs/getting-started/quickstart.md` to point at the companion notebook and
  `docs/reference/index.md` to remove the notebook from the planned surface; verify
  `hatch run docs:build` passes.
- [ ] 4.3 Add the notebook to `CHANGELOG.md` under `[Unreleased]`; verify the entry names the
  notebook and the latest-version Colab workflow.

## 5. Integration verification

- [ ] 5.1 Run `hatch check` and verify it passes.
- [ ] 5.2 Run `hatch test` and verify the full suite passes, including the notebook smoke test and
  the 80% coverage gate.
- [ ] 5.3 Run `hatch run docs:build` and verify it passes with no warnings.
- [ ] 5.4 Run `hatch run verify` and verify it passes.
- [ ] 5.5 Run `openspec validate "thesis-companion-notebook" --strict` and verify the change is
  valid.
- [ ] 5.6 Verify `git diff -- src/dynkinlab` is empty and `hatch test tests/test_package.py --no-cov`
  passes, confirming no public-surface change.

## Workflow follow-up

- Push the implemented `.py` and `.ipynb` commit so the Colab notebook can open from `main`.
- Open the notebook in Colab, run all cells, and verify the latest git install, figures, and final
  verification cell.
- Archive the change after review.
