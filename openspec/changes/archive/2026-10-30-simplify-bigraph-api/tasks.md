# Tasks

## 1. Variadic path construction

- [x] 1.1 Change `Bigraph.from_paths` to `from_paths(*paths, labels=None, arrow_multiplicity=2)`, join path blocks with newlines, reject non-string arguments, and keep the single multiline-string form; verify with focused tests for multiple arguments, one multiline string, empty paths with pinned labels, keyword-only labels, non-string arguments, and line-numbered errors.
- [x] 1.2 Update `docs/reference/index.md` with the variadic signature, a multi-argument A9 example, and the keyword-only label note; verify the examples run in a focused Python check.
- [x] 1.3 Run `hatch check` and the focused bigraph tests; verify no positional-label call sites remain.

## 2. Label-aware matrix rendering

- [x] 2.1 Add `Bigraph.to_matrix_markdown` and `Bigraph.to_matrix_html`, forwarding `DisplayOptions`, bigraph labels, and HTML `collapse`; verify with tests for labeled Markdown, labeled HTML, non-contiguous D8 labels, collapse, per-call options, and an unchanged bigraph.
- [x] 2.2 Update `docs/reference/index.md` to distinguish graph `to_markdown`/`to_html` from `to_matrix_markdown`/`to_matrix_html` and document the label ownership; verify the new examples run in a focused Python check.
- [x] 2.3 Run `hatch check` and the focused display/bigraph tests; verify `__all__` and runtime dependencies are unchanged.

## 3. Notebook simplification

- [x] 3.1 Replace `A9_TEXT`, the triple-quoted B6 text, and embedded `\n` path blocks with separate `from_paths` arguments, keeping explicit labels where first appearance differs; verify the source parses and every code cell remains at most 40 lines.
- [x] 3.2 Replace direct `display(Markdown(value.to_markdown()))` calls with final-expression hooks or multi-object `display`, keep manual wrappers only for raw Markdown/HTML composites, and use `to_matrix_*` where labels were repeated; verify the notebook source parses.
- [x] 3.3 Add a notebook AST guard rejecting direct `display(Markdown(<call>.to_markdown()))` wrappers while allowing raw/composite Markdown; verify the guard fails on an injected direct wrapper and passes on the current source.
- [x] 3.4 Regenerate `notebooks/dynkin-masterclass.ipynb` with executed outputs; verify source/artifact sync, committed images and Markdown outputs, and clean-kernel execution.
- [x] 3.5 Update `notebooks/README.md` to document hook-first display and multi-argument path blocks; verify the guide commands still run.
- [x] 3.6 Run `hatch check` and `hatch test tests/test_notebook.py --no-cov`; verify the line budget, no-definition, no-coordinate-table, Markdown-before-code, LaTeX, and new display guards pass.

## 4. Changelog and integration

- [x] 4.1 Record the breaking `from_paths` signature change, the matrix-view methods, and the notebook simplification under `[Unreleased]` in `CHANGELOG.md`; verify the breaking note is explicit.
- [x] 4.2 Run `hatch check` and full `hatch test`; verify the coverage gate and all library and notebook tests pass.
- [x] 4.3 Run `hatch run docs:build` and `hatch run verify`; verify the strict docs build and Colab smoke check pass.
- [x] 4.4 Validate the change with `npx --yes @fission-ai/openspec validate "simplify-bigraph-api" --type change --strict`; verify it is valid.
- [x] 4.5 Inspect `src/dynkinlab/__init__.py`, `pyproject.toml`, and `git diff --stat`; verify `__all__`, runtime dependency floors, and distribution metadata are unchanged.

## Workflow follow-up

- Review the regenerated notebook before archiving.
- Sync the three delta specs into the main specs, then archive the change and commit the archive with the change name.
