# Proposal

## Why

`Bigraph.from_paths` still requires callers to encode multi-line path blocks as one escaped string,
and the notebook still repeats `labels=bigraph.labels` whenever it renders a matrix. It also wraps
certificate and classification values in `display(Markdown(value.to_markdown()))` even though every
one of those values already provides a notebook hook. These are avoidable layers between the
reader and the mathematics.

## What Changes

- **BREAKING:** Change `Bigraph.from_paths` from one text block to `*paths`, where each argument is
  a path-notation block and the blocks are joined with newlines. `labels` and
  `arrow_multiplicity` become keyword-only. The single multiline-string form continues to work;
  positional labels no longer do.
- Add `Bigraph.to_matrix_markdown(options=None)` and
  `Bigraph.to_matrix_html(options=None, *, collapse=False)` so a bigraph renders its underlying
  matrix with its own labels. These forward to the existing matrix renderers.
- Update the masterclass notebook to:
  - pass one argument per path block instead of embedding `\n` in strings;
  - use `_repr_markdown_`/`_repr_html_` hooks for single values;
  - use multi-object `display` for several values in one cell;
  - keep manual `display(Markdown(...))`/`display(HTML(...))` only for raw Markdown/HTML
    compositions such as captions, flex panels, and computed tables.
- Add a notebook structural guard against direct `display(Markdown(value.to_markdown()))` wrappers.
- Update the API reference, tests, and changelog. No new runtime dependency and no new `__all__`
  name.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `bigraph-flation`: changes path construction to variadic path blocks, with keyword-only labels and
  arrow multiplicity.
- `colab-display`: adds label-aware matrix rendering on `Bigraph`.
- `dynkin-masterclass`: adds hook-first notebook display and multi-argument path-block conventions.

## Impact

- `src/dynkinlab/bigraph.py`: parser entry point, argument joining, matrix-rendering methods.
- `notebooks/dynkin-masterclass.py` and generated `.ipynb`: path blocks and display calls.
- `tests/test_bigraph.py`, display tests, and `tests/test_notebook.py`: signature, rendering, and
  style guards.
- `docs/reference/index.md`, `notebooks/README.md` if needed, and `CHANGELOG.md`.
- `openspec/specs/bigraph-flation/spec.md`, `openspec/specs/colab-display/spec.md`, and
  `openspec/specs/dynkin-masterclass/spec.md`.
- The only breaking behavior is the `from_paths` positional signature; existing `labels=` calls
  keep working. No dependency or package-surface change.
