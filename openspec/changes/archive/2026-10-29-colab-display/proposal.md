# Proposal

## Why

DynkinLab is Colab-first, but its value objects currently display only as dataclass `repr` strings. In a notebook, a quasi-Cartan matrix should render as a readable table, a quadratic form as a formula, and inertia as a compact result summary. Users also need a way to choose one-based matrix indices or change how eagerly a formula is expanded without editing the library.

## What Changes

- Add a `dynkinlab.display` module with a mutable `DisplayOptions` object and a canonical `dynkinlab.display.options` instance.
- Keep display defaults on the `DisplayOptions` fields themselves; do not add separate public `DEFAULT_*` constants.
- Add public `to_markdown()` and `to_html()` methods where each format is useful.
- Add Jupyter/Colab display hooks: `_repr_html_` for matrices and `_repr_markdown_` for forms and inertia.
- Render matrices as Markdown and HTML tables, forms as Markdown formulas, and inertia as a Markdown summary with counts, rank, signature, and definiteness.
- Let a per-call `DisplayOptions` object override the canonical instance. Precedence is per-call options, then `dynkinlab.display.options`, then the field defaults.
- Keep the dataclass `repr` unchanged for terminals, logs, and debugging.
- Test the public methods, hooks, option precedence, matrix index bases, formula expansion, and test isolation.
- Document the display API and the Colab behavior.

Out of scope: `Bigraph` visualization, changes to the Colab runtime gate, IPython as a dependency, and changes to `repr`.

## Capabilities

### New Capabilities

- `colab-display`: Markdown and HTML representations for DynkinLab value objects, plus notebook-wide and per-call display options.

### Modified Capabilities

- None.

## Impact

- New `src/dynkinlab/display.py` module and public `dynkinlab.display` namespace.
- New display methods and hooks on `QuasiCartanMatrix`, `QuadraticForm`, and `Inertia`.
- New tests for rendering, option precedence, and display state isolation.
- Documentation updates in `README.md`, `docs/reference/index.md`, and `docs/getting-started/quickstart.md`.
- No new runtime dependencies and no changes to package metadata, CI, or the pinned Colab gate.
