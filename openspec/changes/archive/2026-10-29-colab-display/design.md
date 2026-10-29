# Design

## Context

See `proposal.md` for motivation and `specs/colab-display/spec.md` for the behavior contract. DynkinLab already has three public value objects with stable dataclass `repr` output, a Colab-first documentation stance, and a pinned Colab gate. The display layer is additive: it must not change mathematical behavior, hashing, equality, or debug output.

## Goals / Non-Goals

**Goals:**

- Rich, automatic Colab rendering for the current public value objects.
- One notebook-wide display configuration plus explicit per-call overrides.
- Public string-returning conversion methods with no IPython dependency.
- A pattern that `Bigraph` can extend later.

**Non-Goals:**

- Implementing `Bigraph` visualization or any new mathematical object.
- Changing the pinned Colab runtime or the existing `colab-compatibility` gate.
- Changing `repr`, `str`, equality, hashing, or validation of the value objects.
- Adding a global constants module or a strict constructor singleton.
- Rendering Markdown or HTML in terminals or log files.

## Decisions

### Canonical mutable options, no public constants

The configuration lives in one mutable object:

```python
# dynkinlab/display.py
@dataclass
class DisplayOptions:
    expanded_form_dimension_limit: int = 3
    matrix_index_base: int = 0

    def reset(self) -> None: ...


options = DisplayOptions()
```

`dynkinlab.display.options` is the canonical instance. The field defaults are the only defaults; there are no separate public `DEFAULT_*` constants. This keeps the API to one concept and lets users read or change the effective configuration in one place.

Alternative considered: public module constants plus setters. Rejected because module rebinding does not reliably propagate to imported names, and setters add global state with more ceremony.

`DisplayOptions` validates assignments so a negative limit or index base cannot silently corrupt rendering. `reset()` restores the field defaults by constructing a fresh `DisplayOptions` and copying its values; this avoids duplicating the defaults in reset logic.

### Per-call overrides

Every rendering method accepts `options: DisplayOptions | None = None`. Resolution is:

```text
explicit options  >  dynkinlab.display.options  >  DisplayOptions field defaults
```

Passing an options object affects only that call. To inherit the canonical settings while changing one field, callers can use `dataclasses.replace(dynkinlab.display.options, ...)`.

Alternative considered: only per-call options. Rejected because it would leave `_repr_html_` and `_repr_markdown_` unable to see notebook-wide customization.

Alternative considered: strict singleton. Rejected because it makes isolated tests and temporary configurations awkward. A module-level canonical instance behaves like a singleton in normal use while still allowing independent `DisplayOptions` values.

### Per-object formats

```text
QuasiCartanMatrix   to_markdown()  to_html()   _repr_html_
QuadraticForm       to_markdown()              _repr_markdown_
Inertia             to_markdown()              _repr_markdown_
```

- The matrix uses HTML for automatic display because a semantic table with row and column headers is clearer than Markdown for numeric alignment. It also exposes Markdown for notebooks, docs, and plain-text exports.
- The quadratic form uses Markdown because formulas and LaTeX render naturally in Colab.
- Inertia uses Markdown because it is a compact summary, not a table-heavy object.

`to_markdown()` and `to_html()` return plain strings. The hooks delegate to them using the canonical options. No IPython import is required.

Alternative considered: Markdown for every object. Rejected for the matrix because HTML gives explicit headers and a stable styling hook. Alternative considered: HTML for every object. Rejected because Markdown is the natural format for formulas.

### Matrix rendering

The matrix table uses the configured index base for both row and column headers. Entries remain integers in the cells. Markdown uses an aligned table; HTML uses `<table class="dynkinlab-matrix">` with `<thead>`, `<tbody>`, `<th>`, and `<td>` elements.

The table renderer is a private helper on `QuasiCartanMatrix` so Markdown and HTML share header calculation and entry formatting. No module-level formatter functions are introduced.

### Quadratic form rendering

`QuadraticForm.to_markdown()` shows the general formula:

```text
q(x) = sum_i x_i^2 + sum_{i<j} a_ij x_i x_j
```

When the form dimension is at or below `expanded_form_dimension_limit`, it also shows the expanded polynomial in `x_0`, `x_1`, and so on. Off-diagonal coefficients of zero are omitted; coefficients of `1` and `-1` are rendered without an explicit numeric factor. Above the limit, only the general formula is shown.

The limit is read from `DisplayOptions`; there is no bare literal in the comparison logic. The field default of 3 is the named setting, not an unnamed magic number.

### Inertia rendering

`Inertia.to_markdown()` renders a compact table with positive, negative, zero, rank, and signature counts, followed by one label:

```text
positive definite / negative definite / degenerate / indefinite
```

The label is derived from the existing `Inertia` properties, so the display layer does not duplicate the definiteness logic.

### Test isolation

Tests never mutate module constants. They either pass an explicit `DisplayOptions` object or use `monkeypatch` to replace or patch `dynkinlab.display.options`. `DisplayOptions.reset()` is tested directly and is available to notebook users. This keeps global display state from leaking between tests.

### Module layout

```text
src/dynkinlab/display.py          DisplayOptions, canonical options
src/dynkinlab/quasi_cartan.py     to_markdown, to_html, _repr_html_
src/dynkinlab/quadratic_form.py   to_markdown, _repr_markdown_
src/dynkinlab/inertia.py          to_markdown, _repr_markdown_
src/dynkinlab/__init__.py         expose the display submodule
tests/test_display.py             options, rendering, hooks, isolation
```

## Risks / Trade-offs

- **Global mutable state can leak between notebooks or tests** -> Provide `reset()`, document it, and use `monkeypatch` in tests.
- **Formula expansion becomes long for larger matrices** -> Keep the limit configurable and default it to a small dimension.
- **HTML rendering can inject future string content** -> Current cells are integers; future labeled objects must escape text before adding it to HTML.
- **Display code drifts from the mathematical objects** -> Keep rendering read-only and test hooks against the public conversion methods.
- **Two formats increase surface area** -> Only implement HTML where it adds value; form and inertia stay Markdown-only.
- **Users may expect `repr` to change** -> Keep `repr` deliberately plain and document that rich output belongs to notebook display hooks.

## Migration Plan

This is additive. Existing imports, equality, hashing, `repr`, and mathematical behavior remain unchanged. If reverted, remove `display.py`, the new methods and hooks, `tests/test_display.py`, and the documentation additions.

## Open Questions

None that affect the spec, approach, or task breakdown.
