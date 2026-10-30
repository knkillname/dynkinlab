# Design

## Context

See `proposal.md` for motivation. The current `Bigraph.from_paths` accepts one `text` string and
joins its `splitlines()` internally. The notebook works around that with `A9_TEXT`, triple-quoted
B6 text, and embedded `\n` in the large A7 piece. The current `Bigraph` owns labels and a matrix, but
matrix rendering has to go through `bigraph.matrix.to_markdown(labels=bigraph.labels)`. Certificate,
classification, form, inertia, and flation values already expose `_repr_markdown_`; the notebook
wraps many of them in `display(Markdown(value.to_markdown()))` anyway.

## Goals / Non-Goals

**Goals:**

- Make multi-block path construction read like the notation itself, without escaped newlines.
- Remove repeated label plumbing when a bigraph renders its matrix.
- Make the notebook demonstrate the public notebook hooks rather than bypassing them.
- Keep the package surface and runtime dependencies unchanged.
- Mechanically guard the notebook against reintroducing redundant Markdown wrappers.

**Non-Goals:**

- A new `LabeledMatrix` or `MatrixView` value type.
- A general Markdown table builder or caption system.
- Per-edge arrow multiplicities or a richer path grammar.
- Changing the graph rendering or NetworkX layout policy.
- Supporting positional `labels` through a compatibility shim.

## Decisions

### 1. Variadic path blocks with keyword-only options

`Bigraph.from_paths` becomes:

```python
@classmethod
def from_paths(
    cls,
    *paths: str,
    labels: Iterable[Hashable] | None = None,
    arrow_multiplicity: int = 2,
) -> Bigraph:
```

The parser receives `"\n".join(paths)`. This preserves the existing line-numbered diagnostics and
keeps a single multiline string working. A non-string argument raises `ValueError` before parsing.
Zero path arguments with pinned labels produce an edgeless labeled graph, matching today's
`from_paths("", labels=...)`.

`labels` and `arrow_multiplicity` become keyword-only because the positional slots now belong to
path blocks. The only breaking call is positional labels, which no repository caller uses. A
compatibility shim that detects a non-string final argument and reinterprets it as labels would make
a typo silently change meaning, so it is rejected.

Alternative considered: add a separate `from_path_lines(*lines)` while keeping `from_paths(text)`.
That preserves the old signature but creates two constructors with almost identical semantics.

### 2. Label-aware matrix rendering on `Bigraph`

Add two forwarding methods:

```python
Bigraph.to_matrix_markdown(options=None) -> str
Bigraph.to_matrix_html(options=None, *, collapse=False) -> str
```

They call `self.matrix.to_markdown(options=options, labels=self.labels)` and
`self.matrix.to_html(options=options, labels=self.labels, collapse=collapse)`. Display options,
label validation, escaping, and collapsing stay in the matrix renderer. The bigraph is not mutated.

This removes eight `labels=bigraph.labels` repetitions from the masterclass and makes the label
ownership explicit: a `Bigraph` is the value that knows both the matrix and its vertex labels.

Alternative considered: a public `LabeledMatrix`/`MatrixView` with its own notebook hooks. Rejected
because it duplicates the label ownership already in `Bigraph`, adds a new public type and likely a
new `__all__` entry, and reverses the prior design decision to avoid a labeled-matrix value. The
string-returning methods fit the existing `to_*` convention and require no new export.

Alternative considered: add `caption` to `to_matrix_markdown`/`to_matrix_html`. Captions are
presentation composition; the notebook can keep its small caption strings. This can be revisited if
captioning spreads beyond the masterclass.

### 3. Hook-first notebook display

The notebook keeps the value as the final expression when a cell renders one hook-backed value:

```python
A9_CERTIFICATE = A9.a_type_certificate()
assert A9_CERTIFICATE is not None
assert A9_CERTIFICATE.verify(A9)
A9_CERTIFICATE
```

For several values, it uses IPython's multi-object `display`:

```python
display(E6_RESULT, E6_RESULT.witness)
display(bad.inertia, bad.classify())
```

Manual `display(Markdown(...))` remains only when the argument is a raw Markdown string or a
composite such as a caption plus table. Manual `display(HTML(...))` remains for flex panels and
matrix tables. This keeps the rendered result identical while exercising the public hooks.

### 4. Multi-argument path blocks in the notebook

The masterclass passes each path block separately and removes `A9_TEXT` and triple-quoted B6 text.
The large A7 piece keeps its explicit labels because first appearance is not numerical order. This
is a readability change; the resulting matrices and certificates are unchanged.

### 5. Test strategy

Library tests:

- `from_paths("1--2", "2--3")` equals `from_paths("1--2\n2--3")`.
- A single multiline string still parses.
- `from_paths(labels=(1, 2))` creates an edgeless labeled graph.
- Positional labels and non-string path arguments raise `ValueError`.
- Line-numbered errors still identify the joined line.
- `to_matrix_markdown`/`to_matrix_html` use the bigraph labels, including non-contiguous D8
  labels, forward options and `collapse`, and leave the bigraph unchanged.

Notebook tests:

- The existing source/artifact sync, execution, 40-line cap, no-definition, no-coordinate-table,
  Markdown-before-code, and LaTeX guards remain.
- A new AST guard rejects direct `display(Markdown(<call>.to_markdown()))` wrappers. It does not
  reject `display(Markdown("..." + <call>.to_markdown()))` because that is a raw composite.
- Add a source-level check that path-construction calls do not contain literal `\n`? This is harder
  to distinguish from other strings. The code review and the multi-argument examples are the guard;
  avoid a brittle string check.

## Risks / Trade-offs

- **Breaking `from_paths` positional labels** → Document in the changelog and API reference; no
  repository caller uses positional labels; keyword labels remain unchanged.
- **Global line numbers across joined path arguments** → Keep the existing line-number diagnostics;
  users can still map them to the joined block. A path-index diagnostic can be added later without
  changing the accepted grammar.
- **Hook display depends on expression position** → Put asserts before the final value; the
  notebook smoke test executes the artifact and the committed outputs are regenerated.
- **The AST guard could block a legitimate future wrapper** → Match only the direct
  `Markdown(call.to_markdown())` pattern, not BinOps or raw strings. If a future exception is
  needed, document it rather than weakening the guard silently.
- **`to_matrix_*` names could be confused with graph `to_*`** → Document that `to_markdown`/`to_html`
  render the bigraph and `to_matrix_*` render its matrix.

## Migration Plan

1. Change `from_paths` and add the matrix-view methods with tests.
2. Update the notebook paths and display cells; regenerate the `.ipynb`.
3. Add the notebook AST guard and update the API reference and changelog.
4. Run `hatch check`, `hatch test`, `hatch run docs:build`, and `hatch run verify`.
5. Sync the three capability specs and archive the change. Rollback is a normal revert.
