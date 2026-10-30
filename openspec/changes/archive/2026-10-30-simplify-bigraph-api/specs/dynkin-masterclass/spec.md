# Spec Delta

## ADDED Requirements

### Requirement: Hook-first notebook display

The notebook SHALL rely on value-object `_repr_markdown_` and `_repr_html_` hooks for single
displayed values. It SHALL use multi-object `display` only when a cell renders several values, and
SHALL reserve manual `display(Markdown(...))` or `display(HTML(...))` for raw Markdown/HTML
compositions such as captions, flex panels, and computed tables.

#### Scenario: Certificate renders directly

- **WHEN** an A-type, D-core gluing, or A-cyclization certificate is the last expression in a cell
- **THEN** Jupyter renders its Markdown hook without a `display(Markdown(...))` wrapper

#### Scenario: Several values share a cell

- **WHEN** a cell displays a form and its inertia, or a classification and its witness
- **THEN** the notebook uses multi-object `display` and both objects render through their hooks

#### Scenario: Raw tables keep an explicit wrapper

- **WHEN** a cell displays the computed boundary table or another raw Markdown string
- **THEN** the notebook wraps that string in `display(Markdown(...))`

#### Scenario: Manual wrappers do not bypass hooks

- **WHEN** a single certificate, classification, form, inertia, or flation sequence is displayed
- **THEN** the notebook does not call `to_markdown()` only to wrap the result in `Markdown`

### Requirement: Multi-argument path blocks

The notebook SHALL pass each path-notation block as a separate `Bigraph.from_paths` argument. It
SHALL NOT encode multi-line path blocks as strings containing `\n` or keep a module-level path-text
constant solely to pass to `from_paths`.

#### Scenario: A9 path is split into blocks

- **WHEN** the A9 example is constructed
- **THEN** each path block is a separate `from_paths` argument

#### Scenario: D13 and B6 paths are split into blocks

- **WHEN** the D13 core, its large A-type piece, or the mixed B6 example are constructed
- **THEN** their path blocks are separate arguments rather than one escaped string

#### Scenario: Labels remain explicit where needed

- **WHEN** first appearance does not match the intended matrix order
- **THEN** the path call keeps a keyword `labels` argument
