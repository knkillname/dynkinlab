# Spec Delta

## Purpose

Defines rich Markdown and HTML representations for DynkinLab value objects in Colab and Jupyter, with a mutable canonical options object and per-call overrides.

## ADDED Requirements

### Requirement: Canonical display options

The system SHALL expose a mutable `dynkinlab.display.options` object with `expanded_form_dimension_limit` and `matrix_index_base` settings, and the object SHALL provide `reset()` to restore the field defaults.

#### Scenario: Canonical options affect rendering

- **WHEN** `dynkinlab.display.options.matrix_index_base` is set to 1 and a matrix is rendered without a per-call options object
- **THEN** the rendered row and column headers start at 1

#### Scenario: Reset restores defaults

- **WHEN** `dynkinlab.display.options.reset()` is called after either setting changes
- **THEN** both settings return to their field defaults

### Requirement: Per-call display overrides

The system SHALL accept an optional `DisplayOptions` object on rendering methods, and its values MUST take precedence over the canonical object for that call only.

#### Scenario: Per-call override wins

- **WHEN** the canonical `matrix_index_base` is 1 and a call passes `DisplayOptions(matrix_index_base=0)`
- **THEN** that call renders zero-based headers and the canonical object remains unchanged

### Requirement: Matrix Markdown representation

`QuasiCartanMatrix` SHALL provide `to_markdown(options=None)` returning a Markdown table whose row and column headers use the configured index base and whose cells contain the matrix entries.

#### Scenario: Zero-based Markdown table

- **WHEN** the `A_2` matrix is rendered with index base 0
- **THEN** the Markdown contains headers `0` and `1` and the entries `2`, `-1`, `-1`, and `2`

#### Scenario: One-based Markdown table

- **WHEN** the `A_2` matrix is rendered with index base 1
- **THEN** the Markdown contains headers `1` and `2`

### Requirement: Matrix HTML representation

`QuasiCartanMatrix` SHALL provide `to_html(options=None)` returning an HTML table whose row and column headers use the configured index base and whose cells contain the matrix entries.

#### Scenario: HTML table

- **WHEN** the `A_2` matrix is rendered as HTML
- **THEN** the result is an HTML table with row and column headers and the matrix entries

### Requirement: Quadratic form Markdown representation

`QuadraticForm` SHALL provide `to_markdown(options=None)` returning a Markdown block with the general form and, when the form dimension is at or below the configured expansion limit, the expanded formula.

#### Scenario: Small form is expanded

- **WHEN** the `A_2` form is rendered with an expansion limit of 3
- **THEN** the Markdown includes the expanded terms for `x_0` and `x_1`

#### Scenario: Large form stays general

- **WHEN** a form has more variables than the configured expansion limit
- **THEN** the Markdown shows the general summation formula instead of expanded terms

### Requirement: Inertia Markdown representation

`Inertia` SHALL provide `to_markdown()` returning a Markdown summary with positive, negative, and zero counts, rank, signature, and a definiteness or degeneracy label.

#### Scenario: Positive definite inertia

- **WHEN** `Inertia(2, 0, 0)` is rendered
- **THEN** the Markdown shows the counts, rank 2, signature 2, and `positive definite`

#### Scenario: Degenerate inertia

- **WHEN** `Inertia(2, 0, 1)` is rendered
- **THEN** the Markdown shows a zero count of 1 and identifies the result as degenerate

### Requirement: Display hooks

`QuasiCartanMatrix` SHALL provide `_repr_html_`, while `QuadraticForm` and `Inertia` SHALL provide `_repr_markdown_`. The hooks MUST return plain strings, MUST require no IPython dependency, and MUST use the canonical display options.

#### Scenario: Matrix HTML hook

- **WHEN** a notebook displays a matrix object
- **THEN** `_repr_html_` returns the same HTML as `to_html()` with the canonical options

#### Scenario: Form and inertia Markdown hooks

- **WHEN** a notebook displays a form or inertia object
- **THEN** `_repr_markdown_` returns the same Markdown as `to_markdown()` with the canonical options

### Requirement: Debug representation remains unchanged

`repr()` on the value objects SHALL remain the dataclass-style representation and SHALL NOT return Markdown or HTML.

#### Scenario: Repr stays plain

- **WHEN** `repr` is called on a matrix, form, or inertia value
- **THEN** the result is dataclass-style text containing no HTML tags or Markdown table syntax
