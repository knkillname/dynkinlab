# Spec Delta

## ADDED Requirements

### Requirement: Label-aware matrix view

`Bigraph.to_matrix_markdown(options=None)` and
`Bigraph.to_matrix_html(options=None, *, collapse=False)` SHALL render the underlying matrix with
the bigraph's labels. They SHALL forward display options and HTML collapsing to the existing matrix
renderers and SHALL NOT change the bigraph.

#### Scenario: Markdown view uses bigraph labels

- **WHEN** a bigraph with non-contiguous labels such as D8 labels `(1, 2, 3, 4, 5, 7, 8, 9)` renders
  its matrix as Markdown
- **THEN** the table headers are exactly those labels in that order

#### Scenario: HTML view uses bigraph labels

- **WHEN** a labeled bigraph renders its matrix as HTML
- **THEN** the table headers use the bigraph labels rather than matrix indices

#### Scenario: Collapsed HTML is forwarded

- **WHEN** a bigraph renders its matrix as HTML with `collapse=True`
- **THEN** the result contains the collapsed disclosure around the labeled table

#### Scenario: Per-call options are forwarded

- **WHEN** a bigraph renders its matrix with an explicit `DisplayOptions` object
- **THEN** that object controls the matrix rendering without changing the canonical options

#### Scenario: The bigraph is unchanged

- **WHEN** a bigraph renders its matrix as Markdown or HTML
- **THEN** the bigraph labels, edges, and underlying matrix are unchanged
