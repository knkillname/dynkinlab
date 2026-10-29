# Spec Delta

## ADDED Requirements

### Requirement: Non-symmetric quasi-Cartan matrix display

`QuasiCartanMatrix` SHALL render Markdown and HTML tables and provide `_repr_html_` for non-symmetric matrices using the configured index base. The table SHALL show the original entries `A`, not the symmetrized entries `S`.

#### Scenario: Non-symmetric matrix renders original entries

- **WHEN** `((2, -1), (-2, 2))` is rendered as Markdown with index base 0
- **THEN** the table contains headers 0 and 1 and the original entries 2, -1, -2, and 2

#### Scenario: HTML table and hook are available

- **WHEN** the same matrix is rendered as HTML or displayed in a notebook
- **THEN** the result is an HTML table with row and column headers and the original entries

#### Scenario: Canonical index base is used by the hook

- **WHEN** the canonical `matrix_index_base` is 1 and the matrix is displayed in a notebook
- **THEN** the row and column headers start at 1

### Requirement: Bigraph Markdown representation

`Bigraph` SHALL provide `to_markdown(options=None)` returning a Markdown summary of vertex labels and edges or arrows, including direction, style, and multiplicity. The canonical display options SHALL be used when no options are supplied.

#### Scenario: Mixed edges and arrows are summarized

- **WHEN** a bigraph with labels `("a", "b")` over `((2, -1), (-2, 2))` is rendered as Markdown
- **THEN** the summary identifies a solid arrow from `"a"` to `"b"` with multiplicity 2

#### Scenario: Dotted edges are distinguished

- **WHEN** a bigraph contains a positive off-diagonal pair
- **THEN** the Markdown summary marks that edge or arrow as dotted

#### Scenario: Labels appear in the summary

- **WHEN** a bigraph uses string labels
- **THEN** the Markdown summary uses those labels rather than matrix indices

### Requirement: Bigraph HTML representation and hook

`Bigraph` SHALL provide `to_html(options=None)` and `_repr_html_` returning a self-contained inline SVG or HTML rendering with labeled vertices, solid and dotted edges, arrows, and multiplicity labels. The rendering SHALL require no IPython dependency and SHALL use canonical display options.

#### Scenario: Rendered bigraph carries its visual structure

- **WHEN** a bigraph with a solid arrow of multiplicity 2 is rendered as HTML
- **THEN** the result contains the vertex labels, an arrow marker, and the multiplicity label 2

#### Scenario: Dotted and solid styles both render

- **WHEN** a bigraph contains one solid and one dotted edge
- **THEN** the rendered markup distinguishes the two styles

#### Scenario: Hook matches the canonical rendering

- **WHEN** a notebook displays a bigraph object
- **THEN** `_repr_html_` returns the same HTML as `to_html()` with the canonical options

### Requirement: Dynkin type and classification display

`DynkinType` and `DynkinClassification` SHALL provide Markdown rendering and notebook hooks. Classification rendering SHALL include status, exact inertia, component labels and types, step count, applied bound, and a normal-form summary.

#### Scenario: Dynkin type renders its name

- **WHEN** `A_3` is rendered as Markdown or displayed in a notebook
- **THEN** the output identifies the type as `A_3`

#### Scenario: Classified result renders its certificate summary

- **WHEN** a classified `A_2` result is rendered
- **THEN** the output includes the classified status, the `A_2` component, its inertia, and its step count

#### Scenario: Non-finite result renders its diagnostics

- **WHEN** a result with status `NOT_POSITIVE_DEFINITE` is rendered
- **THEN** the output includes the status, the exact inertia, and the partial step count

### Requirement: Bigraph display options

`DisplayOptions` SHALL add bigraph layout seed, figure size, label visibility, multiplicity visibility, and arrow style settings. Assignment validation and `reset()` SHALL cover the new fields, and per-call options SHALL override canonical options.

#### Scenario: New fields have usable defaults

- **WHEN** a fresh `DisplayOptions` is created
- **THEN** the bigraph layout seed, figure size, label visibility, multiplicity visibility, and arrow style have documented defaults

#### Scenario: New fields validate assignments

- **WHEN** a negative layout seed, a non-positive figure size, or an unknown arrow style is assigned
- **THEN** assignment fails with `ValueError`

#### Scenario: Reset restores bigraph defaults

- **WHEN** `reset()` is called after changing any bigraph display setting
- **THEN** every bigraph setting returns to its default

#### Scenario: Per-call bigraph override wins

- **WHEN** the canonical options hide multiplicity labels and a call passes options showing them
- **THEN** that call shows multiplicity labels and the canonical object remains unchanged

### Requirement: New values keep plain debug representation

`repr()` on `Bigraph`, `DynkinType`, `DynkinClassification`, and `FlationSequence` SHALL remain dataclass-style text and SHALL NOT return HTML tags or Markdown table syntax.

#### Scenario: Repr stays plain for new values

- **WHEN** `repr` is called on a bigraph, Dynkin type, classification result, or flation sequence
- **THEN** the result contains no HTML tags or Markdown table syntax
