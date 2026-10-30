# colab-display Specification

## Purpose

Defines rich Markdown and HTML representations for DynkinLab value objects in Colab and Jupyter, with a mutable canonical options object and per-call overrides.

## Requirements

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

### Requirement: Labeled matrix tables

`QuasiCartanMatrix.to_markdown` and `to_html` SHALL accept an optional display-only label sequence.
Supplied labels SHALL be unique, hashable, and match the matrix dimension; headers SHALL use them
instead of the index base. Invalid labels SHALL raise `ValueError`.

#### Scenario: HTML table uses labels

- **WHEN** the `A_2` matrix is rendered as HTML with labels `("a", "b")`
- **THEN** the header row and header column use `"a"` and `"b"` and the entries are unchanged

#### Scenario: Markdown table uses labels

- **WHEN** the `A_2` matrix is rendered as Markdown with labels `("a", "b")`
- **THEN** the Markdown table uses `"a"` and `"b"` as row and column headers

#### Scenario: Non-contiguous labels are preserved

- **WHEN** the D8 matrix is rendered with labels `(1, 2, 3, 4, 5, 7, 8, 9)`
- **THEN** the headers show exactly those labels in that order

#### Scenario: Invalid labels are rejected

- **WHEN** labels have the wrong length, duplicates, or an unhashable entry
- **THEN** rendering fails with `ValueError`

#### Scenario: Labels are escaped

- **WHEN** a label contains HTML or Markdown table characters
- **THEN** the rendered cell escapes them without changing the label text

### Requirement: Collapsed wide matrix HTML

`QuasiCartanMatrix.to_html` SHALL accept an optional `collapse` flag that wraps the table in a
collapsed HTML disclosure. The default SHALL render the table expanded, and collapsing SHALL NOT
change matrix entries or labels.

#### Scenario: Wide matrix collapses

- **WHEN** a matrix is rendered with `collapse=True`
- **THEN** the HTML contains a disclosure whose summary names the matrix dimension and whose body
  contains the unchanged table

#### Scenario: Default stays expanded

- **WHEN** a matrix is rendered without the flag
- **THEN** the HTML contains no disclosure wrapper

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

### Requirement: NetworkX-driven bigraph layout

`Bigraph.layout(layout=None, *, seed=None, iterations=None, options=None)` SHALL return a
coordinate mapping for every label using NetworkX. The `auto` policy SHALL prefer Kamada-Kawai for
connected graphs when SciPy is available, SHALL use a planar seed plus energy-based spring
refinement for planar graphs containing a four-vertex clique, and SHALL otherwise use an unweighted,
energy-based seeded spring layout. Unknown layout names SHALL raise `ValueError`.

#### Scenario: Auto layout prefers Kamada-Kawai

- **WHEN** a connected bigraph such as D13 is laid out with `auto` and SciPy is available
- **THEN** the positions come from the NetworkX Kamada-Kawai layout

#### Scenario: Planar K4 graphs are relaxed from a planar seed

- **WHEN** the A9 bigraph, whose four-vertex block is a K4, is laid out with `auto`
- **THEN** the positions come from a planar embedding followed by energy-based spring refinement

#### Scenario: Auto layout falls back

- **WHEN** a connected bigraph is laid out with `auto` without SciPy, or a disconnected bigraph is
  laid out with `auto`
- **THEN** the positions come from the seeded NetworkX spring layout and every label has a
  coordinate

#### Scenario: Named layouts are available

- **WHEN** a caller requests `planar`, `kamada_kawai`, `spring`, or `circular`
- **THEN** the corresponding NetworkX layout supplies the positions

#### Scenario: Unknown layout is rejected

- **WHEN** a caller requests an unknown layout name
- **THEN** the operation fails with `ValueError`

#### Scenario: Layout edges are unweighted

- **WHEN** a bigraph with solid, dotted, and arrow edges is laid out
- **THEN** every edge participates with equal adjacency weight and signed algebraic weights do not
  affect the positions

#### Scenario: Seeded layout is repeatable

- **WHEN** the same bigraph is laid out twice with the same seed and layout
- **THEN** every coordinate is identical

#### Scenario: Spring iterations are configurable

- **WHEN** a caller requests the spring layout with an explicit iteration count
- **THEN** NetworkX receives that iteration count and the result has a coordinate for every label

#### Scenario: Spring defaults to many energy iterations

- **WHEN** the spring layout is requested without an explicit iteration count
- **THEN** the default is 10000 iterations using the energy-based spring method

### Requirement: Bigraph drawing

`Bigraph.draw` SHALL render a faithful bigraph on a matplotlib axes using a NetworkX-computed
layout and SHALL preserve solid and dotted styles, arrow direction, multiplicity, labels, optional
highlights, and optional display labels. It SHALL return the axes and SHALL NOT require IPython.

#### Scenario: Mixed styles render

- **WHEN** a bigraph with a solid arrow and a dotted edge is drawn
- **THEN** the axes contain an arrow patch and a dotted line with the expected colors

#### Scenario: Highlights are applied

- **WHEN** highlighted edges and nodes are supplied
- **THEN** those edges and nodes are visually distinguished from the rest

#### Scenario: Computed positions are reusable

- **WHEN** a caller passes the mapping returned by `Bigraph.layout`
- **THEN** the drawing uses those positions instead of recomputing them

#### Scenario: Non-faithful input is rejected

- **WHEN** the `2/4` matrix is drawn as a bigraph
- **THEN** drawing fails with `ValueError`

#### Scenario: No IPython dependency

- **WHEN** the drawing method is called outside a notebook
- **THEN** it returns the axes without importing or requiring IPython

### Requirement: Block-cut tree drawing

`BlockCutTree.draw` SHALL render a block-cut tree on a matplotlib axes. By default it SHALL use a
NetworkX bipartite layout. When original-graph positions are supplied, it SHALL fix articulation
points, remove non-cut vertices, place blocks at centroids, and connect blocks to articulation
points. Block nodes SHALL be squares, vertex nodes circles, articulation points highlighted, and
block label sets shown. It SHALL return the axes.

#### Scenario: A9 block-cut tree renders

- **WHEN** the A9 block-cut tree is drawn without positions
- **THEN** the axes contain four block nodes and nine vertex nodes with the expected labels

#### Scenario: Articulation points are highlighted

- **WHEN** the tree is drawn
- **THEN** vertices 3, 4, and 5 use the articulation-point style

#### Scenario: Centroid layout keeps cut vertices in place

- **WHEN** the A9 tree is drawn with the A9 bigraph positions
- **THEN** vertices 3, 4, and 5 keep those positions, the six non-cut vertices are removed, each of
  the four block nodes lies at the centroid of its block's vertices, and tree edges join each block
  only to its articulation points

#### Scenario: Layout is repeatable

- **WHEN** the same tree is drawn twice in the same environment
- **THEN** every node has the same position

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

### Requirement: Structural certificate Markdown

`ATypeCertificate`, `DCoreGluingCertificate`, and `ACyclizationCertificate` SHALL provide
`to_markdown` and `_repr_markdown_`. A-type SHALL show blocks, partitions, and articulation points;
gluing SHALL show pieces, core edges, labels, and A-types; cyclization SHALL show endpoints, path,
result labels, and type.

#### Scenario: A-type certificate table

- **WHEN** the A9 certificate is rendered
- **THEN** the Markdown lists the four blocks, their sides, and articulation points 3, 4, and 5

#### Scenario: D-core gluing table

- **WHEN** the D13 certificate is rendered
- **THEN** the Markdown lists the five pieces, their core edges, labels, and A-types

#### Scenario: A-cyclization summary

- **WHEN** the A9-to-D8 certificate is rendered
- **THEN** the Markdown reports the endpoints, path, result labels, and type D8

#### Scenario: Hooks match the methods

- **WHEN** a certificate is displayed in a notebook
- **THEN** its `_repr_markdown_` returns the same string as `to_markdown`

### Requirement: Flation witness Markdown

`FlationSequence` SHALL provide `to_markdown` and `_repr_markdown_` listing its steps in application
order. A step with equal source and target SHALL be identified as a sign inversion.

#### Scenario: Sequence table

- **WHEN** a flation sequence is rendered
- **THEN** the Markdown lists every step in application order

#### Scenario: Sign inversion is identified

- **WHEN** a step has equal source and target labels
- **THEN** the Markdown labels that step as a sign inversion

#### Scenario: Hook matches the method

- **WHEN** a flation sequence is displayed in a notebook
- **THEN** its `_repr_markdown_` returns the same string as `to_markdown`

### Requirement: Certificate debug representation remains plain

`repr()` on the structural certificates and on `FlationSequence` SHALL remain dataclass-style text
and SHALL NOT return HTML tags or Markdown table syntax.

#### Scenario: Certificate repr stays plain

- **WHEN** `repr` is called on an A-type, gluing, or cyclization certificate or a flation sequence
- **THEN** the result contains no HTML tags or Markdown table syntax

### Requirement: Bigraph display options

`DisplayOptions` SHALL add bigraph layout policy, layout iteration count, layout seed, figure size,
label visibility, multiplicity visibility, and arrow style settings. Assignment validation and
`reset()` SHALL cover the new fields, and per-call options SHALL override canonical options.

#### Scenario: New fields have usable defaults

- **WHEN** a fresh `DisplayOptions` is created
- **THEN** the bigraph layout policy, layout iteration count, layout seed, figure size, label
  visibility, multiplicity visibility, and arrow style have documented defaults, with the layout
  iteration count defaulting to 10000

#### Scenario: New fields validate assignments

- **WHEN** a negative layout seed, a non-positive iteration count, a non-positive figure size, an
  unknown layout policy, or an unknown arrow style is assigned
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
