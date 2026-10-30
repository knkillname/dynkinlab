# Spec Delta

## ADDED Requirements

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

## MODIFIED Requirements

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
