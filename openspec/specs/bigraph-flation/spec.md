# bigraph-flation Specification

## Purpose

Defines label-generic mixed bigraphs over quasi-Cartan matrices, their edge and arrow representation, and first-class flation witnesses.

## Requirements

### Requirement: Label-generic Bigraph view

The system SHALL construct an immutable `Bigraph` view over a `QuasiCartanMatrix` with an optional sequence of unique hashable labels. Labels SHALL default to `0..n-1`, SHALL match the matrix dimension, and SHALL be copied into the value.

#### Scenario: String labels are preserved

- **WHEN** a bigraph is created from the `A_2` matrix with labels `("left", "right")`
- **THEN** the labels are `("left", "right")` and the underlying matrix is unchanged

#### Scenario: Labels default to indices

- **WHEN** a bigraph is created without labels
- **THEN** its labels are the integers from 0 through dimension minus 1

#### Scenario: Invalid label sequences are rejected

- **WHEN** labels are duplicated or their count differs from the matrix dimension
- **THEN** construction fails with `ValueError`

### Requirement: Path notation construction

The system SHALL provide `Bigraph.from_paths(text, labels=None, *, arrow_multiplicity=2)` to
construct a labeled bigraph from the six-operator path notation. It SHALL accept integer and
identifier labels, comments, whitespace or comma separators, isolated vertices, and explicit label
order; conflicting declarations and labels outside a pinned order SHALL raise `ValueError`.

#### Scenario: A9 path text is parsed

- **WHEN** the A9 path text and labels `1..9` are passed to `from_paths`
- **THEN** the resulting bigraph has the expected matrix and labels in first-appearance order

#### Scenario: Explicit label order is preserved

- **WHEN** labels are pinned explicitly
- **THEN** the matrix order follows that sequence and a path label outside it raises `ValueError`

#### Scenario: Conflicting declarations are rejected

- **WHEN** the same vertex pair is declared twice with different entries
- **THEN** construction fails with `ValueError`

#### Scenario: Arrow multiplicity is configurable

- **WHEN** `arrow_multiplicity=3` is used
- **THEN** every arrow has reverse magnitude 3 and the resulting matrix is faithful

#### Scenario: Invalid path text is rejected

- **WHEN** the text is empty without pinned labels or has a malformed path
- **THEN** construction fails with `ValueError`

### Requirement: Mixed edge and arrow derivation

The system SHALL derive one public edge value per non-zero vertex pair that has a faithful
representation. Equal matrix entries SHALL produce an undirected edge; unequal magnitudes with one
side of magnitude one SHALL produce an arrow from the smaller-magnitude side to the larger. The
signed weight SHALL be the common sign times the larger magnitude. Pairs outside this range SHALL
raise `ValueError` when their edges are requested.

#### Scenario: Symmetric pair produces a solid edge

- **WHEN** a bigraph is created from `((2, -1), (-1, 2))`
- **THEN** it has one solid undirected edge with multiplicity 1 between the two labels

#### Scenario: Asymmetric negative pair produces a solid arrow

- **WHEN** a bigraph with labels `("a", "b")` is created from `((2, -1), (-2, 2))`
- **THEN** it has a solid arrow from `"a"` to `"b"` with multiplicity 2

#### Scenario: Asymmetric positive pair produces a dotted arrow

- **WHEN** a bigraph with labels `("a", "b")` is created from `((2, 1), (2, 2))`
- **THEN** it has a dotted arrow from `"a"` to `"b"` with multiplicity 2

#### Scenario: Zero pair produces no edge

- **WHEN** the corresponding matrix entries are both zero
- **THEN** no edge or arrow is present for that pair

#### Scenario: Non-faithful pair is rejected

- **WHEN** a bigraph over `((2, 2), (4, 2))` is asked for its edges
- **THEN** the request fails with `ValueError` instead of returning a lossy multiplicity-four arrow

### Requirement: Faithful bigraph predicate

`QuasiCartanMatrix.has_faithful_bigraph` SHALL report whether every off-diagonal pair is zero,
symmetric, or an arrow with one side of magnitude one. The predicate SHALL be consistent with edge
derivation: a non-faithful pair SHALL raise `ValueError` when edges are requested.

#### Scenario: Positive definite matrices are faithful

- **WHEN** the A2 and A3 matrices are inspected
- **THEN** `has_faithful_bigraph` is true for both

#### Scenario: The 2/4 matrix is not faithful

- **WHEN** the matrix `((2, 2), (4, 2))` is inspected
- **THEN** `has_faithful_bigraph` is false and requesting its bigraph edges raises `ValueError`

#### Scenario: A unit arrow is faithful

- **WHEN** the matrix `((2, 1), (3, 2))` is inspected
- **THEN** `has_faithful_bigraph` is true and its edge is an arrow of multiplicity 3

#### Scenario: A symmetric dotted pair is faithful

- **WHEN** the matrix `((2, 1), (1, 2))` is inspected
- **THEN** `has_faithful_bigraph` is true and its edge is dotted

### Requirement: Public edge values

The system SHALL expose immutable edge values with source label, target label, direction flag, edge style, and positive integer multiplicity. Styles SHALL be solid or dotted, and source and target SHALL be distinct labels present in the bigraph.

#### Scenario: Edge value is inspectable and hashable

- **WHEN** a caller reads an edge from a bigraph
- **THEN** it exposes its source, target, direction, style, and multiplicity and can be used in a set

#### Scenario: Invalid edge values are rejected

- **WHEN** an edge has equal endpoints, a non-positive multiplicity, or an unknown style
- **THEN** construction fails with `ValueError`

### Requirement: Flation delegation

The system SHALL delegate `Bigraph.flation(source_label, target_label)` to the underlying matrix and return a new bigraph with the same labels. Equal source and target labels SHALL perform the matrix sign inversion. Unknown labels SHALL be rejected.

#### Scenario: Dotted edge flips to solid

- **WHEN** the flation of a bigraph over `((2, 1), (1, 2))` is applied from the first label to the second
- **THEN** the resulting bigraph has a solid edge between the same labels

#### Scenario: Sign inversion is available through the bigraph

- **WHEN** the flation is applied with the same source and target label
- **THEN** the incident edge styles are inverted and the labels are preserved

#### Scenario: Unknown labels are rejected

- **WHEN** a flation uses a label that is not in the bigraph
- **THEN** the operation fails with `ValueError`

#### Scenario: Flation preserves inertia

- **WHEN** a bigraph over a positive definite matrix is flated
- **THEN** the underlying matrix inertia is unchanged

### Requirement: First-class flation witness

The system SHALL expose immutable flation steps and sequences. A step SHALL identify source and target labels; a sequence SHALL apply to a bigraph, be invertible by reversing its steps, and verify a claimed final bigraph.

#### Scenario: A sequence replays transformations

- **WHEN** a two-step sequence is applied to a bigraph
- **THEN** the result equals applying the same steps one at a time

#### Scenario: Reversing a sequence recovers the start

- **WHEN** the inverse of a sequence is applied to its result
- **THEN** the original bigraph is recovered

#### Scenario: Verification accepts and rejects correctly

- **WHEN** a sequence is verified against its actual result or a different bigraph
- **THEN** verification succeeds for the actual result and fails for the different bigraph

### Requirement: Classification delegation

The system SHALL delegate `Bigraph.classify()` to the underlying matrix and map component indices to the bigraph labels. Component labels and witness steps in the result SHALL use the bigraph labels.

#### Scenario: Labeled components are returned

- **WHEN** a labeled bigraph over a disconnected positive definite matrix is classified
- **THEN** each component reports its original labels and Dynkin type

#### Scenario: Witness uses bigraph labels

- **WHEN** a classified bigraph carries a flation witness
- **THEN** every step in the witness identifies source and target by bigraph labels

### Requirement: Public surface for bigraphs and flations

The system SHALL export `Bigraph`, its public edge value, its style value, `FlationStep`, and `FlationSequence` from the top-level `dynkinlab` package and SHALL list them in `__all__`.

#### Scenario: Imports succeed

- **WHEN** a caller imports the bigraph and flation names from `dynkinlab`
- **THEN** the imports succeed and the names appear in `dynkinlab.__all__`
