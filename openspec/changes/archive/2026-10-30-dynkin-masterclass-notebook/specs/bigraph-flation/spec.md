# Spec Delta

## ADDED Requirements

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

## MODIFIED Requirements

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
