# Spec Delta

## Purpose

Defines structural inspection of bigraphs: NetworkX conversion, biconnected blocks, block-cut forests, A-block partitions, and verifiable A-type certificates.

## ADDED Requirements

### Requirement: NetworkX graph conversion

The system SHALL convert a bigraph to an undirected NetworkX graph containing every vertex label and one edge per nonzero pair, with style, direction, multiplicity, and signed weight attributes. Isolated labels SHALL remain nodes.

#### Scenario: Labels and edge attributes are preserved

- **WHEN** a labeled mixed bigraph is converted
- **THEN** every label is a node and each edge exposes style, direction, multiplicity, and weight

#### Scenario: Isolated vertices are retained

- **WHEN** a bigraph has a label with no incident edges
- **THEN** the converted graph contains that label as an isolated node

### Requirement: Block-cut forest

The system SHALL compute one block-cut tree per connected component, exposing each tree's blocks and articulation labels. A block SHALL expose its vertex labels and induced bigraph.

#### Scenario: A9 block-cut structure

- **WHEN** the 9-vertex A-block example with labels 1 through 9 is inspected
- **THEN** the four blocks are `{1,2,3}`, `{3,4,5,9}`, `{4,7}`, and `{5,6,8}`, and the articulation labels are 3, 4, and 5

#### Scenario: A bridge is a block

- **WHEN** two triangles are joined by a single solid edge
- **THEN** that edge forms a block and its two endpoints are articulation labels

#### Scenario: Disconnected components have separate trees

- **WHEN** the bigraph has two connected components
- **THEN** the forest contains one tree per component

#### Scenario: A single vertex is represented

- **WHEN** a component consists of one label
- **THEN** its tree contains that label and no articulation points

### Requirement: A-block partition

The system SHALL decide whether an induced vertex set is an A-block and, when it is, return disjoint sides X and Y such that cross pairs are solid and within-side pairs are dotted.

#### Scenario: A standard A-block is accepted

- **WHEN** a block has X={a,b}, Y={c,d}, cross pairs solid, and within-side pairs dotted
- **THEN** the returned partition verifies against the block

#### Scenario: An all-dotted block is accepted

- **WHEN** a block has only dotted edges
- **THEN** it is an A-block with Y empty

#### Scenario: A solid triangle is rejected

- **WHEN** a triangle has three solid edges
- **THEN** no A-block partition is returned

#### Scenario: A single solid edge is accepted

- **WHEN** a block consists of one solid edge
- **THEN** the partition places one endpoint in X and the other in Y

### Requirement: A-type certificate

The system SHALL decide whether every connected component is A-type and return immutable per-component certificates. A certificate SHALL verify the block-cut tree, A-block partitions, and that each articulation vertex belongs to exactly two blocks.

#### Scenario: A9 is certified

- **WHEN** the 9-vertex A-block example is checked
- **THEN** the result is A-type and the certificate verifies

#### Scenario: D13 is rejected

- **WHEN** the 13-vertex D-core example is checked
- **THEN** the result is not A-type and no certificate is returned

#### Scenario: A1 and A2 are certified

- **WHEN** a single vertex or a single edge is checked
- **THEN** both are A-type and their certificates verify

#### Scenario: A disjoint union is certified componentwise

- **WHEN** a bigraph is the disjoint union of A2 and A3
- **THEN** the result is A-type with one certificate per component

#### Scenario: A tampered certificate is rejected

- **WHEN** a valid A-type certificate is verified against a different bigraph
- **THEN** verification fails

### Requirement: Public surface for structural bigraph values

The system SHALL export the structural block, forest, partition, and certificate values from the top-level package and list them in `__all__`.

#### Scenario: Structural imports succeed

- **WHEN** a caller imports the structural values from `dynkinlab`
- **THEN** the imports succeed and every name appears in `dynkinlab.__all__`
