# bigraph-structure Specification

## Purpose

Defines structural inspection of bigraphs: NetworkX conversion, biconnected blocks, block-cut forests, A-block partitions, and verifiable A-type certificates.

## Requirements

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

### Requirement: Edge multiset arithmetic

The system SHALL represent a finite labeled multiset of undirected solid and dotted edges, including
opposite parallel edges between the same pair. Addition SHALL union labels and add edge
multiplicities. Simplification SHALL cancel opposite-style pairs and return a bigraph.

#### Scenario: Opposite parallel edges are preserved

- **WHEN** a solid edge and a dotted edge join the same two labels
- **THEN** the multiset retains both until simplification

#### Scenario: Same-style parallel edges combine

- **WHEN** two solid edges join the same two labels
- **THEN** the multiset contains one solid edge with multiplicity two

#### Scenario: Simplification cancels equal opposite pairs

- **WHEN** a solid edge and a dotted edge of equal multiplicity are simplified
- **THEN** no edge remains between their endpoints

#### Scenario: Simplification leaves the excess style

- **WHEN** a solid edge of multiplicity three and a dotted edge of multiplicity one are simplified
- **THEN** a solid edge of multiplicity two remains

#### Scenario: Directed edges are rejected

- **WHEN** a caller adds a directed edge to the multiset
- **THEN** construction fails with `ValueError`

### Requirement: Bigraph sum

The system SHALL combine two or more symmetric bigraphs by identifying shared labels, adding
off-diagonal weights, keeping diagonal entries at 2, and simplifying after addition. The result
SHALL accept an optional explicit label order covering exactly the union of the piece labels.

#### Scenario: Disjoint pieces form a disjoint union

- **WHEN** an A2 bigraph and an A3 bigraph with disjoint labels are summed
- **THEN** the result contains both components and all five labels

#### Scenario: Shared labels are identified

- **WHEN** two pieces share one label
- **THEN** the sum contains that label once and combines its incident edges

#### Scenario: Opposite edges cancel in the sum

- **WHEN** two pieces contribute solid and dotted edges between the same pair of labels
- **THEN** the sum contains no edge between that pair

#### Scenario: Explicit label order is preserved

- **WHEN** a caller supplies an explicit order for all union labels
- **THEN** the result uses that label order

#### Scenario: Incomplete explicit order is rejected

- **WHEN** the explicit order omits a label that appears in a piece
- **THEN** the sum fails with `ValueError`

#### Scenario: A mixed piece is rejected

- **WHEN** one piece contains an arrow
- **THEN** the sum fails with `ValueError`

### Requirement: Bigraph vertex identification

The system SHALL merge two distinct labels of a symmetric bigraph, drop any loop created by the
merge, add incident edge weights, and simplify opposite parallel edges. The merged label SHALL keep
the position of the retained label in the result order.

#### Scenario: Non-adjacent vertices merge

- **WHEN** two non-adjacent labels are identified
- **THEN** their neighbors become neighbors of one merged label

#### Scenario: Adjacent vertices drop the new loop

- **WHEN** two adjacent labels are identified
- **THEN** the edge between them becomes a loop and is discarded

#### Scenario: Opposite edges cancel after merging

- **WHEN** two identified labels each contribute an opposite edge to a common neighbor
- **THEN** the result contains no edge to that neighbor

#### Scenario: Label order is retained

- **WHEN** a label is merged into an earlier label
- **THEN** the result keeps the earlier label's position and removes the other

#### Scenario: Invalid labels are rejected

- **WHEN** the two labels are equal or one label is absent
- **THEN** identification fails with `ValueError`

### Requirement: D-core cycle value

The system SHALL represent a D-core as a cyclic sequence of at least two distinct labels with one
edge from each label to the next, permitting a two-edge double link when the length is two. A core
SHALL be valid only when its number of dotted edges is odd.

#### Scenario: Odd five-cycle is accepted

- **WHEN** a five-label cycle has exactly three dotted edges
- **THEN** the core is valid and exposes its cycle condition as true

#### Scenario: Double link is accepted

- **WHEN** a two-label core has one solid edge and one dotted edge
- **THEN** the core is valid and represents the double link

#### Scenario: Even dotted count is rejected

- **WHEN** a core has an even number of dotted edges
- **THEN** construction fails with `ValueError`

#### Scenario: A chord is rejected

- **WHEN** a core contains an edge that is not between consecutive cycle labels
- **THEN** construction fails with `ValueError`

#### Scenario: Duplicate labels are rejected

- **WHEN** a core repeats a label
- **THEN** construction fails with `ValueError`

#### Scenario: Missing cycle edges are rejected

- **WHEN** a core omits one of its consecutive cycle edges
- **THEN** construction fails with `ValueError`

### Requirement: D-core gluing construction

The system SHALL construct a D-core gluing from a valid core and one connected positive definite
A-type piece per core edge. Each piece SHALL contain its core edge with the same style and internal
endpoints, overlaps SHALL be exactly the consecutive core vertices, and the sum SHALL be the claimed
D_n bigraph.

#### Scenario: D13 gluing verifies

- **WHEN** the thesis's five pieces A3, A4, A2, A7, and A2 are glued along the odd five-cycle
- **THEN** the result is the 13-vertex D13 bigraph and its certificate verifies

#### Scenario: Double-link D4 gluing verifies

- **WHEN** a solid A2 edge and an A4 piece are glued along a two-edge core with one solid and one
  dotted edge
- **THEN** the result is D4 and its certificate verifies

#### Scenario: A non-A piece is rejected

- **WHEN** a piece has no A-type certificate
- **THEN** the gluing fails with `ValueError`

#### Scenario: A non-positive-definite piece is rejected

- **WHEN** a structurally A-type piece has a zero eigenvalue
- **THEN** the gluing fails with `ValueError`

#### Scenario: A mismatched core edge is rejected

- **WHEN** a piece contains its core edge with a different style
- **THEN** the gluing fails with `ValueError`

#### Scenario: An articulation endpoint is rejected

- **WHEN** a core vertex is an articulation point of its piece
- **THEN** the gluing fails with `ValueError`

#### Scenario: An unexpected overlap is rejected

- **WHEN** two pieces share a vertex outside their consecutive core intersection
- **THEN** the gluing fails with `ValueError`

#### Scenario: The wrong number of pieces is rejected

- **WHEN** the number of pieces differs from the number of core edges
- **THEN** the gluing fails with `ValueError`

### Requirement: D-core gluing certificate verification

A D-core gluing certificate SHALL verify by recomputing the construction from its stored core,
pieces, and label order and comparing the result with its stored D_n gluing.

#### Scenario: A valid certificate verifies

- **WHEN** a certificate produced by a valid D13 gluing is checked
- **THEN** verification returns true

#### Scenario: A tampered piece is rejected

- **WHEN** a certificate's stored piece is replaced after construction
- **THEN** verification returns false

#### Scenario: A tampered gluing is rejected

- **WHEN** a certificate's stored gluing is replaced after construction
- **THEN** verification returns false

### Requirement: A-cyclization construction

The system SHALL construct an A-cyclization from a connected positive definite A-type bigraph with
at least five vertices and two distinct internal vertices. It SHALL require a unique shortest path of
length at least two with an odd number of dotted edges, identify the endpoints, and return a
certificate for D_{n-1}.

#### Scenario: A9 to D8 verifies

- **WHEN** vertices 1 and 6 of the 9-vertex A-block tree are identified
- **THEN** the result is D8 and its certificate verifies

#### Scenario: Adjacent vertices are rejected

- **WHEN** the chosen vertices have a shortest path of length one
- **THEN** cyclization fails with `ValueError`

#### Scenario: Even dotted parity is rejected

- **WHEN** the unique shortest path contains an even number of dotted edges
- **THEN** cyclization fails with `ValueError`

#### Scenario: An articulation vertex is rejected

- **WHEN** either chosen vertex is an articulation point of the source
- **THEN** cyclization fails with `ValueError`

#### Scenario: A non-A source is rejected

- **WHEN** the source has no A-type certificate
- **THEN** cyclization fails with `ValueError`

#### Scenario: A disconnected source is rejected

- **WHEN** the A-type source has more than one connected component
- **THEN** cyclization fails with `ValueError`

#### Scenario: Too few vertices are rejected

- **WHEN** the A-type source has fewer than five vertices
- **THEN** cyclization fails with `ValueError`

### Requirement: A-cyclization certificate verification

An A-cyclization certificate SHALL verify by recomputing the path conditions and the identified
bigraph from its stored source, endpoints, and path, then comparing them with its stored D_{n-1}
result.

#### Scenario: A valid certificate verifies

- **WHEN** a certificate produced by the A9 to D8 cyclization is checked
- **THEN** verification returns true

#### Scenario: A tampered result is rejected

- **WHEN** a certificate's stored result is replaced after construction
- **THEN** verification returns false

#### Scenario: A tampered path is rejected

- **WHEN** a certificate's stored path is replaced after construction
- **THEN** verification returns false

### Requirement: Public surface for D structural values

The system SHALL export `EdgeMultiset`, `DCore`, `DCoreGluingCertificate`, and
`ACyclizationCertificate` from the top-level package and list them in `__all__`.

#### Scenario: D structural imports succeed

- **WHEN** a caller imports the D structural values from `dynkinlab`
- **THEN** the imports succeed and every name appears in `dynkinlab.__all__`
