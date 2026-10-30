# bigraph-solidification Specification

## Purpose

Decides whether a bigraph admits a sequence of flations to a solid bigraph, returning either a
verifiable witness, a proof-backed impossibility certificate, or an explicit inconclusive result.

## Requirements

### Requirement: Solidification entry point

`Bigraph` SHALL expose `solidify(*, max_states=..., max_entry=...)` returning an immutable
`SolidificationResult`. `max_states` and `max_entry` SHALL be positive integers, and invalid values
SHALL raise `ValueError`.

#### Scenario: Dotted A2 solidifies

- **WHEN** an A2 bigraph with a dotted edge is solidified
- **THEN** the status is `SolidificationStatus.SOLIDIFIED`, the final bigraph is solid, and the sequence is non-empty

#### Scenario: Already solid input uses the empty sequence

- **WHEN** a bigraph with no dotted edges is solidified
- **THEN** the status is `SolidificationStatus.SOLIDIFIED`, the sequence is empty, and the final bigraph equals the input

#### Scenario: Invalid search caps are rejected

- **WHEN** `max_states` or `max_entry` is zero, negative, or not an integer
- **THEN** the call fails with `ValueError`

### Requirement: Solidification witness

A solidified result SHALL expose a `FlationSequence` and the final `Bigraph`. Replaying the
sequence on the input SHALL produce the final bigraph, the final bigraph SHALL have no dotted
edges, and every step SHALL use the original labels.

#### Scenario: Witness replays exactly

- **WHEN** a solidified result is verified
- **THEN** the sequence maps the input to the reported final bigraph and the final bigraph is solid

#### Scenario: Reversing the witness recovers the input

- **WHEN** the inverse sequence is applied to the final bigraph
- **THEN** the original bigraph is recovered

#### Scenario: Labels are preserved

- **WHEN** a labelled bigraph is solidified
- **THEN** every step in the sequence and the final bigraph use the original labels

### Requirement: Positive semidefinite impossibility certificate

For a connected positive semidefinite input with corank at least two, the result SHALL be
`SolidificationStatus.IMPOSSIBLE` and SHALL carry a `NoSolidCertificate` with the exact inertia
and a reason. The certificate SHALL verify against the input. This branch is rigorous because a
connected positive semidefinite Cartan matrix has corank at most one, and flation preserves inertia
and connectedness.

#### Scenario: Corank-two PSD matrix is impossible

- **WHEN** the connected PSD matrix with all off-diagonal entries 2 is solidified
- **THEN** the status is `IMPOSSIBLE` and the certificate reports corank two or more

#### Scenario: Certificate verifies

- **WHEN** the certificate from the previous scenario is verified against the input
- **THEN** verification succeeds

#### Scenario: Indefinite inputs are not certified impossible

- **WHEN** an indefinite input reaches the search cap without a solid witness
- **THEN** the status is `INCONCLUSIVE`, not `IMPOSSIBLE`, and no impossibility certificate is returned

### Requirement: Inconclusive bounded search

When the search reaches `max_states` or an intermediate entry exceeds `max_entry` without finding a
solid matrix, an indefinite result SHALL be `SolidificationStatus.INCONCLUSIVE`. The result SHALL
report the explored exact-state count, frame-class count, and applied caps. It SHALL NOT claim that
no solid representative exists.

#### Scenario: Candidate is inconclusive at a small cap

- **WHEN** the indefinite candidate from the investigation is searched with a small state cap
- **THEN** the status is `INCONCLUSIVE` and the result reports the applied caps and explored states

#### Scenario: Inconclusive is not a witness

- **WHEN** an inconclusive result is verified
- **THEN** verification fails because neither a sequence nor a certificate is present

### Requirement: Componentwise composition

For a disconnected input, the system SHALL decide each connected component independently. It SHALL
concatenate component witnesses in bigraph-label order. If any component is impossible, the whole
result SHALL be impossible; otherwise, if any component is inconclusive, the whole result SHALL be
inconclusive.

#### Scenario: Two solidifiable components compose

- **WHEN** a bigraph consists of two dotted A2 components
- **THEN** the result is solidified and the witness contains steps for both components

#### Scenario: One impossible component wins

- **WHEN** a disconnected bigraph has one solidifiable component and one corank-two PSD component
- **THEN** the whole result is impossible

#### Scenario: One inconclusive component propagates

- **WHEN** a disconnected bigraph has one solidifiable component and one indefinite component that exhausts the cap
- **THEN** the whole result is inconclusive

### Requirement: Exact search moves and arithmetic

The search SHALL use exact integer arithmetic and SHALL consider sign inversions at every vertex
and flations over dotted edges in both directions. It SHALL be deterministic for the same input and
caps.

#### Scenario: Sign inversions are available

- **WHEN** a state requires a sign inversion before its dotted edges can be removed
- **THEN** the returned sequence includes that sign inversion

#### Scenario: Large intermediate entries stay exact

- **WHEN** the search needs an intermediate entry beyond the machine-integer range
- **THEN** the computation remains exact until `max_entry` rejects the state

#### Scenario: Repeated runs agree

- **WHEN** the same input and caps are solidified twice
- **THEN** both results have the same status, sequence, and search counters

### Requirement: Frame and sign accounting

The result SHALL report the number of exact states explored and the number of frame-isomorphism
classes. The frame SHALL use absolute off-diagonal weights up to vertex isomorphism. Sign-inverted
states SHALL remain distinct in the exact-state count when they can change the next move.

#### Scenario: Frame classes do not exceed exact states

- **WHEN** any search completes or stops at a cap
- **THEN** the reported frame-class count is at most the exact-state count

#### Scenario: Relabellings share a frame class

- **WHEN** two search states differ only by a vertex permutation
- **THEN** they contribute one frame class

### Requirement: Public surface and rendering

The system SHALL export `SolidificationStatus`, `NoSolidCertificate`, and `SolidificationResult`
from the top-level `dynkinlab` package and list them in `dynkinlab.__all__`. Results and
certificates SHALL provide Markdown and notebook rendering.

#### Scenario: Imports succeed

- **WHEN** the solidification names are imported from `dynkinlab`
- **THEN** the imports succeed and the names appear in `dynkinlab.__all__`

#### Scenario: Result renders its status

- **WHEN** a result is rendered as Markdown
- **THEN** the output identifies the status and, when present, the step count or certificate reason

#### Scenario: Notebook hook uses the same rendering

- **WHEN** a result or certificate is displayed in a notebook
- **THEN** its Markdown hook returns the same content as `to_markdown()`
