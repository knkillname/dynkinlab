# dynkin-classification Specification

## Purpose

Defines the finite Cartan-Killing types and the exact cubic classification of positive definite quasi-Cartan matrices, including witnesses, normal forms, and non-finite outcomes.

## Requirements

### Requirement: Finite Dynkin type value

The system SHALL represent the finite Cartan-Killing types `A_n` (n >= 1), `B_n` (n >= 2), `C_n` (n >= 3), `D_n` (n >= 4), `E_6`, `E_7`, `E_8`, `F_4`, and `G_2` as immutable comparable values exposing family, rank, and canonical name. Invalid family and rank combinations SHALL be rejected.

#### Scenario: Valid types expose family and rank

- **WHEN** a caller creates `A_3`, `D_4`, `E_8`, `F_4`, and `G_2`
- **THEN** each value exposes its family, rank, and canonical name such as `A_3`

#### Scenario: Invalid types are rejected

- **WHEN** a caller requests `A_0`, `B_1`, `C_2`, `D_3`, `E_5`, `F_3`, or `G_3`
- **THEN** construction fails with `ValueError`

### Requirement: Exact classification of connected positive definite matrices

The system SHALL classify any connected admissible positive definite quasi-Cartan matrix, including non-symmetric ones, and return its exact finite Dynkin type in O(n^3) integer operations. It SHALL use exact arithmetic and SHALL NOT rely on floating-point eigenvalues.

#### Scenario: Simply-laced types are classified

- **WHEN** the canonical matrices for `A_3`, `D_4`, `E_6`, `E_7`, and `E_8` are classified
- **THEN** each result reports the corresponding Dynkin type

#### Scenario: Non-simply-laced types are classified

- **WHEN** the canonical matrices for `B_4`, `C_4`, and `F_4` are classified
- **THEN** each result reports the corresponding Dynkin type

#### Scenario: Flation-equivalent input keeps its type

- **WHEN** a Dynkin matrix is transformed by flations, deflations, and sign inversions and then classified
- **THEN** the result reports the original Dynkin type

#### Scenario: Symmetric and non-symmetric inputs use the same classifier

- **WHEN** a symmetric matrix and a non-symmetric symmetrizable matrix are classified
- **THEN** both are accepted as `QuasiCartanMatrix` values and use the same classifier

#### Scenario: Reported steps respect the symmetric inflation bound

- **WHEN** the canonical `E_8` matrix is classified
- **THEN** the result reports a step count that does not exceed the symmetric inflation bound for dimension 8

### Requirement: G2 exception

The system SHALL recognize the 2-by-2 `G_2` Cartan matrix and classify it as `G_2` even though its entries have magnitude 3 and fall outside the admissibility limits.

#### Scenario: Standard G2 orientation is classified

- **WHEN** `((2, -1), (-3, 2))` is classified
- **THEN** the result is `G_2`

#### Scenario: Transposed G2 orientation is classified

- **WHEN** `((2, -3), (-1, 2))` is classified
- **THEN** the result is `G_2`

### Requirement: Admissibility and non-finite outcomes

The system SHALL return `NOT_ADMISSIBLE` for matrices that fail the admissibility conditions, except `G_2`, and `NOT_POSITIVE_DEFINITE` when the inflation bound is exhausted without a Dynkin normal form. It SHALL never loop indefinitely and SHALL report a reason.

#### Scenario: Entry magnitude above the admissible limit is rejected

- **WHEN** a non-G2 matrix contains an off-diagonal entry of magnitude 3 or greater
- **THEN** the result status is `NOT_ADMISSIBLE`

#### Scenario: Parallel edges are rejected

- **WHEN** a matrix has an undirected pair with magnitude greater than 1
- **THEN** the result status is `NOT_ADMISSIBLE`

#### Scenario: Non-positive-definite admissible input stops at the bound

- **WHEN** a degenerate admissible matrix that exhausts the inflation bound or the affine `A_2^(1)` matrix is classified
- **THEN** the result status is `NOT_POSITIVE_DEFINITE` and the call terminates

### Requirement: Rich classification result

The system SHALL return an immutable result exposing status, exact input inertia, component labels and Dynkin types, a flation witness, the final normal form, step count, applied bound, and a verification operation. A classified normal form SHALL be a bigraph whose connected components are Dynkin diagrams.

#### Scenario: Classified result exposes the certificate

- **WHEN** a positive definite `A_2` matrix is classified
- **THEN** the result status is `CLASSIFIED`, the component type is `A_2`, the witness is present, and the normal form is an `A_2` bigraph

#### Scenario: Non-finite result keeps diagnostics

- **WHEN** an admissible matrix is not positive definite
- **THEN** the result status is `NOT_POSITIVE_DEFINITE`, includes the exact inertia, and retains any partial witness and step count

#### Scenario: Result verification replays the witness

- **WHEN** a classified result is verified
- **THEN** applying its witness to the input produces the reported normal form

#### Scenario: Inertia is exact

- **WHEN** the affine `A_2^(1)` matrix is classified
- **THEN** the result inertia has two positive, zero negative, and one zero count

### Requirement: Disconnected classification

The system SHALL classify each connected component independently, return one Dynkin type per component in a deterministic order, preserve duplicate types, and include each component's original labels.

#### Scenario: Mixed components are classified separately

- **WHEN** a disconnected matrix contains an `A_2` block and an `E_6` block
- **THEN** the result contains both component types with their respective labels

#### Scenario: Isolated vertex is A1

- **WHEN** the matrix is `((2,))`
- **THEN** the result is a single `A_1` component

#### Scenario: Duplicate types are preserved

- **WHEN** two disconnected `A_2` components are classified
- **THEN** the result contains two `A_2` entries

### Requirement: First-class witness and normal form

The classifier SHALL return a `FlationSequence` that transforms the input into the normal form and SHALL include every sign inversion and flation used by preprocessing and classification. Reversing the witness SHALL recover the input.

#### Scenario: Witness includes sign inversions

- **WHEN** a non-symmetric input requires a sign inversion during preprocessing
- **THEN** that sign inversion appears as a step in the witness

#### Scenario: Inverse witness recovers the input

- **WHEN** the inverse witness is applied to the normal form
- **THEN** the original matrix is recovered

### Requirement: Public surface for Dynkin classification

The system SHALL export `DynkinType`, `DynkinClassification`, and the classification status value from the top-level `dynkinlab` package and SHALL list them in `__all__`.

#### Scenario: Imports succeed

- **WHEN** a caller imports the Dynkin type and classification names from `dynkinlab`
- **THEN** the imports succeed and the names appear in `dynkinlab.__all__`

### Requirement: Early rejection of indefinite components

The system SHALL reject an admissible component with a negative eigenvalue as `NOT_POSITIVE_DEFINITE` before running any flation. The result SHALL report reason `indefinite`, zero steps, and an empty witness. Non-negative components SHALL retain the bounded flation loop.

#### Scenario: Indefinite admissible component is rejected before flation

- **WHEN** an admissible component has at least one negative eigenvalue
- **THEN** the result status is `NOT_POSITIVE_DEFINITE`, the reason is `indefinite`, the step count is zero, and the witness is empty

#### Scenario: Non-negative components keep the bounded behavior

- **WHEN** an admissible component has no negative eigenvalue but is degenerate
- **THEN** the classifier retains the existing bounded flation behavior and reports `NOT_POSITIVE_DEFINITE`

#### Scenario: Indefinite rejection does not affect classified components

- **WHEN** a disconnected matrix contains a classified positive definite component and an indefinite component
- **THEN** the classified component retains its Dynkin type and labels, and the result status is `NOT_POSITIVE_DEFINITE`
