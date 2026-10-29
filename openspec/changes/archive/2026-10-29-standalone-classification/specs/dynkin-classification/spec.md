# Spec Delta

## ADDED Requirements

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

## MODIFIED Requirements

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

## REMOVED Requirements

### Requirement: Cubic classification of connected positive definite matrices

**Reason**: Replaced by "Exact classification of connected positive definite matrices" so the requirement and its scenario describe the algorithm directly rather than through source-relative wording.

**Migration**: No caller migration; inputs, outputs, behavior, and complexity are unchanged.
