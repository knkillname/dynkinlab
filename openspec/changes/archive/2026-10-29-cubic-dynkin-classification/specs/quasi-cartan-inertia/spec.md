# Spec Delta

## MODIFIED Requirements

### Requirement: Quasi-Cartan matrix construction

The system SHALL construct an immutable non-empty square integer matrix with diagonal 2 and a positive diagonal integer symmetrizer `D` for which `D A` is symmetric. Symmetric matrices are the special case `D = (1, ..., 1)`. Construction SHALL fail for invalid shape, entries, diagonal, or symmetrizability.

#### Scenario: Valid matrix is accepted

- **WHEN** a caller supplies `((2, -1), (-1, 2))`
- **THEN** construction succeeds and the value reports dimension 2, the original entries, symmetrizer `(1, 1)`, and symmetrized entries equal to the original entries

#### Scenario: Valid non-symmetric matrix is accepted

- **WHEN** a caller supplies `((2, -1), (-2, 2))`
- **THEN** construction succeeds with dimension 2, the original entries, symmetrizer `(2, 1)`, and symmetrized entries `((4, -2), (-2, 2))`

#### Scenario: Invalid shapes and entries are rejected

- **WHEN** a caller supplies an empty, non-square, ragged, non-integer, or non-2-diagonal matrix
- **THEN** construction fails with `ValueError` and no object is created

#### Scenario: Non-symmetrizable matrix is rejected

- **WHEN** a caller supplies `((2, -1, -2), (-2, 2, -1), (-1, -2, 2))`
- **THEN** construction fails with `ValueError` because no positive diagonal integer symmetrizer exists

#### Scenario: Strict symmetric factory rejects a non-symmetric matrix

- **WHEN** a caller supplies `((2, -1), (-2, 2))` to the strict symmetric factory
- **THEN** it fails with `ValueError`, even though the general constructor accepts the same matrix

### Requirement: Integer quadratic form

The system SHALL expose the exact integer form `q_A(x) = sum_i d_i x_i^2 + sum_{i<j} d_i a_ij x_i x_j` for a quasi-Cartan matrix with symmetrizer `D` and SHALL evaluate it exactly for integer vectors of the matrix dimension.

#### Scenario: Known values are exact

- **WHEN** the form of `((2, -1), (-1, 2))` is evaluated at `(1, 1)`
- **THEN** the result is the integer `1`

#### Scenario: Non-symmetric values are exact

- **WHEN** the form of `((2, -1), (-2, 2))` is evaluated at `(1, 1)`
- **THEN** the result is the integer `1`

#### Scenario: Invalid vectors are rejected

- **WHEN** a vector has the wrong length or a non-integer entry
- **THEN** evaluation fails with `ValueError`

### Requirement: Exact inertia

The system SHALL report the number of positive, negative, and zero eigenvalues of the symmetrized matrix `S = D A` using exact arithmetic, and MUST NOT rely on floating-point eigenvalue approximations.

#### Scenario: Positive definite Cartan matrix

- **WHEN** the inertia of the `A_2` matrix `((2, -1), (-1, 2))` is requested
- **THEN** the result is two positive, zero negative, and zero zero counts

#### Scenario: Positive definite non-symmetric matrix

- **WHEN** the inertia of `((2, -1), (-2, 2))` is requested
- **THEN** the result is two positive, zero negative, and zero zero counts

#### Scenario: Degenerate affine matrix

- **WHEN** the inertia of the affine `A_2^(1)` matrix is requested
- **THEN** the result is two positive, zero negative, and one zero count

#### Scenario: Indefinite matrix

- **WHEN** the inertia of `((2, 3), (3, 2))` is requested
- **THEN** the result is one positive, one negative, and zero zero counts

#### Scenario: Reordering indices preserves inertia

- **WHEN** a quasi-Cartan matrix is reordered by the same permutation of rows and columns
- **THEN** the reported inertia is unchanged

## ADDED Requirements

### Requirement: Canonical symmetrizer and symmetrized entries

The system SHALL expose a canonical positive diagonal integer symmetrizer with minimal trace and symmetrized entries `S = D A`. The canonical choice SHALL be deterministic and SHALL normalize each connected component independently.

#### Scenario: Connected non-symmetric matrix has a canonical symmetrizer

- **WHEN** the symmetrizer of `((2, -1), (-2, 2))` is requested
- **THEN** it is `(2, 1)`, not `(4, 2)` or another positive scalar multiple

#### Scenario: Symmetric matrix uses the identity symmetrizer

- **WHEN** the matrix is `((2, -1), (-1, 2))`
- **THEN** the symmetrizer is `(1, 1)` and the symmetrized entries equal the original entries

#### Scenario: Original entries remain available

- **WHEN** the original entries and symmetrized entries of `((2, -1), (-2, 2))` are requested
- **THEN** the original entries are `((2, -1), (-2, 2))` and the symmetrized entries are `((4, -2), (-2, 2))`

#### Scenario: Disconnected components are normalized independently

- **WHEN** the matrix is block diagonal with a symmetric `A_2` block and the non-symmetric block `((2, -1), (-2, 2))`
- **THEN** the symmetrizer is `(1, 1, 2, 1)` and each block has minimal trace

### Requirement: Symmetric special case

`QuasiCartanMatrix` SHALL expose an `is_symmetric` property and a strict `symmetric(entries)` factory that accepts only symmetric matrices. The general constructor SHALL accept both symmetric and non-symmetric symmetrizable matrices.

#### Scenario: Symmetry is reported

- **WHEN** `is_symmetric` is read for `((2, -1), (-1, 2))` and `((2, -1), (-2, 2))`
- **THEN** it is true for the first matrix and false for the second

#### Scenario: Strict factory accepts a symmetric matrix

- **WHEN** `((2, -1), (-1, 2))` is passed to the strict symmetric factory
- **THEN** construction succeeds and the result is symmetric

#### Scenario: Strict factory rejects a non-symmetric matrix

- **WHEN** `((2, -1), (-2, 2))` is passed to the strict symmetric factory
- **THEN** construction fails with `ValueError`

### Requirement: Quasi-Cartan flation

`QuasiCartanMatrix` SHALL apply the flation `T_s,r`, including the sign inversion when `s = r`, and return a new matrix. Flation SHALL preserve the quasi-Cartan form, the symmetrizer family, and inertia, and SHALL be an involution.

#### Scenario: Dotted edge becomes solid

- **WHEN** the flation of `((2, 1), (1, 2))` is applied with source 0 and target 1
- **THEN** the result is `((2, -1), (-1, 2))`

#### Scenario: Sign inversion flips incident entries

- **WHEN** the flation of `((2, 1), (1, 2))` is applied with source 0 and target 0
- **THEN** the result is `((2, -1), (-1, 2))`

#### Scenario: Flation is an involution

- **WHEN** the same flation is applied twice to a quasi-Cartan matrix
- **THEN** the original matrix is recovered

#### Scenario: Flation preserves inertia

- **WHEN** a positive definite non-symmetric matrix is flated
- **THEN** its inertia is unchanged

#### Scenario: Only the target row and column change

- **WHEN** a flation with distinct source and target is applied
- **THEN** every entry outside the target row and target column is unchanged
