# Spec Delta

## Purpose

Defines symmetric quasi-Cartan matrices, their exact integer quadratic forms, and exact inertia so later bigraph and classification work can depend on stable algebraic invariants.

## ADDED Requirements

### Requirement: Quasi-Cartan matrix construction

The system SHALL construct an immutable quasi-Cartan matrix from a non-empty square matrix of integers that is symmetric and has 2 on every diagonal entry.

#### Scenario: Valid matrix is accepted

- **WHEN** a caller supplies `((2, -1), (-1, 2))`
- **THEN** construction succeeds and the value reports dimension 2 and its original entries

#### Scenario: Invalid shapes and entries are rejected

- **WHEN** a caller supplies an empty, non-square, non-integer, non-symmetric, or non-2 diagonal matrix
- **THEN** construction fails with `ValueError` and no object is created

### Requirement: Integer quadratic form

The system SHALL expose the quadratic form `q(x) = sum(x_i**2) + sum(a_ij * x_i * x_j for i < j)` and SHALL evaluate it exactly for integer vectors of the matrix dimension.

#### Scenario: Known values are exact

- **WHEN** the form of `((2, -1), (-1, 2))` is evaluated at `(1, 1)`
- **THEN** the result is the integer `1`

#### Scenario: Invalid vectors are rejected

- **WHEN** a vector has the wrong length or a non-integer entry
- **THEN** evaluation fails with `ValueError`

### Requirement: Exact inertia

The system SHALL report the number of positive, negative, and zero eigenvalues of a quasi-Cartan matrix using exact arithmetic, and MUST NOT rely on floating-point eigenvalue approximations.

#### Scenario: Positive definite Cartan matrix

- **WHEN** the inertia of the `A_2` Cartan matrix `((2, -1), (-1, 2))` is requested
- **THEN** the result is two positive, zero negative, and zero zero counts

#### Scenario: Degenerate affine matrix

- **WHEN** the inertia of the affine `A_2^(1)` matrix `((2, -1, -1), (-1, 2, -1), (-1, -1, 2))` is requested
- **THEN** the result is two positive, zero negative, and one zero count

#### Scenario: Indefinite matrix

- **WHEN** the inertia of `((2, 3), (3, 2))` is requested
- **THEN** the result is one positive, one negative, and zero zero counts

#### Scenario: Reordering indices preserves inertia

- **WHEN** a quasi-Cartan matrix is reordered by the same permutation of rows and columns
- **THEN** the reported inertia is unchanged

### Requirement: Inertia result invariants

The system SHALL expose `rank = positive + negative`, `signature = positive - negative`, `is_positive_definite`, `is_negative_definite`, and `is_degenerate` from an inertia result.

#### Scenario: Positive definite result reports its invariants

- **WHEN** the inertia is `(2, 0, 0)`
- **THEN** rank is 2, signature is 2, positive definiteness is true, negative definiteness is false, and degeneracy is false

#### Scenario: Degenerate result reports its invariants

- **WHEN** the inertia is `(2, 0, 1)`
- **THEN** rank is 2, signature is 2, positive definiteness is false, and degeneracy is true

### Requirement: Public surface

The system SHALL export `QuasiCartanMatrix`, `QuadraticForm`, and `Inertia` from the top-level `dynkinlab` package and SHALL list them in `__all__`.

#### Scenario: Imports succeed

- **WHEN** a caller imports `QuasiCartanMatrix`, `QuadraticForm`, and `Inertia` from `dynkinlab`
- **THEN** the imports succeed and the names appear in `dynkinlab.__all__`
