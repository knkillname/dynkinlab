# Spec Delta

## ADDED Requirements

### Requirement: Non-symmetrizability certificate

When construction or canonical symmetrization fails because no positive diagonal integer symmetrizer exists, the system SHALL raise `NotSymmetrizableError`, a `ValueError` subclass. The error SHALL expose a `SymmetrizerCertificate` through a public `certificate` attribute. The certificate SHALL identify one of `zero_entry`, `opposite_signs`, or `inconsistent_cycle`, and its `verify(entries)` method SHALL return true for the rejected matrix.

#### Scenario: Zero-entry obstruction

- **WHEN** a matrix has an edge with `a_ij = 0` and `a_ji != 0`
- **THEN** construction raises `NotSymmetrizableError` whose certificate has kind `zero_entry` and verifies against the matrix

#### Scenario: Opposite-sign obstruction

- **WHEN** a matrix has an edge with `a_ij * a_ji < 0`
- **THEN** construction raises `NotSymmetrizableError` whose certificate has kind `opposite_signs` and verifies against the matrix

#### Scenario: Inconsistent-cycle obstruction

- **WHEN** a matrix has an edge-ratio product different from `1` around a cycle
- **THEN** construction raises `NotSymmetrizableError` whose certificate has kind `inconsistent_cycle`, names that cycle, and verifies against the matrix

#### Scenario: Existing error compatibility

- **WHEN** a caller catches the symmetrizability failure as `ValueError`
- **THEN** the catch succeeds and the message still contains `symmetrizable`

### Requirement: Validation failures remain distinct

Construction failures caused by invalid shape, non-integer entries, a non-2 diagonal, or an invalid supplied symmetrizer SHALL remain ordinary `ValueError` instances and SHALL NOT be reported as `NotSymmetrizableError`.

#### Scenario: Invalid input has no nonexistence certificate

- **WHEN** construction rejects an empty, ragged, non-square, non-integer, or non-2-diagonal matrix
- **THEN** the raised error is a `ValueError` that is not a `NotSymmetrizableError`

## MODIFIED Requirements

### Requirement: Public surface

The system SHALL export `QuasiCartanMatrix`, `QuadraticForm`, `Inertia`, `NotSymmetrizableError`, `SymmetrizerCertificate`, and `SymmetrizerObstruction` from the top-level `dynkinlab` package and SHALL list them in `dynkinlab.__all__`.

#### Scenario: Imports succeed

- **WHEN** a caller imports `QuasiCartanMatrix`, `QuadraticForm`, `Inertia`, `NotSymmetrizableError`, `SymmetrizerCertificate`, and `SymmetrizerObstruction` from `dynkinlab`
- **THEN** the imports succeed and the names appear in `dynkinlab.__all__`
