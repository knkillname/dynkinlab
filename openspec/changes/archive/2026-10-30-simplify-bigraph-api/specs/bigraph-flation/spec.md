# Spec Delta

## MODIFIED Requirements

### Requirement: Path notation construction

The system SHALL provide `Bigraph.from_paths(*paths, labels=None, arrow_multiplicity=2)` to
construct a labeled bigraph from one or more path-notation blocks joined with newlines. It SHALL
accept the existing label, comment, separator, isolated-vertex, and explicit-order forms; `labels`
and `arrow_multiplicity` SHALL be keyword-only. Conflicts, non-string arguments, and out-of-order
labels SHALL raise `ValueError`.

#### Scenario: A9 path text is parsed

- **WHEN** the A9 path text and labels `1..9` are passed to `from_paths`
- **THEN** the resulting bigraph has the expected matrix and labels in first-appearance order

#### Scenario: Multiple path arguments are joined

- **WHEN** the same path blocks are passed as separate arguments or as one newline-separated string
- **THEN** both calls produce the same labels and matrix

#### Scenario: Labels are keyword-only

- **WHEN** a label sequence is passed as a positional path argument
- **THEN** construction fails with `ValueError` instead of treating the sequence as a path block

#### Scenario: Non-string path arguments are rejected

- **WHEN** a path argument is not a string
- **THEN** construction fails with `ValueError`

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

- **WHEN** the path arguments are empty without pinned labels or contain a malformed path
- **THEN** construction fails with `ValueError`
