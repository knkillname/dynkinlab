# Spec Delta

## ADDED Requirements

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
