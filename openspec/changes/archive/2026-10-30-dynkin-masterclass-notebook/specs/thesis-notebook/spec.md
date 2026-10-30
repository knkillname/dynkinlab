# Spec Delta

## REMOVED Requirements

### Requirement: Script-first source format

**Reason**: The thesis-companion artifact is replaced by the masterclass, which has new source and
artifact paths.

**Migration**: Use `notebooks/dynkin-masterclass.py` as the editable source and regenerate
`notebooks/dynkin-masterclass.ipynb`; the thesis-companion files are removed.

### Requirement: Executable paper-like notebook

**Reason**: The paper-companion format is replaced by a ground-up pedagogical masterclass with a
fixed tour and parameter labs.

**Migration**: Use the `dynkin-masterclass` capability's executable-masterclass requirements.

### Requirement: Focused scope

**Reason**: The masterclass deliberately adds ground-up definitions, finite-type context, and
interactive labs rather than limiting itself to the old A/D-only paper scope.

**Migration**: Use the `dynkin-masterclass` capability's pedagogical-structure and A/D case-study
requirements.

### Requirement: Matrix-first presentation

**Reason**: Matrix-first presentation is retained and strengthened in the new capability.

**Migration**: Use the `dynkin-masterclass` matrix-first requirement.

### Requirement: A-type worked example

**Reason**: The A9 case study is retained under the new capability.

**Migration**: Use the `dynkin-masterclass` A-type A9 case-study requirement.

### Requirement: D-core gluing worked example

**Reason**: The D13 case study is retained under the new capability.

**Migration**: Use the `dynkin-masterclass` D-core gluing D13 case-study requirement.

### Requirement: A-cyclization worked example

**Reason**: The A9-to-D8 case study is retained under the new capability.

**Migration**: Use the `dynkin-masterclass` A-cyclization A9-to-D8 case-study requirement.

### Requirement: Flation example

**Reason**: The A3 flation case study is retained under the new capability.

**Migration**: Use the `dynkin-masterclass` flation case-study requirement.

### Requirement: Boundary table

**Reason**: The boundary table is retained but reframed around the classifier oracle and structural
certificates rather than a thesis-specific algorithm.

**Migration**: Use the `dynkin-masterclass` boundary-table requirement.

### Requirement: Latest-version Colab setup

**Reason**: The latest-version git install policy is retained for the masterclass artifact.

**Migration**: Use the `dynkin-masterclass` latest-version Colab setup requirement with the new
notebook path.

### Requirement: Notebook smoke execution

**Reason**: The smoke test is retained and extended with code-budget, no-definition, and
no-coordinate-table guards.

**Migration**: Use the `dynkin-masterclass` smoke-execution and artifact-sync requirement.

### Requirement: Deterministic visualizations

**Reason**: Fixed or manually chosen positions are replaced by NetworkX-computed layouts with no
stored coordinate tables.

**Migration**: Use the `dynkin-masterclass` NetworkX-driven visualizations requirement and the
`colab-display` layout and drawing requirements.

### Requirement: No public surface changes

**Reason**: The replacement deliberately adds notebook-facing methods to existing public values,
although it keeps `__all__` and runtime dependencies unchanged.

**Migration**: Use the `dynkin-masterclass` package-surface-stability requirement.

### Requirement: Notebook guide and docs

**Reason**: The documentation contract moves to the masterclass name and paths.

**Migration**: Use the `dynkin-masterclass` notebook-guide and docs requirement.
