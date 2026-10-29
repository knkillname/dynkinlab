# Spec Delta

## MODIFIED Requirements

### Requirement: Supported Python and dependency floors

The package MUST require Python 3.13 or newer and MUST declare `numpy>=2.1`, `networkx>=3.7`, and `matplotlib>=3.10` as runtime dependencies.

#### Scenario: Installation on the supported interpreter

- **WHEN** the package is installed in a Python 3.13 environment
- **THEN** installation succeeds and `import dynkinlab` exposes a version string

#### Scenario: Unsupported interpreter is rejected

- **WHEN** installation is attempted on Python 3.12 or older
- **THEN** the package metadata rejects it because `Requires-Python` is `>=3.13`

#### Scenario: Dependency floors are declared

- **WHEN** the installed distribution metadata is inspected
- **THEN** it declares `numpy>=2.1`, `networkx>=3.7`, and `matplotlib>=3.10`
