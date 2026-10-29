# Spec Delta

## Purpose

Defines the installable typed package foundation and the quality gates that every DynkinLab change must satisfy.

## ADDED Requirements

### Requirement: Supported Python and dependency floors

The package MUST require Python 3.13 or newer and MUST declare `networkx>=3.7` and `matplotlib>=3.10` as runtime dependencies.

#### Scenario: Installation on the supported interpreter

- **WHEN** the package is installed in a Python 3.13 environment
- **THEN** installation succeeds and `import dynkinlab` exposes a version string

#### Scenario: Unsupported interpreter is rejected

- **WHEN** installation is attempted on Python 3.12 or older
- **THEN** the package metadata rejects it because `Requires-Python` is `>=3.13`

#### Scenario: Dependency floors are declared

- **WHEN** the installed distribution metadata is inspected
- **THEN** it declares `networkx>=3.7` and `matplotlib>=3.10`

### Requirement: Typed distribution

The built wheel MUST include a `py.typed` marker and a `LICENSE` file, and the package MUST import from the wheel without the source tree present.

#### Scenario: Wheel installs in a clean environment

- **WHEN** the wheel is installed into an empty virtual environment
- **THEN** `import dynkinlab` succeeds and reports the packaged version

#### Scenario: Wheel contains typing and license markers

- **WHEN** the wheel contents are inspected
- **THEN** `dynkinlab/py.typed` and `LICENSE` are present

### Requirement: Hatch quality gates

The project MUST use `hatch check --fix` as the single formatting, linting, and type-checking command and `hatch test` as the pytest command. No separate formatter or parallel lint/type workflow SHALL be required.

#### Scenario: Check is idempotent

- **WHEN** `hatch check --fix` runs twice on an unchanged tree
- **THEN** both runs exit successfully and the second run leaves files unchanged

#### Scenario: Tests enforce coverage

- **WHEN** `hatch test` runs the Python 3.13 environment
- **THEN** pytest executes and the coverage gate of at least 80% is enforced

#### Scenario: CI uses the same gates

- **WHEN** CI checks a change
- **THEN** it runs `hatch check` without `--fix` and `hatch test`
