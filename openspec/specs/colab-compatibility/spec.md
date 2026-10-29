# colab-compatibility Specification

## Purpose

Defines the Colab runtime compatibility assertions that run inside the dev container.

## Requirements

### Requirement: Dev container runtime

The test environment MUST be the dev container built from the latest official Colab CPU runtime image. Test and verification steps MUST NOT download or start a separate container image.

#### Scenario: Dev container is the only image pull

- **WHEN** the dev container is created
- **THEN** it uses the official Colab CPU runtime image
- **AND** no test or verification step pulls or starts another image

### Requirement: Colab environment assertions

The verification suite MUST assert that the interpreter is Python 3.13 and that NumPy, NetworkX, and matplotlib import at or above the declared floors.

#### Scenario: Interpreter matches the target runtime

- **WHEN** the verification suite runs
- **THEN** `sys.version_info[:2]` equals `(3, 13)`

#### Scenario: Visualization dependencies meet the floors

- **WHEN** the verification suite imports the visualization dependencies
- **THEN** NetworkX is at least 3.7 and matplotlib is at least 3.10

#### Scenario: NumPy meets the floor

- **WHEN** the verification suite imports the numerical dependency
- **THEN** NumPy is at least 2.1

### Requirement: Package import smoke

The verification suite MUST import `dynkinlab` from the source tree and MUST fail when the import or packaged version lookup fails.

#### Scenario: Source tree imports in the Colab interpreter

- **WHEN** `hatch run colab:smoke` runs with `PYTHONPATH` pointing at `src`
- **THEN** `import dynkinlab` succeeds and `dynkinlab.__version__` is a non-empty string

### Requirement: Verification entry point

`hatch run verify` MUST run the host checks, the test suite, and the Colab environment assertions, and MUST be run inside the dev container.

#### Scenario: Full verification runs inside the dev container

- **WHEN** `hatch run verify` runs inside the dev container
- **THEN** the host checks, tests, and Colab assertions run
- **AND** no additional image is downloaded
