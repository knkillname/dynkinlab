# colab-compatibility Specification

## Purpose

Defines the pinned Google Colab runtime check that verifies DynkinLab against the environment used by Colab notebooks.

## Requirements

### Requirement: Pinned Colab runtime image

The Colab smoke gate MUST use the official `us-docker.pkg.dev/colab-images/public/cpu-runtime` image pinned by digest, not a mutable tag, and MUST run on `linux/amd64`.

#### Scenario: Digest is the source of truth

- **WHEN** the smoke configuration is inspected
- **THEN** it references the image by a `sha256:` digest
- **AND** it does not depend on the `latest` tag

#### Scenario: CPU image is sufficient

- **WHEN** the smoke gate runs
- **THEN** it uses the CPU runtime image and does not require a GPU

### Requirement: Colab environment assertions

The smoke gate MUST assert that the interpreter is Python 3.13 and that NumPy, NetworkX, and matplotlib import at or above the declared floors.

#### Scenario: Interpreter matches the target runtime

- **WHEN** the smoke gate starts the container
- **THEN** `sys.version_info[:2]` equals `(3, 13)`

#### Scenario: Visualization dependencies meet the floors

- **WHEN** the smoke gate imports the visualization dependencies
- **THEN** NetworkX is at least 3.7 and matplotlib is at least 3.10

#### Scenario: NumPy meets the floor

- **WHEN** the smoke gate imports the numerical dependency
- **THEN** NumPy is at least 2.1

### Requirement: Package import smoke

The smoke gate MUST import `dynkinlab` from the mounted source or an installed wheel and MUST fail when the import or packaged version lookup fails.

#### Scenario: Source tree imports in the Colab interpreter

- **WHEN** the smoke gate runs with `PYTHONPATH` pointing at `src`
- **THEN** `import dynkinlab` succeeds and `dynkinlab.__version__` is a non-empty string

### Requirement: Selective execution and CI coverage

The verification entry point MUST support skipping the Colab gate for fast local runs, while CI MUST run the gate in a dedicated job.

#### Scenario: Local verification can skip Colab

- **WHEN** `scripts/verify --skip-colab` runs
- **THEN** the host quality gates run and the Colab gate is reported as skipped

#### Scenario: CI runs the Colab gate

- **WHEN** CI executes the Colab compatibility job
- **THEN** it runs the pinned smoke gate without the skip option
