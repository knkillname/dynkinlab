# Spec Delta

## MODIFIED Requirements

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
