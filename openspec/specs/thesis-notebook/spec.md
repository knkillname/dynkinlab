# thesis-notebook Specification

## Purpose

Defines the script-first, executable, paper-like thesis companion notebook and the guarantees that
keep its computed examples, figures, and Colab setup reproducible.

## Requirements

### Requirement: Script-first source format

The repository SHALL keep `notebooks/thesis-companion.py` as the editable source. It SHALL use
`# %%` for code cells and `# %% [markdown]` for Markdown cells, and SHALL parse into the same cell
sequence as the generated `notebooks/thesis-companion.ipynb`.

#### Scenario: Cell markers parse

- **WHEN** the Python source is parsed with the percent-format notebook reader
- **THEN** it yields the expected code and Markdown cells

#### Scenario: Paired notebook matches

- **WHEN** the source and the generated notebook are compared
- **THEN** their cell types and normalized cell sources match

#### Scenario: Colab artifact exists

- **WHEN** the notebook is opened from the repository in Colab
- **THEN** the generated `notebooks/thesis-companion.ipynb` is available

### Requirement: Executable paper-like notebook

The repository SHALL provide `notebooks/thesis-companion.ipynb`, generated from the Python source,
as an English notebook that reads as a short computational paper with a title, abstract, numbered
sections, theorem statements, figure captions, a boundary table, and references. It SHALL run
end-to-end from a clean kernel without manual edits.

#### Scenario: Clean-kernel execution

- **WHEN** the notebook is executed top to bottom
- **THEN** every code cell succeeds and the final cell reports that all certificates verified

#### Scenario: Paper structure

- **WHEN** the Markdown is read without the code
- **THEN** it presents the A/D results as a self-contained narrative rather than an API tutorial

#### Scenario: Committed outputs

- **WHEN** the notebook is viewed on GitHub or Colab before execution
- **THEN** the worked figures and tables are already present

### Requirement: Focused scope

The notebook SHALL limit its mathematical content to the A/D structural results, the flation
example, and the boundary table. It SHALL NOT include full proofs, a full thesis translation, a
reimplementation of the classifier, or a library API tutorial.

#### Scenario: Structural focus

- **WHEN** the notebook sections are inspected
- **THEN** the worked examples are A9, D13, the A9-to-D8 cyclization, and the A3 flation

#### Scenario: No classifier reimplementation

- **WHEN** the algorithm section is read
- **THEN** it presents the thesis's algorithm at a high level and uses the existing classifier as an
  oracle rather than implementing a second classifier

### Requirement: Matrix-first presentation

The notebook SHALL present labeled quasi-Cartan matrices as the primary objects, with row and
column labels shown for each worked matrix. It SHALL draw a bigraph only when the matrix admits a
faithful bigraph, and SHALL state the entry condition that determines drawability.

#### Scenario: Labeled matrix

- **WHEN** a worked matrix is displayed
- **THEN** its rows and columns carry the vertex labels used by the accompanying figure

#### Scenario: Faithful drawing

- **WHEN** a matrix is in the drawable range
- **THEN** the notebook shows both the matrix and its bigraph figure

#### Scenario: Matrix-only case

- **WHEN** the `2/4` matrix is evaluated
- **THEN** the notebook reports that no faithful bigraph exists and shows only the matrix, its
  exact inertia, and its classifier status

### Requirement: A-type worked example

The notebook SHALL compute the A-type certificate for the 9-vertex A9 matrix, display its four
blocks and articulation points, and verify the certificate. It SHALL include a deterministic
block-cut tree visualization.

#### Scenario: A9 certificate

- **WHEN** the A9 example is evaluated
- **THEN** the A-type certificate verifies and the reported type is A9

#### Scenario: Block structure

- **WHEN** the block-cut tree is displayed
- **THEN** the blocks are {1,2,3}, {3,4,5,9}, {4,7}, and {5,6,8}, and the articulation points are
  3, 4, and 5

#### Scenario: Deterministic tree

- **WHEN** the notebook is executed twice
- **THEN** the A9 graph and block-cut tree layouts are identical

### Requirement: D-core gluing worked example

The notebook SHALL build the D13 D-core gluing from the five A-type pieces and the odd five-cycle
core, verify the certificate, and display the resulting D13 graph with the core highlighted.

#### Scenario: D13 certificate

- **WHEN** the gluing is evaluated
- **THEN** the certificate verifies and the reported type is D13

#### Scenario: Piece table

- **WHEN** the construction is displayed
- **THEN** the five pieces, their labels, their A-types, and their core edges are listed

#### Scenario: Core highlight

- **WHEN** the D13 figure is displayed
- **THEN** the five core edges are visually distinguished from the piece edges

### Requirement: A-cyclization worked example

The notebook SHALL cyclize the A9 example at internal vertices 1 and 6, verify the resulting
certificate, and display the unique shortest path 1-3-5-6 and the D8 result.

#### Scenario: Cyclization certificate

- **WHEN** the cyclization is evaluated
- **THEN** the certificate verifies and the result type is D8

#### Scenario: Shortest path

- **WHEN** the construction is displayed
- **THEN** the path is reported as (1, 3, 5, 6) and shown on the A9 figure

#### Scenario: D8 result

- **WHEN** the D8 figure is displayed
- **THEN** the identified vertex is labeled 1=6 and the result has 8 vertices

### Requirement: Flation example

The notebook SHALL compute a flation of the A3 example, display the before and after labeled
matrices, and verify that inertia and Dynkin type are preserved.

#### Scenario: Flation invariance

- **WHEN** A3 is flated from vertex 1 to vertex 2
- **THEN** the inertia and classified type remain A3

#### Scenario: Visual comparison

- **WHEN** the flation is displayed
- **THEN** both matrices and both bigraphs are shown side by side

#### Scenario: Determinism

- **WHEN** the notebook is re-executed
- **THEN** the flation result is unchanged

### Requirement: Boundary table

The notebook SHALL end with a boundary table containing A9, D13, D8, E6, the mixed B6 example,
affine A2, and a matrix-only `2/4` matrix. Each row SHALL report inertia, classifier status,
bigraph availability, and certificate kind. The notebook SHALL state that `classify()` is an
independent bounded-flation oracle, not the thesis's polynomial algorithm.

#### Scenario: Rows and statuses

- **WHEN** the table is displayed
- **THEN** A9, D13, D8, E6, and mixed B6 are classified with their types, affine A2 is
  `NOT_POSITIVE_DEFINITE`, and the `2/4` case is `NOT_ADMISSIBLE`

#### Scenario: Certificate column

- **WHEN** the table is read
- **THEN** A9, D13, and D8 name their structural certificates while the other rows show none

#### Scenario: Oracle caveat

- **WHEN** the algorithm section is read
- **THEN** it distinguishes the bounded-flation oracle from the thesis's Bareiss plus structural
  algorithm

### Requirement: Latest-version Colab setup

The notebook SHALL install the latest DynkinLab from the project's git default branch when running
in Colab and SHALL NOT install from PyPI. It SHALL NOT select a commit SHA, tag, or package version.
In a local checkout it SHALL skip installation and use the installed package. The final cell SHALL
report the package version.

#### Scenario: Colab install

- **WHEN** the notebook runs in Colab
- **THEN** the setup cell installs from the default branch before importing DynkinLab

#### Scenario: Local checkout

- **WHEN** the notebook runs outside Colab
- **THEN** the setup cell performs no installation and imports the local package

#### Scenario: No version pin

- **WHEN** the setup cell and the notebook guide are inspected
- **THEN** neither selects a commit SHA, tag, or dependency version

#### Scenario: No PyPI

- **WHEN** the notebook and its guide are inspected
- **THEN** neither instructs the reader to install `dynkinlab` from PyPI

### Requirement: Notebook smoke execution

`hatch test` SHALL parse `notebooks/thesis-companion.py` into notebook cells and execute them in a
clean kernel, failing if any cell raises. The test-only environment SHALL declare the notebook
execution dependencies, and the check SHALL require no manual notebook step.

#### Scenario: Smoke test passes

- **WHEN** the test suite runs
- **THEN** the source parses into cells, the notebook executes successfully, and the final
  assertions pass

#### Scenario: Broken cell fails

- **WHEN** a source cell raises
- **THEN** the notebook smoke test fails

#### Scenario: No manual step

- **WHEN** the smoke test starts
- **THEN** it parses and executes the committed Python source without pre-running cells

#### Scenario: Source and artifact stay in sync

- **WHEN** the smoke test reads the generated notebook
- **THEN** its cell types and normalized sources match the Python source

### Requirement: Deterministic visualizations

The notebook's figures SHALL use NetworkX for graph structure and deterministic positions or a
fixed seed, so repeated executions produce the same layout. The notebook SHALL NOT add a runtime
dependency beyond the package's declared dependencies.

#### Scenario: Repeatable layout

- **WHEN** the notebook is executed twice
- **THEN** every figure has the same node and edge placement

#### Scenario: Declared dependencies

- **WHEN** the notebook is installed in Colab
- **THEN** it uses NumPy, NetworkX, matplotlib, and DynkinLab only

### Requirement: No public surface changes

The notebook SHALL keep its path notation, matrix display, drawability predicate, and drawing
helpers local to the notebook. The package's public names, `__all__`, wheel contents, and runtime
dependency floors SHALL remain unchanged.

#### Scenario: Package surface

- **WHEN** the notebook change is complete
- **THEN** `dynkinlab.__all__` and the wheel contents are unchanged

#### Scenario: Local helpers

- **WHEN** the notebook is inspected
- **THEN** the notation and drawing helpers are defined in the notebook, not imported from a new
  package module

### Requirement: Notebook guide and docs

The repository SHALL document the script-first source, the generated notebook, the latest-version
Colab workflow, and the local smoke test in `notebooks/README.md`; update the quickstart's notebook
pointer and the API reference's planned-surface note; and record the notebook in `CHANGELOG.md`.

#### Scenario: Notebook guide

- **WHEN** a reader opens `notebooks/README.md`
- **THEN** it explains editing the `.py` source, regenerating the `.ipynb`, the latest git install,
  how to run the notebook, and the smoke test

#### Scenario: Docs consistency

- **WHEN** `hatch run docs:build` runs
- **THEN** the quickstart and API reference contain no stale planned-notebook statement

#### Scenario: Changelog

- **WHEN** the changelog is read
- **THEN** the notebook appears under `[Unreleased]`
