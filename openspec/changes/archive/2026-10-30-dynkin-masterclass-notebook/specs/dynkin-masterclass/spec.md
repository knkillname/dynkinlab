# Spec Delta

## Purpose

Defines the script-first, executable, ground-up masterclass notebook that teaches quasi-Cartan
matrices, flation, and the A/D structural classification with computed certificates and
NetworkX-driven figures.

## ADDED Requirements

### Requirement: Script-first masterclass source

The repository SHALL keep `notebooks/dynkin-masterclass.py` as the editable percent-format source
and SHALL generate `notebooks/dynkin-masterclass.ipynb` from it. The source SHALL use `# %%` for
code cells and `# %% [markdown]` for Markdown cells, and its cell types and normalized sources
SHALL match the generated notebook.

#### Scenario: Cell markers parse

- **WHEN** the Python source is parsed with the percent-format notebook reader
- **THEN** it yields the expected code and Markdown cells

#### Scenario: Paired notebook matches

- **WHEN** the source and the generated notebook are compared
- **THEN** their cell types and normalized cell sources match

#### Scenario: Colab artifact exists

- **WHEN** the notebook is opened from the repository in Colab
- **THEN** the generated `notebooks/dynkin-masterclass.ipynb` is available

### Requirement: Executable masterclass

The repository SHALL provide the generated notebook as an English masterclass that runs end-to-end
from a clean kernel. It SHALL include a short orientation, numbered sections, definitions and
propositions, computed examples, figure captions, exercises or labs, a boundary table, and
references.

#### Scenario: Clean-kernel execution

- **WHEN** the notebook is executed top to bottom
- **THEN** every code cell succeeds and the final cell reports that all certificates verified

#### Scenario: Committed outputs

- **WHEN** the notebook is viewed on GitHub or Colab before execution
- **THEN** the worked figures and tables are already present

#### Scenario: Pedagogical structure

- **WHEN** the Markdown is read without the code
- **THEN** it defines each object before using it and presents the A/D results as a self-contained
  lesson rather than an API tour

### Requirement: Ground-up independence

The notebook SHALL teach the material from first principles and SHALL NOT present itself as a
thesis companion. It SHALL NOT depend on thesis-specific notation, terminology, or algorithm
comparisons; the path notation SHALL be presented as DynkinLab notation and the classifier SHALL
be presented as an independent bounded-flation oracle.

#### Scenario: No thesis framing

- **WHEN** the notebook title and prose are inspected
- **THEN** they describe a masterclass rather than a thesis companion

#### Scenario: Independent oracle

- **WHEN** the classification section is read
- **THEN** it explains the bounded-flation oracle without comparing it to a thesis-specific
  polynomial algorithm

#### Scenario: Definitions before use

- **WHEN** the notebook introduces quasi-Cartan matrices, inertia, flation, or Dynkin types
- **THEN** the term is defined before the first computed example uses it

### Requirement: Matrix-first presentation

The notebook SHALL present labeled quasi-Cartan matrices as the primary objects, SHALL display row
and column labels for every worked matrix, and SHALL draw a bigraph only when the matrix admits a
faithful representation. It SHALL state the entry condition that determines drawability.

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

### Requirement: A-type A9 case study

The notebook SHALL construct A9 from path notation, compute and verify its A-type certificate,
display its four blocks, partitions, and articulation points, and draw its block-cut tree with a
NetworkX bipartite layout.

#### Scenario: A9 certificate

- **WHEN** the A9 example is evaluated
- **THEN** the A-type certificate verifies and the reported type is A9

#### Scenario: Block structure

- **WHEN** the block-cut tree is displayed
- **THEN** the blocks are {1,2,3}, {3,4,5,9}, {4,7}, and {5,6,8}, and the articulation points are
  3, 4, and 5

#### Scenario: Deterministic tree

- **WHEN** the notebook is executed twice in the same environment
- **THEN** the block-cut tree layout places every node identically

### Requirement: D-core gluing D13 case study

The notebook SHALL build D13 by gluing the five A-type pieces onto the odd five-cycle D-core,
verify the gluing certificate, display the piece table, and draw the result with the core
highlighted.

#### Scenario: D13 certificate

- **WHEN** the gluing is evaluated
- **THEN** the certificate verifies and the reported type is D13

#### Scenario: Piece table

- **WHEN** the construction is displayed
- **THEN** the five pieces, their labels, their A-types, and their core edges are listed

#### Scenario: Core highlight

- **WHEN** the D13 figure is displayed
- **THEN** the five core edges are visually distinguished from the piece edges

### Requirement: A-cyclization A9 to D8 case study

The notebook SHALL cyclize A9 at internal vertices 1 and 6, verify the certificate, report the
shortest path `(1, 3, 5, 6)`, and draw the D8 result with the identified vertex labeled `1=6`.

#### Scenario: Cyclization certificate

- **WHEN** the cyclization is evaluated
- **THEN** the certificate verifies and the result type is D8

#### Scenario: Shortest path

- **WHEN** the construction is displayed
- **THEN** the path is reported as `(1, 3, 5, 6)` and shown on the A9 figure

#### Scenario: D8 result

- **WHEN** the D8 figure is displayed
- **THEN** the identified vertex is labeled `1=6` and the result has 8 vertices

#### Scenario: Independent D8 layout

- **WHEN** the D8 figure is displayed
- **THEN** its coordinates come from its own layout rather than the A9 positions

### Requirement: Flation case study

The notebook SHALL flate A3 from vertex 1 to vertex 2, display the before and after matrices and
figures, and verify that inertia and Dynkin type are preserved.

#### Scenario: Flation invariance

- **WHEN** A3 is flated from vertex 1 to vertex 2
- **THEN** the inertia and classified type remain A3

#### Scenario: Visual comparison

- **WHEN** the flation is displayed
- **THEN** both matrices and both bigraphs are shown side by side

#### Scenario: Determinism

- **WHEN** the notebook is re-executed in the same environment
- **THEN** the flation result and both figures are unchanged

### Requirement: Boundary table

The notebook SHALL end with a boundary table containing A9, D13, D8, E6, mixed B6, affine A2, and
the `2/4` matrix. Each row SHALL report inertia, classifier outcome, faithful-bigraph availability,
and certificate kind.

#### Scenario: Rows and statuses

- **WHEN** the table is displayed
- **THEN** A9, D13, D8, E6, and mixed B6 are classified with their types, affine A2 is
  `NOT_POSITIVE_DEFINITE`, and the `2/4` case is `NOT_ADMISSIBLE`

#### Scenario: Certificate column

- **WHEN** the table is read
- **THEN** A9, D13, and D8 name their structural certificates while the other rows show none

#### Scenario: Oracle caveat

- **WHEN** the algorithm section is read
- **THEN** it distinguishes the bounded-flation oracle from the structural certificates without
  referring to a thesis-specific algorithm

### Requirement: Literary code budget

Every code cell in the source SHALL contain at most 40 physical lines. No code cell SHALL define a
function or class. Every code cell SHALL be preceded by Markdown that explains its purpose, and
reusable behavior SHALL live in the public DynkinLab API rather than the notebook.

#### Scenario: Line cap

- **WHEN** the source is parsed into code cells
- **THEN** every code cell has at most 40 lines

#### Scenario: No local definitions

- **WHEN** each code cell is parsed as Python
- **THEN** it contains no function or class definition

#### Scenario: Explanation precedes code

- **WHEN** a code cell is inspected
- **THEN** the preceding cell is Markdown that states the claim or construction being demonstrated

### Requirement: NetworkX-driven visualizations

The notebook SHALL NOT store vertex coordinates or use manual position dictionaries. Every figure
layout SHALL be computed by NetworkX through the public drawing API, with a deterministic layout or
seeded algorithm, and repeated executions in the same environment SHALL produce the same node
placement. Layout distances SHALL use unweighted adjacency, not the signed solid, dotted, or arrow
weights.

#### Scenario: No hardcoded coordinates

- **WHEN** the notebook source is inspected
- **THEN** it contains no coordinate table or literal vertex-position dictionary

#### Scenario: Repeatable layout

- **WHEN** the notebook is executed twice in the same environment
- **THEN** every figure has the same node and edge placement

#### Scenario: Edge style does not drive layout

- **WHEN** the notebook lays out a graph containing both solid and dotted edges
- **THEN** the positions depend on unweighted adjacency rather than the signed edge weights

#### Scenario: Layout lab

- **WHEN** a reader changes the layout name
- **THEN** the same graph is redrawn with the newly selected NetworkX layout

### Requirement: Interactive fixed tour and labs

The notebook SHALL provide a fixed canonical tour that runs without edits and a set of parameter
labs. Labs SHALL expose path text, flation endpoints, cyclization endpoints, and layout choice, and
SHALL verify invariants rather than only one hardcoded result.

#### Scenario: Fixed tour

- **WHEN** the notebook is executed with its committed parameters
- **THEN** the canonical A9, D13, D8, and A3 examples verify

#### Scenario: Parameter lab

- **WHEN** a reader changes a flation or cyclization endpoint and reruns the lab
- **THEN** the notebook recomputes the result without requiring a code edit outside the parameter
  cell

#### Scenario: Invariant checks

- **WHEN** a lab uses a different valid input
- **THEN** its checks assert preserved invariants or certificate validity rather than the original
  hardcoded type

### Requirement: Latest-version Colab setup

The notebook SHALL install the latest DynkinLab from the project's git default branch when running
in Colab and SHALL NOT install from PyPI, a commit SHA, a tag, or a package version. In a local
checkout it SHALL skip installation. The final cell SHALL report the package version.

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

### Requirement: Notebook smoke execution and artifact sync

`hatch test` SHALL parse the masterclass source, compare its cell types and normalized sources
with the generated notebook, and execute it in a clean kernel. The smoke test SHALL also enforce
the 40-line limit, the no-definition rule, and the no-coordinate-table rule.

#### Scenario: Smoke test passes

- **WHEN** the test suite runs
- **THEN** the source parses into cells, the notebook executes successfully, and the final
  assertions pass

#### Scenario: Broken cell fails

- **WHEN** a source cell raises
- **THEN** the notebook smoke test fails

#### Scenario: Drift fails

- **WHEN** the generated notebook does not match the source
- **THEN** the synchronization test fails

#### Scenario: Structural guard fails

- **WHEN** a code cell exceeds 40 lines, defines a function or class, or a coordinate table is
  introduced
- **THEN** the notebook test fails with a diagnostic naming the cell

### Requirement: Package surface stability

The masterclass and its notebook-facing API SHALL NOT add a runtime dependency or a new top-level
`dynkinlab` export. The package `__all__`, runtime dependency floors, and distribution metadata
SHALL remain unchanged while existing value classes gain notebook-facing methods.

#### Scenario: No new dependency

- **WHEN** the package dependencies are inspected after the change
- **THEN** they are still NumPy, NetworkX, and matplotlib with their existing floors

#### Scenario: No new top-level names

- **WHEN** `dynkinlab.__all__` is inspected
- **THEN** it contains the same names as before the change

#### Scenario: Existing values gain methods

- **WHEN** the notebook-facing API is used
- **THEN** the behavior is reachable through existing public values rather than a new exported class

### Requirement: Notebook guide and docs

The repository SHALL document the masterclass source, generation, Colab workflow, and smoke test
in `notebooks/README.md`; update the quickstart pointer and API reference; and record the change in
`CHANGELOG.md`.

#### Scenario: Notebook guide

- **WHEN** a reader opens `notebooks/README.md`
- **THEN** it explains editing the `.py` source, regenerating the `.ipynb`, the latest git install,
  how to run the notebook, and the smoke test

#### Scenario: Docs consistency

- **WHEN** `hatch run docs:build` runs
- **THEN** the quickstart and API reference contain no stale thesis-companion or planned-notebook
  statement

#### Scenario: Changelog

- **WHEN** the changelog is read
- **THEN** the masterclass and notebook-facing API changes appear under `[Unreleased]`
