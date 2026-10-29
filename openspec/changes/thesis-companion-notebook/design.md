# Design

## Context

See `proposal.md` for motivation and `specs/thesis-notebook/spec.md` for the behavioral contract.

The library already provides the algebraic and structural surface the notebook needs:
`QuasiCartanMatrix`, `Inertia`, `Bigraph`, the A-type certificate, `EdgeMultiset`, `DCore`,
`DCoreGluingCertificate`, `ACyclizationCertificate`, and the exact classifier. `QuasiCartanMatrix`
is the central value; `Bigraph` is a labeled incidence view over it. The notebook must preserve that
hierarchy rather than presenting the drawing as the object.

The repository currently has only `notebooks/README.md`, which still describes a future quickstart
and a PyPI install. There is no notebook source, no generated notebook, no notebook execution test,
and no Colab entry point. The notebook will be authored as a percent-format Python script and
generated into a Colab notebook. It will track the project's default branch, so the latest pushed
`main` is what Colab installs; the feature commits are pushed with this planning change.

Constraints that shape the design:

- Exact arithmetic stays in the library; the notebook does not reimplement classification.
- The notebook is a repository artifact, not part of the wheel or the public API.
- The library must stay IPython-free; display and drawing helpers in the notebook may use IPython.
- Runtime dependencies remain NumPy, NetworkX, and matplotlib. Notebook execution dependencies are
  test-only.
- The Sphinx build is strict; any docs pointer must resolve without warnings.
- The notebook must remain brief: the thesis narrative and the algorithms belong in Markdown and
  small cells, not in a second library.

## Goals / Non-Goals

**Goals:**

- A single English notebook that reads as a short computational paper and runs from a clean kernel.
- A script-first source with `# %%` cell markers and a generated Colab notebook that stays in sync.
- Matrix-first displays with correct row and column labels, including the non-contiguous D8 labels.
- Deterministic figures for A9, D13, D8, and the A3 flation example.
- Both D certificates and the A-type certificate verified in the notebook.
- A boundary table that separates structural certificates, classifier-only cases, and matrix-only
  cases.
- A latest-version Colab install and a smoke test integrated with `hatch test`.
- Local helpers only; no new public names or runtime dependencies.

**Non-Goals:**

- Fixing `Bigraph.edges` for the lossy `2/4` case. That is a separate library change.
- Making labels first-class on `QuasiCartanMatrix`.
- Implementing the thesis's Bareiss plus structural polynomial algorithm.
- Translating the full thesis, including proofs and all appendices.
- Adding the notebook to the Sphinx site.

## Decisions

### 1. Notebook shape and narrative

`notebooks/thesis-companion.py` is the editable source and uses seven numbered sections; it is
generated into `notebooks/thesis-companion.ipynb` for Colab:

1. Setup and notation.
2. Labeled matrices and the drawable range.
3. Flation and inertia.
4. A-type: A9 and its block-cut tree.
5. D-core gluing: D13.
6. A-cyclization: A9 to D8.
7. The algorithm and the boundary table.

The setup section contains the latest-version Colab cell, imports, display defaults, the path
parser, the matrix table helper, the drawability predicate, and the drawing helpers. These are
marked as infrastructure. The mathematical sections contain one-line constructions and a short
verification cell each. The final cell asserts every certificate and prints the package version.

#### Source format

The source uses the Jupytext percent format: `# %%` starts a code cell and `# %% [markdown]` starts
a Markdown cell whose lines are prefixed with `# `. The source is normal Python otherwise; it
contains no IPython magics, so it remains valid Python as well as a parseable notebook source. The
generated `.ipynb` carries the Colab artifact and its executed outputs; the `.py` remains the
editable source of truth.

Generation uses Jupytext's notebook reader/writer or the equivalent CLI. Jupytext is a test-only
dependency; it is not imported by the library and does not enter the wheel. The smoke test reads
the source, compares its cell types and normalized sources with the committed `.ipynb`, and
executes the source-derived notebook. This catches drift between the two artifacts.

The generated notebook is committed with its computed outputs so it reads as a paper on GitHub or
Colab without re-execution. The smoke test re-executes the source from a clean kernel, so the
outputs cannot silently go stale.

Alternative considered: edit the `.ipynb` JSON directly. Rejected because the cell-marker source is
easier to review and diff. Alternative considered: a custom parser for a bespoke comment syntax.
Rejected because the percent format is already supported by Jupytext, VS Code, and other notebook
tooling.

### 2. Notebook-local path notation

The notebook defines one local function of the form `graph(text, labels=None)` returning a
`Bigraph`. Each non-empty line is a path; `#` starts a comment. The grammar is:

```text
graph  ::= path (NEWLINE path)*
path   ::= vertex (edgeop vertex)+
edgeop ::= "--" | ".." | "->" | "<-" | ".>" | "<."
vertex ::= integer | bare identifier
```

Whitespace and commas separate tokens. Labels default to first appearance; an explicit `labels`
argument pins matrix order and rejects paths that introduce labels outside it.

The operator mapping follows the thesis's mixed-bigraph convention:

| operator | matrix entries |
|---|---|
| `--` | both directions `-1` |
| `..` | both directions `+1` |
| `->` | arrow direction `-1`, reverse `-2` |
| `<-` | arrow direction `-1`, reverse `-2` |
| `.>` | arrow direction `+1`, reverse `+2` |
| `<.` | arrow direction `+1`, reverse `+2` |

The reverse magnitude is fixed at 2 because every arrow in the worked examples has multiplicity 2.
This is a notebook convenience, not a general DOT parser. A second declaration of the same pair
with a different style or direction raises instead of silently overwriting; the examples do not
need parallel edges. The parser builds `Bigraph(QuasiCartanMatrix(rows), labels)`; the matrix remains
the source of truth.

Alternatives considered:

- Public `Bigraph.from_paths`: rejected for this change because the notation is presentation, and a
  public grammar would need a broader contract for labels, arrows, multiplicities, and conflicts.
- Full Graphviz DOT: rejected as a new syntax and, if a real parser were used, a new dependency.
- Writing every matrix by hand: rejected because A9 and D13 become noisy and the structural
  similarity to the thesis disappears.

If the notation later proves broadly useful, it can be promoted to a library feature in a separate
change; the notebook call sites would not need to change much.

### 3. Matrix display

Every worked example displays a labeled matrix before its figure. A local `show_matrix(matrix,
labels)` helper renders an HTML table with a header row of labels, a header column of labels, and
the exact integer entries. The table is wrapped in a horizontally scrollable container; the 13 by
13 D13 matrix is additionally placed in a collapsed `<details>` block so the narrative stays
readable.

The custom helper is necessary because the D8 result has labels `(1, 2, 3, 4, 5, 7, 8, 9)`, not
`1..8`. Using index-based rendering would silently mislabel the matrix. For contiguous examples the
helper produces the same labels as the canonical 1-based display, so the notebook uses it uniformly.

The display options still set `matrix_index_base = 1` for any library-rendered objects. The notebook
does not rely on `Bigraph.to_markdown()`, which lists edges rather than matrix entries.

Alternative considered: a Markdown table for every matrix. Rejected for D13 because the wide table
dominates the page; HTML with scrolling and collapsing keeps the matrix primary without breaking the
narrative.

### 4. Drawability predicate

The notebook defines a local `has_faithful_bigraph(matrix)` predicate:

- For each pair `i < j`, let `a = A[i][j]` and `b = A[j][i]`.
- Both zero means no edge.
- `a == b` means an undirected edge.
- `min(abs(a), abs(b)) == 1` means an arrow of multiplicity `max(abs(a), abs(b))`.
- Any other nonzero pair has no faithful bigraph.

This is the thesis's representability condition. The notebook states the supporting fact: a
positive definite quasi-Cartan matrix has `4 - a*b > 0` for every pair, so every positive definite
example is drawable. The `2/4` matrix `((2, 2), (4, 2))` has inertia `(1, 1, 0)` and
`NOT_ADMISSIBLE`; it is shown as a matrix-only case.

The predicate is local because `Bigraph.edges` currently turns that matrix into a multiplicity-4
arrow, which is the bigraph of the `1/4` matrix, not the `2/4` matrix. The notebook never calls
`.edges` for a non-drawable case. A separate change will make `Bigraph.edges` raise for the
non-representable range; this notebook change must not depend on that fix.

Alternative considered: silently showing the lossy arrow. Rejected because it contradicts the
matrix-first claim and would misrepresent the matrix.

### 5. Drawing and layout

The drawing helper uses matplotlib directly and NetworkX only for the block-cut tree. It iterates
`Bigraph.edges` rather than `to_networkx()` so arrow direction and style are preserved.

- Undirected edges are `Axes.plot` lines; directed edges are `FancyArrowPatch` with shrink values
  that keep arrowheads outside the node markers.
- Solid edges use `"-"`; dotted edges use `":"`.
- Multiplicity labels are shown when the thesis convention requires them: multiplicity greater than
  one, except directed edges of multiplicity two. The worked examples are simple, so most figures
  have no multiplicity labels.
- Highlighted core edges and the cyclization path use a distinct color and line width; highlighted
  vertices use a different marker fill.
- Node labels are the `Bigraph` labels.
- Every figure has a fixed `positions` dictionary. D13 positions are adapted from the thesis's TikZ
  figure; A9, D8, and A3 positions are chosen once and stored in the notebook. No spring layout is
  used for the paper figures.
- The block-cut tree uses `BlockCutTree.to_networkx()` and a deterministic bipartite layout: vertex
  nodes on one horizontal line, block nodes at the centroid of their incident labels, block nodes as
  squares, vertex nodes as circles, and block labels as their label sets.
- The helper returns the axes; callers use `plt.subplots` for side-by-side comparisons and
  `plt.show()` at the end.

Alternative considered: `nx.draw_networkx`. Rejected because it does not preserve arrow direction
from a collapsed `to_networkx()` view and gives less control over dotted edges and highlights.

### 6. Exact example data

The notebook uses the validated examples from the structural tests and golden matrices.

A9, labels `1..9`:

```text
1--2..3--1
3..4--5--3
3--9--4
5..9
5..6..8..5
4--7
```

D13 core, labels `1..5`:

```text
1..2--3..4..5--1
```

D13 pieces, one per core edge:

```text
F1: 1..2..8..1
F2: 2--3--6..2
    6--7
F3: 3..4
F4: 4--10--5..4
    5..9..4
    9--10
    9--12--13..9
    10--11
F5: 1--5
```

The gluing is built with `DCore(EdgeMultiset.from_bigraph(core)).gluing(pieces, labels=range(1, 14))`.
The notebook renders the resulting D13 graph with the five core edges highlighted and lists the
pieces, their labels, their A-types, and their core edges.

Mixed B6, labels `1..6`, used in the boundary table:

```text
1--2--3..1
3.>6
3--4..5--3
4->6
5->6
```

This expands to the golden mixed matrix in `tests/test_classifier_golden.py`. It is shown as a
matrix table, not as a figure, to keep the notebook brief.

Flation uses `A3 = graph("1--2--3", labels=(1, 2, 3))` and `A3.flation(1, 2)`. The notebook verifies
that the inertia and classified type remain A3 and shows the two matrices and two graphs side by
side.

A-cyclization uses `A9.a_cyclization(1, 6)`. The notebook asserts the path `(1, 3, 5, 6)`, the D8
result, and the certificate. The D8 figure labels the identified vertex `1=6`.

E6 uses `DynkinType("E", 6).cartan_matrix()` with labels `1..6`. Affine A2 uses the symmetric
matrix with all off-diagonal entries `-1`, labels `1..3`, inertia `(2, 0, 1)`, and
`NOT_POSITIVE_DEFINITE`. The matrix-only case uses `((2, 2), (4, 2))`.

### 7. Boundary table

The table has one row per case and columns for inertia, classifier outcome, bigraph availability,
and certificate kind. It is generated from a list of `(name, matrix, labels, certificate)` tuples
and rendered as Markdown.

The table uses `matrix.classify()` rather than `Bigraph.classify()` so the `2/4` row does not enter
the lossy bigraph path. The type column shows the finite type when classified, otherwise the status
name. The certificate column reads `A-type`, `D-core`, `A-cyclization`, or `None`.

The algorithm section presents the thesis's high-level branch structure, then states explicitly that
`classify()` is an exact bounded-flation oracle, not a reimplementation of the thesis's polynomial
algorithm. This distinction is part of the table caption as well as the prose.

### 8. Latest-version Colab setup

The setup cell detects Colab with `"google.colab" in sys.modules` and runs:

```python
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "-U",
     "dynkinlab @ git+https://github.com/knkillname/dynkinlab.git@main"]
)
```

Outside Colab the cell is a no-op and the notebook imports the checkout or installed package. The
final cell prints `dynkinlab.__version__` without a pin. PyPI is never used.

The command deliberately tracks the `main` branch and uses `-U`, so the notebook installs the
latest DynkinLab and lets pip resolve current dependency versions. It does not select a commit SHA,
tag, or package version. The cost is that a future `main` change can break the notebook; the smoke
test and CI execute it on every commit, and the committed outputs show the last verified run. If
the notebook needs a stable target later, a separate change can introduce a release tag, but this
change follows the requested latest-version behavior.

Alternative considered: install a pinned commit or tag. Rejected by the requested no-pin policy.
Alternative considered: install the latest PyPI release. Rejected because the A/D structural
surface may not be published there, and the project guide already treats the git checkout as the
source of truth.

### 9. Smoke execution

`tests/test_notebook.py` reads `notebooks/thesis-companion.py` with Jupytext's percent-format reader,
compares its cell types and normalized sources with the committed
`notebooks/thesis-companion.ipynb`, and executes the source-derived notebook with
`nbclient.NotebookClient` using the `python3` kernel, a 180-second timeout, and the notebook
directory as the working directory. `ipykernel` already provides the kernel spec in the Hatch test
environment; the test-only extras add `jupytext`, `nbclient`, and `nbformat`.

The notebook's own final assertions are the behavioral checks. The test fails if any cell raises,
if execution times out, or if the generated notebook has drifted from the source. It does not
compare committed outputs, so the notebook can be re-rendered without brittle output diffs.

The Colab install branch is skipped because `google.colab` is not imported in the test kernel. The
notebook uses ordinary `plt.show()` calls and no IPython magics; the kernel's default backend
handles figure display without blocking.

Alternative considered: a separate `hatch run notebook:smoke` script. Rejected because `hatch test`
already runs in CI and provides the kernel environment; adding a second entry point would make the
authoritative gate easy to miss.

### 10. Documentation and public surface

`notebooks/README.md` becomes the notebook guide: editing the `.py` source, regenerating the
`.ipynb` with Jupytext, the latest git install, the Colab workflow, the local smoke test, and the
fact that the notebook is a companion rather than an API tutorial.
`docs/getting-started/quickstart.md` points to the notebook instead of a future quickstart.
`docs/reference/index.md` removes the notebook from the planned surface. `CHANGELOG.md` records the
notebook under `[Unreleased]`.

No `src/dynkinlab` file changes, no `__all__` changes, no wheel-content changes, and no runtime
dependency changes. The test-only dependencies (`jupytext`, `nbclient`, and `nbformat`) are added
to the `hatch-test` extra.

## Risks / Trade-offs

- **The generated `.ipynb` drifts from the `.py` source** -> The smoke test compares cell types and
  normalized sources; regenerate the notebook before committing.
- **Jupytext adds a test-only dependency** -> It is not imported by the library, does not enter the
  wheel, and can be replaced by a small percent-format parser if the dependency becomes a problem.
- **Markdown cell prefixes can be malformed** -> Use `# %% [markdown]` and prefix every Markdown
  line with `# `; verify a Jupytext read/round-trip in the smoke test.
- **Notebook-local helpers duplicate presentation code** -> Keep them small and exercised by the
  smoke test. If the notation becomes generally useful, promote it in a separate change.
- **The parser is not a full DOT implementation** -> Document the six operators, the fixed arrow
  multiplicity, and the conflict rule in the notebook; the worked examples stay inside that range.
- **`Bigraph.edges` is lossy for the `2/4` case** -> The notebook uses a local drawability predicate
  and never calls `.edges` for that matrix. The library fix is explicitly a separate change.
- **D8 has non-contiguous labels** -> Use the custom labeled matrix helper and verify the header
  against `cyc.result.labels`.
- **Tracking `main` can pick up a breaking change** -> The smoke test and CI execute the notebook on
  every commit; the committed outputs record the last verified run, and a future change can add a
  release tag if stability becomes more important than latest-version behavior.
- **The latest PyPI release may lag behind `main`** -> The notebook installs from the git default
  branch, where the A/D surface is present, rather than relying on a package release.
- **Committed outputs can go stale** -> The smoke test executes the notebook from a clean kernel;
  the outputs are regenerated as part of the implementation and verified by the test.
- **Matplotlib or kernel differences in CI** -> Use the `python3` kernel that `ipykernel` registers,
  the inline backend in the notebook, and the existing matplotlib version floor. The smoke test
  runs in the same Hatch test environment as the rest of the suite.
- **Third-party test imports may be untyped** -> If Pyrefly reports missing stubs for `jupytext`,
  `nbclient`, or `nbformat`, add a minimal import ignore in the test only and keep the test
  functions annotated.
- **The table's classifier caveat could be missed** -> Put the caveat in the algorithm prose and in
  the table caption, not only in a code comment.

## Migration Plan

Not applicable. This is an additive repository artifact; there is no runtime migration or rollback
path. The notebook can be removed without changing the package.

## Open Questions

None. The script-first source format, latest-version install, matrix display, drawability handling,
and smoke-test entry point are resolved above. The notebook intentionally tracks `main`; no version
selection is deferred.
