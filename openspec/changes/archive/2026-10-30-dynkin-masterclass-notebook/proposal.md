# Proposal

## Why

The current thesis companion is a paper with a shadow library inside it: 713 code lines, five code
cells over 40 lines, and hardcoded A3/A9/D8/D13 coordinates. It cannot serve as a ground-up
masterclass, and readers cannot change an input without confronting notebook-local infrastructure.
The library already owns the mathematics; it should also own the reusable notation, layout,
rendering, and certificate summaries that the notebook keeps reimplementing.

## What Changes

- Replace `notebooks/thesis-companion.py` and `notebooks/thesis-companion.ipynb` with
  `notebooks/dynkin-masterclass.py` and its generated `.ipynb`, and retire the `thesis-notebook`
  capability.
- Rewrite the notebook as an English, ground-up masterclass: matrices and quadratic forms, exact
  inertia, faithful bigraphs and the drawable boundary, flation, finite types, the A-type block-cut
  certificate, the D-core gluing certificate, the A-cyclization certificate, and a boundary/lab
  section. It SHALL NOT present itself as a thesis companion or compare against a thesis-specific
  algorithm.
- Add notebook-facing public methods without adding new `__all__` names:
  - `Bigraph.from_paths` for the compact six-operator path notation.
  - `QuasiCartanMatrix.has_faithful_bigraph`; `Bigraph.edges` rejects non-faithful pairs instead of
    deriving a lossy arrow. **BREAKING** for matrices outside the faithful range, including `2/4`.
  - Optional labels on `QuasiCartanMatrix.to_markdown` and `to_html`, plus optional HTML
    collapsing for wide matrices.
  - `Bigraph.layout`, `Bigraph.draw`, and `BlockCutTree.draw`, driven by NetworkX layouts, plus a
    `DisplayOptions.bigraph_layout` setting.
  - `to_markdown` and notebook hooks for the A-type, D-core gluing, and A-cyclization certificates,
    and for flation sequences.
- Make literary coding a contract: every code cell is at most 40 physical lines, no code cell
  defines a function or class, no coordinate tables are stored, and every code cell is preceded by
  Markdown that explains its purpose.
- Keep the script-first Jupytext source, the committed Colab artifact, the latest-version git
  install, clean-kernel smoke execution, and source/artifact synchronization.
- Update the notebook guide, quickstart, API reference, changelog, and notebook tests.

## Capabilities

### New Capabilities

- `dynkin-masterclass`: the script-first, executable, ground-up masterclass notebook, its
  pedagogical structure, fixed tour and parameter labs, matrix-first A/D examples, NetworkX-driven
  figures, boundary table, Colab setup, and smoke execution.

### Modified Capabilities

- `bigraph-flation`: adds the compact path notation, the faithful-bigraph predicate on matrices, and
  truthful non-faithful edge rejection; extends the flation witness with Markdown rendering.
- `colab-display`: adds labeled matrix tables, collapsed wide tables, the bigraph layout setting,
  NetworkX-driven layout/drawing for bigraphs and block-cut trees, and Markdown rendering for
  structural certificates.
- `thesis-notebook`: retired in full and replaced by `dynkin-masterclass`; its local-helper,
  paper-companion, and thesis-algorithm requirements no longer describe the repository.

## Impact

- New: `notebooks/dynkin-masterclass.py`, `notebooks/dynkin-masterclass.ipynb`, and a new
  `openspec/specs/dynkin-masterclass/spec.md` after archive.
- Removed: `notebooks/thesis-companion.py`, `notebooks/thesis-companion.ipynb`, and the
  `thesis-notebook` capability.
- Modified source: `src/dynkinlab/bigraph.py`, `bigraph_display.py`, `quasi_cartan.py`,
  `structure.py`, `structure_d.py`, `flation.py`, `display.py`, and their tests.
- Modified repository artifacts: `tests/test_notebook.py`, `notebooks/README.md`,
  `docs/getting-started/quickstart.md`, `docs/reference/index.md`, and `CHANGELOG.md`.
- No new runtime dependency, no new `__all__` names, no wheel-content change beyond the source
  modules already in the package. The notebook still installs the latest `main` from git in Colab.
- The deliberate `2/4` edge behavior change is the only breaking API behavior; the classifier reads
  matrix entries directly and is unaffected.
