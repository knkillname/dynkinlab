# Notebooks

This directory holds the DynkinLab masterclass: a ground-up, executable lesson on quasi-Cartan
matrices, exact inertia, mixed bigraphs, flation, and the structural A/D classification. The
library owns the reusable notation, layout, rendering, and certificate summaries; the notebook
chooses examples, displays results, and checks invariants.

## Files

- `dynkin-masterclass.py` is the editable source. It uses the Jupytext percent format: `# %%`
  starts a code cell and `# %% [markdown]` starts a Markdown cell whose lines are prefixed with
  `# `.
- `dynkin-masterclass.ipynb` is generated from the source and is the artifact to open in Colab. It
  is committed with its computed tables and figures.

Edit the Python source, then regenerate the notebook:

```console
jupytext --to ipynb notebooks/dynkin-masterclass.py --output notebooks/dynkin-masterclass.ipynb
```

The committed `.ipynb` is the Colab artifact with executed tables and figures. After regenerating
it, run all cells in Colab or Jupyter to restore the outputs before committing. The smoke test
executes the source-derived notebook in a clean local kernel and checks the final message, so it
also verifies that the source runs without modifying the committed artifact. The source has no
local function or class definitions, every code cell is at most 40 lines, and every code cell is
preceded by Markdown that explains it.

## Colab workflow

1. Open `dynkin-masterclass.ipynb` from the repository in Google Colab and select a CPU runtime.
2. In Colab, the setup cell installs the latest DynkinLab from the project's git default branch:

   ```text
   dynkinlab @ git+https://github.com/knkillname/dynkinlab.git@main
   ```

   The command uses `-U` and selects no commit, tag, or package version. A local checkout skips the
   install and uses the package available in the current environment.
3. Run all cells. The final cell reports the installed package version and confirms that every
   certificate verified.

The notebook uses only DynkinLab, NumPy, NetworkX, matplotlib, and the Jupyter/IPython runtime
provided by Colab. The automatic layout policy prefers Kamada-Kawai for connected graphs when SciPy
is available; a planar graph containing a four-vertex clique starts from a planar embedding and is
refined by the energy-based spring layout. Other graphs use the same spring layout with
`bigraph_layout_iterations`, which defaults to 10000. Layout distances use unweighted adjacency, so
solid and dotted edges participate equally. The Colab runtime provides SciPy; `planar`, `spring`,
and `circular` need no extra package.

## Fixed tour and parameter labs

The numbered sections are a fixed canonical tour: A2 and A3, the `2/4` matrix-only boundary, A3
flation, E6, A9 and its block-cut tree, D13 gluing, and A9-to-D8 cyclization. The final section
contains a boundary table and parameter labs. Change one of these inputs and rerun its cell:

- `LAYOUT` selects a NetworkX layout policy.
- `PATHS` selects the path-notation graph.
- `SOURCE` and `TARGET` select a flation.
- `X` and `Y` select A-cyclization endpoints.

The labs check invariants or certificate validity rather than one hardcoded type.

## Local verification

The smoke test parses the Python source, checks that the committed notebook has the same cell types
and normalized sources, enforces the 40-line cap, the no-definition rule, and the no-coordinate-table
rule, then executes the source-derived notebook in a clean kernel:

```console
hatch test tests/test_notebook.py --no-cov
```

The full `hatch test` run includes this check. Regenerate the notebook whenever the source changes;
the comparison fails on drift.

## Rich display

Library value objects render as HTML or Markdown in Colab. Notebook-wide defaults live in
`dynkinlab.display.options`:

```python
import dynkinlab

dynkinlab.display.options.matrix_index_base = 1
dynkinlab.display.options.expanded_form_dimension_limit = 4
dynkinlab.display.options.bigraph_layout = "spring"
```

Rendering methods also accept a per-call `DisplayOptions` object. See the
[API reference](../docs/reference/index.md) for the full display API.
