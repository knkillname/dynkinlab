# Notebooks

This directory holds the thesis companion notebook, a short computational paper about the A/D
classification of quasi-Cartan matrices. It is deliberately not an API tutorial: the matrix and its
structural certificates are the subject, and the drawings are supporting views.

## Files

- `thesis-companion.py` is the editable source. It uses the Jupytext percent format: `# %%` starts a
  code cell and `# %% [markdown]` starts a Markdown cell whose lines are prefixed with `# `.
- `thesis-companion.ipynb` is generated from the source and is the artifact to open in Colab. It is
  committed with its computed tables and figures.

Edit the Python source, then regenerate the notebook:

```console
jupytext --to ipynb notebooks/thesis-companion.py --output notebooks/thesis-companion.ipynb
```

The source contains no IPython magics, so it also remains valid Python. The notebook-local display
and drawing helpers stay in the source; they are not part of the package API.

## Colab workflow

1. Open `thesis-companion.ipynb` from the repository in Google Colab and select a CPU runtime.
2. In Colab, the setup cell installs the latest DynkinLab from the project's git default branch:

   ```text
   dynkinlab @ git+https://github.com/knkillname/dynkinlab.git@main
   ```

   The command uses `-U` and selects no commit, tag, or package version. A local checkout skips the
   install and uses the package available in the current environment.
3. Run all cells. The final cell reports the installed package version and confirms that every
   certificate verified.

The notebook uses only DynkinLab, NumPy, NetworkX, matplotlib, and the Jupyter/IPython runtime
provided by Colab.

## Local verification

The smoke test parses the Python source, checks that the committed notebook has the same cell
types and normalized sources, and executes the source-derived notebook in a clean kernel:

```console
hatch test tests/test_notebook.py --no-cov
```

The full `hatch test` run includes this check. Regenerate the notebook whenever the source changes;
the comparison fails on drift.

## Rich display

Library value objects still render as HTML or Markdown in Colab. Notebook-wide defaults live in
`dynkinlab.display.options`:

```python
import dynkinlab

dynkinlab.display.options.matrix_index_base = 1
dynkinlab.display.options.expanded_form_dimension_limit = 4
```

Rendering methods also accept a per-call `DisplayOptions` object. See the
[API reference](../docs/reference/index.md) for the full display API.
