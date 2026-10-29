# Notebooks

This directory holds the Colab notebooks for DynkinLab. The current value objects render as HTML
tables or Markdown in Colab; a full quickstart notebook will follow.

## Colab workflow

1. Open the notebook in Google Colab and select a CPU runtime.
2. Install the package in the first cell:

   ```console
   !pip install dynkinlab
   ```

3. Import the package and keep any generated files under `/content`. The Colab filesystem is
   ephemeral; mount Google Drive when a file must outlive the runtime.

The compatibility assertions run inside the dev container against the same latest CPU image that
Colab uses, so the dependency floors and interpreter version are checked before a release.

## Rich display

Matrices render as HTML tables, while quadratic forms and inertia render as Markdown. Notebook-wide
defaults live in `dynkinlab.display.options`:

```python
import dynkinlab

dynkinlab.display.options.matrix_index_base = 1
dynkinlab.display.options.expanded_form_dimension_limit = 4
```

Rendering methods also accept a per-call `DisplayOptions` object. See the
[API reference](../docs/reference/index.md) for the full display API.

## Future quickstart notebook

`notebooks/quickstart.ipynb` will demonstrate the first mathematical workflow: building a
quasi-Cartan matrix, computing its quadratic form and inertia, rendering the objects inline, and
visualizing a bigraph. Until that notebook exists, use the
[quickstart page](../docs/getting-started/quickstart.md) for the installation check.
