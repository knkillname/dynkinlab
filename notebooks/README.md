# Notebooks

This directory holds the Colab notebooks for DynkinLab. The first quickstart notebook will be
added in the follow-up change that introduces the mathematical API.

## Colab workflow

1. Open the notebook in Google Colab and select a CPU runtime.
2. Install the package in the first cell:

   ```console
   !pip install dynkinlab
   ```

3. Import the package and keep any generated files under `/content`. The Colab filesystem is
   ephemeral; mount Google Drive when a file must outlive the runtime.

The compatibility gate in `scripts/colab-smoke` runs against the same pinned CPU image that Colab
uses, so the dependency floors and interpreter version are checked before a release.

## Future quickstart notebook

`notebooks/quickstart.ipynb` will demonstrate the first mathematical workflow: building a
quasi-Cartan matrix, computing its quadratic form and inertia, and rendering a bigraph inline.
Until that notebook exists, use the [quickstart page](../docs/getting-started/quickstart.md) for the
installation check.
