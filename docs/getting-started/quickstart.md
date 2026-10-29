# Quickstart

:::{admonition} tl;dr
:class: tldr

Install DynkinLab in a Python 3.13 environment or a Colab notebook, then import the package and
check its version.
:::

## Install locally

DynkinLab requires Python 3.13 or newer. NetworkX 3.7 or newer and matplotlib 3.10 or newer are
installed as runtime dependencies.

```{code-block} console
pip install dynkinlab
```

## Install in Google Colab

The current Colab CPU runtime already provides the supported Python and dependency versions.
Install the package in a notebook cell:

```{code-block} console
!pip install dynkinlab
```

## Verify the installation

```python
import dynkinlab

assert dynkinlab.__version__
print(f"DynkinLab {dynkinlab.__version__}")
```

The package currently exposes only `__version__`. The next change adds the mathematical API for
Dynkin diagrams and bigraphs.

## Notebooks in Colab

The [notebooks guide](https://github.com/knkillname/dynkinlab/blob/main/notebooks/README.md)
describes the Colab workflow and the planned quickstart notebook. Notebook files live in the
repository's `notebooks/` directory.
