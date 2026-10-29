# DynkinLab

:::{admonition} tl;dr
:class: tldr

DynkinLab is a typed Python library for Dynkin diagrams, quivers, and bigraphs. It targets the
Google Colab runtime: install it in a notebook cell and import the package, or install it locally
on Python 3.13 or newer.
:::

DynkinLab is a Colab-first library for symmetrizable quasi-Cartan matrices, exact forms and
inertia, label-generic mixed bigraphs, flation witnesses, and the full finite Dynkin
classification. The package installs on Python 3.13 or newer and renders value objects inline in
Colab/Jupyter.

```{code-block} console
pip install dynkinlab
```

```python
import dynkinlab

print(f"DynkinLab {dynkinlab.__version__}")
```

The package requires Python 3.13 or newer, NumPy 2.1 or newer, NetworkX 3.7 or newer, and
matplotlib 3.10 or newer. Those floors match the Google Colab CPU runtime used by the dev container.

```{toctree}
:hidden:
:maxdepth: 2
:caption: Start

getting-started/quickstart
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Reference

reference/index
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Contributing

contributing/colab
```
