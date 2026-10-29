# DynkinLab

:::{admonition} tl;dr
:class: tldr

DynkinLab is a typed Python library for Dynkin diagrams, quivers, and bigraphs. It targets the
Google Colab runtime: install it in a notebook cell and import the package, or install it locally
on Python 3.13 or newer.
:::

DynkinLab is being rebuilt as a Colab-first library. This bootstrap release establishes the
installable package, the supported dependency floors, and the documentation and verification
infrastructure. The mathematical API arrives in the next change.

```{code-block} console
pip install dynkinlab
```

```python
import dynkinlab

print(f"DynkinLab {dynkinlab.__version__}")
```

The package requires Python 3.13 or newer, NetworkX 3.7 or newer, and matplotlib 3.10 or newer.
Those floors match the pinned Google Colab CPU runtime used by the compatibility gate.

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
