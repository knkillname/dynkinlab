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

## Quasi-Cartan matrices

Build a symmetric quasi-Cartan matrix, evaluate its integer quadratic form, and inspect its exact
inertia:

```python
from dynkinlab import QuasiCartanMatrix

matrix = QuasiCartanMatrix(((2, -1), (-1, 2)))
form = matrix.quadratic_form
inertia = matrix.inertia

assert form.evaluate((1, 1)) == 1
assert inertia.is_positive_definite
print(f"dimension={matrix.dimension}, inertia={inertia}")
```

The `A_2` Cartan matrix above has inertia `(2, 0, 0)`. The affine `A_2^(1)` matrix has inertia
`(2, 0, 1)`, and indefinite forms have both positive and negative counts. Inertia is computed by
exact rational congruence reduction.

The current mathematical surface covers quasi-Cartan matrices, quadratic forms, and inertia, plus
Colab-ready Markdown and HTML representations. Later changes add the bigraph, flation, and Dynkin
classifier layers.

## Rich display in Colab

`QuasiCartanMatrix` displays as an HTML table in Colab, while `QuadraticForm` and `Inertia` display
as Markdown. The same strings are available without a notebook:

```python
from dynkinlab import QuasiCartanMatrix
from dynkinlab.display import DisplayOptions

matrix = QuasiCartanMatrix(((2, -1), (-1, 2)))
options = DisplayOptions(matrix_index_base=1)

print(matrix.to_markdown(options=options))
print(matrix.to_html(options=options))
print(matrix.quadratic_form.to_markdown(options=options))
print(matrix.inertia.to_markdown())
```

Set notebook-wide defaults through the canonical options object:

```python
import dynkinlab

dynkinlab.display.options.matrix_index_base = 1
dynkinlab.display.options.expanded_form_dimension_limit = 4

dynkinlab.display.options.reset()
```

A per-call `DisplayOptions` object overrides the canonical settings for that call only.

## Notebooks in Colab

The [notebooks guide](https://github.com/knkillname/dynkinlab/blob/main/notebooks/README.md)
describes the Colab workflow and the planned quickstart notebook. Notebook files live in the
repository's `notebooks/` directory.
