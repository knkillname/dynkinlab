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

The current mathematical surface covers quasi-Cartan matrices, quadratic forms, and inertia.
Later changes add the bigraph, flation, and Dynkin classifier layers.

## Notebooks in Colab

The [notebooks guide](https://github.com/knkillname/dynkinlab/blob/main/notebooks/README.md)
describes the Colab workflow and the planned quickstart notebook. Notebook files live in the
repository's `notebooks/` directory.
