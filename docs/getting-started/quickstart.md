# Quickstart

:::{admonition} tl;dr
:class: tldr

Install DynkinLab in a Python 3.13 environment or a Colab notebook, then import the package and
check its version.
:::

## Install locally

DynkinLab requires Python 3.13 or newer. NumPy 2.1 or newer, NetworkX 3.7 or newer, and
matplotlib 3.10 or newer are installed as runtime dependencies.

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

Build a quasi-Cartan matrix, evaluate its integer quadratic form, and inspect its exact inertia.
Symmetric matrices are the special case where the symmetrizer is all ones:

```python
from dynkinlab import QuasiCartanMatrix

matrix = QuasiCartanMatrix(((2, -1), (-1, 2)))
form = matrix.quadratic_form
inertia = matrix.inertia

assert form.evaluate((1, 1)) == 1
assert inertia.is_positive_definite
assert matrix.symmetrizer == (1, 1)

non_symmetric = QuasiCartanMatrix(((2, -1), (-2, 2)))
assert non_symmetric.symmetrizer == (2, 1)
assert non_symmetric.symmetrized_entries == ((4, -2), (-2, 2))
print(f"dimension={matrix.dimension}, inertia={inertia}")
```

The $A_2$ Cartan matrix above has inertia $(2, 0, 0)$. The affine $A_2^{(1)}$ matrix has inertia
$(2, 0, 1)$, and indefinite forms have both positive and negative counts. Inertia is computed by
exact rational congruence reduction on $S = DA$.

## Bigraphs and Dynkin classification

`Bigraph` is a label-generic view over `QuasiCartanMatrix`. It exposes flation and a verifiable
classifier result:

```python
from dynkinlab import Bigraph, DynkinType

matrix = DynkinType("B", 5).cartan_matrix()
bigraph = Bigraph(matrix, ("a", "b", "c", "d", "e"))
flated = bigraph.flation("a", "b")
result = flated.classify()

assert result.components[0].dynkin_type == DynkinType("B", 5)
assert result.verify()
print(result.to_markdown())
```

The classifier supports the full finite list A–G and returns a first-class flation witness, the
normal form, exact inertia, and clear `NOT_ADMISSIBLE` / `NOT_POSITIVE_DEFINITE` outcomes. $G_2$
is handled as a documented extension to the admissibility conditions.

## Rich display in Colab

`QuasiCartanMatrix` displays as an HTML table in Colab, while `QuadraticForm` and `Inertia` display
as Markdown. `Bigraph` displays as inline SVG, and `DynkinType` / `DynkinClassification` display as
Markdown. The same strings are available without a notebook:

```python
from dynkinlab import Bigraph, DynkinType, QuasiCartanMatrix
from dynkinlab.display import DisplayOptions

matrix = QuasiCartanMatrix(((2, -1), (-2, 2)))
options = DisplayOptions(matrix_index_base=1)
bigraph = Bigraph(matrix, ("a", "b"))
result = DynkinType("A", 2).cartan_matrix().classify()

print(matrix.to_markdown(options=options))
print(matrix.to_html(options=options))
print(matrix.quadratic_form.to_markdown(options=options))
print(matrix.inertia.to_markdown())
print(bigraph.to_html())
print(result.to_markdown())
```

Set notebook-wide defaults through the canonical options object:

```python
import dynkinlab

dynkinlab.display.options.matrix_index_base = 1
dynkinlab.display.options.expanded_form_dimension_limit = 4
dynkinlab.display.options.bigraph_layout_iterations = 10000
dynkinlab.display.options.bigraph_layout_seed = 7
dynkinlab.display.options.bigraph_show_multiplicity = False

dynkinlab.display.options.reset()
```

A per-call `DisplayOptions` object overrides the canonical settings for that call only.

## Masterclass notebook

The [DynkinLab masterclass](https://github.com/knkillname/dynkinlab/blob/main/notebooks/dynkin-masterclass.ipynb)
is an executable, ground-up lesson covering quasi-Cartan matrices, exact inertia, faithful
bigraphs, flation, the finite types, the A9 certificate, the D13 D-core gluing, and the A9-to-D8
cyclization. Its editable Python source, parameter labs, and Colab workflow are described in the
[notebooks guide](https://github.com/knkillname/dynkinlab/blob/main/notebooks/README.md).
