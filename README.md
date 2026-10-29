# DynkinLab

[![PyPI - Version](https://img.shields.io/pypi/v/dynkinlab.svg)](https://pypi.org/project/dynkinlab)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/dynkinlab.svg)](https://pypi.org/project/dynkinlab)

DynkinLab is a typed Python library for Dynkin diagrams, quivers, and bigraphs. It is designed to
run in Google Colab as well as in a local Python 3.13+ environment.

The current surface covers symmetrizable quasi-Cartan matrices, exact integer quadratic forms and
inertia, label-generic mixed bigraphs, first-class flation witnesses, and the paper-faithful cubic
Dynkin classifier. Value objects render as Markdown or HTML in Colab and Jupyter.

## Requirements

- Python 3.13 or newer
- NumPy 2.1 or newer
- NetworkX 3.7 or newer
- matplotlib 3.10 or newer

## Installation

```console
pip install dynkinlab
```

In a Google Colab notebook, install the package in a cell:

```console
!pip install dynkinlab
```

The current Colab CPU runtime ships Python 3.13, NumPy 2.1, NetworkX 3.7, and matplotlib 3.10, so
the package installs without extra setup.

## Quick start

```python
from dynkinlab import QuasiCartanMatrix
from dynkinlab.display import DisplayOptions

matrix = QuasiCartanMatrix(((2, -1), (-1, 2)))
form = matrix.quadratic_form
inertia = matrix.inertia
options = DisplayOptions(matrix_index_base=0, expanded_form_dimension_limit=3)

assert matrix.dimension == 2
assert form.evaluate((1, 1)) == 1
assert inertia.is_positive_definite
assert "|   | 0 | 1 |" in matrix.to_markdown(options=options)
assert "<table" in matrix.to_html(options=options)
assert "positive definite" in inertia.to_markdown()
print(f"A_2: dimension={matrix.dimension}, q(1,1)={form.evaluate((1, 1))}, inertia={inertia}")
print(matrix.to_markdown(options=options))
print(form.to_markdown(options=options))
```

Every Python snippet in this README is executed by the test suite, so the example stays runnable.

## Symmetrizability certificates

A valid quasi-Cartan matrix with no positive diagonal symmetrizer raises `NotSymmetrizableError`.
The exception carries a verifiable `SymmetrizerCertificate`:

```python
from dynkinlab import NotSymmetrizableError, QuasiCartanMatrix

entries = ((2, -1, -2), (-2, 2, -1), (-1, -2, 2))
try:
    QuasiCartanMatrix(entries)
except NotSymmetrizableError as error:
    assert error.certificate.verify(entries)
    print(error.certificate.kind)
```

The certificate reports a zero entry on one side of an edge, opposite signs on an edge, or an
inconsistent cycle of edge ratios.

## Bigraphs and Dynkin classification

`Bigraph` is a label-generic view over a `QuasiCartanMatrix`. Flation is delegated to the matrix,
and `classify()` returns a verifiable `DynkinClassification` with a first-class flation witness:

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

`Bigraph.to_markdown()` summarizes edges and arrows; `Bigraph.to_html()` and `_repr_html_` provide
an inline SVG for Colab. `FlationStep` and `FlationSequence` expose the transformation sequence.

## Colab display

Matrices render as HTML tables in Colab, while quadratic forms and inertia render as Markdown.
Mixed bigraphs render as inline SVG, and Dynkin results render as Markdown.
`dynkinlab.display.options` holds the notebook-wide defaults:

```text
dynkinlab.display.options.matrix_index_base
dynkinlab.display.options.expanded_form_dimension_limit
dynkinlab.display.options.bigraph_layout_seed
dynkinlab.display.options.bigraph_figure_size
dynkinlab.display.options.bigraph_show_labels
dynkinlab.display.options.bigraph_show_multiplicity
dynkinlab.display.options.bigraph_arrow_style
dynkinlab.display.options.reset()
```

Pass a `DisplayOptions` object to a rendering method for a one-call override. The override wins over
the canonical settings and leaves them unchanged.

## Development

The project uses Hatch for the quality gates:

```console
hatch check --fix
hatch test
```

The pre-push hook and the full host verification run:

```console
scripts/verify --skip-colab
```

The Colab compatibility gate is separate because the pinned runtime image is large:

```console
scripts/verify
```

### Dev container

The repository includes a VS Code Dev Container based on the latest Google Colab CPU runtime, with
Hatch, OpenCode, pre-created test and docs environments, and Docker-in-Docker for the Colab gate.
Install the **Dev Containers** extension, open the repository, and run
**Dev Containers: Reopen in Container**.

The container has tasks for the usual workflows: **Tasks: Run Task** → `verify: full`,
`verify: skip Colab`, `hatch: test`, or `docs: serve`. Run `opencode` in the integrated terminal to
work inside the container with the agent. The selected Hatch environment includes `ipykernel`, so
VS Code notebooks run without installing a separate kernel. The first full verification downloads
the separate pinned Colab image used by the compatibility gate into the container's Docker daemon.

## Documentation

The Sphinx documentation lives under `docs/`. Build it with:

```console
hatch run docs:build
```

## License

DynkinLab is distributed under the terms of the MIT license. See [LICENSE](LICENSE).
