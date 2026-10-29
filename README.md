# DynkinLab

[![PyPI - Version](https://img.shields.io/pypi/v/dynkinlab.svg)](https://pypi.org/project/dynkinlab)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/dynkinlab.svg)](https://pypi.org/project/dynkinlab)

DynkinLab is a typed Python library for Dynkin diagrams, quivers, and bigraphs. It is designed to
run in Google Colab as well as in a local Python 3.13+ environment.

This repository currently provides the package foundation plus the first mathematical layer:
symmetric quasi-Cartan matrices, exact integer quadratic forms, and exact inertia. The value objects
render as Markdown or HTML in Colab. Bigraphs, flation, and Dynkin classification arrive in later
changes.

## Requirements

- Python 3.13 or newer
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

The current Colab CPU runtime ships Python 3.13, NetworkX 3.7, and matplotlib 3.10, so the package
installs without extra setup.

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

## Colab display

Matrices render as HTML tables in Colab, while quadratic forms and inertia render as Markdown.
`dynkinlab.display.options` holds the notebook-wide defaults:

```text
dynkinlab.display.options.matrix_index_base
dynkinlab.display.options.expanded_form_dimension_limit
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

## Documentation

The Sphinx documentation lives under `docs/`. Build it with:

```console
hatch run docs:build
```

## License

DynkinLab is distributed under the terms of the MIT license. See [LICENSE](LICENSE).
