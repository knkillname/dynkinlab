# DynkinLab

[![PyPI - Version](https://img.shields.io/pypi/v/dynkinlab.svg)](https://pypi.org/project/dynkinlab)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/dynkinlab.svg)](https://pypi.org/project/dynkinlab)

DynkinLab is a typed Python library for Dynkin diagrams, quivers, and bigraphs. It is designed to
run in Google Colab as well as in a local Python 3.13+ environment.

This repository currently provides the package foundation plus the first mathematical layer:
symmetric quasi-Cartan matrices, exact integer quadratic forms, and exact inertia. Bigraphs,
flation, and Dynkin classification arrive in later changes.

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

matrix = QuasiCartanMatrix(((2, -1), (-1, 2)))
form = matrix.quadratic_form
inertia = matrix.inertia

assert matrix.dimension == 2
assert form.evaluate((1, 1)) == 1
assert inertia.is_positive_definite
print(f"A_2: dimension={matrix.dimension}, q(1,1)={form.evaluate((1, 1))}, inertia={inertia}")
```

Every Python snippet in this README is executed by the test suite, so the example stays runnable.

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
