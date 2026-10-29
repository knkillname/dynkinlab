# API reference

:::{admonition} tl;dr
:class: tldr

The public API is intentionally empty apart from the package version. The mathematical reference
will be generated here as the library grows.
:::

## Current surface

The package exposes one public name:

- `dynkinlab.__version__` — the installed package version as a non-empty string.

## Planned surface

The next change introduces `QuasiCartanMatrix`, `QuadraticForm`, `Inertia`, and exact inertia.
Later changes add `Bigraph`, flation, and the Dynkin classifier. None of those names are part of the
bootstrap package.
