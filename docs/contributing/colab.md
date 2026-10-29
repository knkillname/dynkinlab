# Colab compatibility

:::{admonition} tl;dr
:class: tldr

Develop inside the dev container. It is built from the latest official Colab CPU runtime, and
`hatch run verify` checks compatibility against that container's Python without downloading another
image.
:::

## Runtime image

The source of truth is `.devcontainer/Dockerfile`, which builds from the official Colab CPU runtime
on the `latest` tag. Rebuilding the dev container picks up the current Colab runtime. No other step
pulls an image.

## What is checked

`hatch run colab:smoke` uses the container's system Python through Hatch's `system` environment and
asserts:

1. `sys.version_info[:2] == (3, 13)`.
2. NumPy is at least 2.1, NetworkX is at least 3.7, and matplotlib is at least 3.10.
3. `import dynkinlab` succeeds and `dynkinlab.__version__` is a non-empty string.

The command does not create a virtual environment or install dependencies; it runs against the
Colab runtime already present in the container.

## Run the checks

Inside the dev container:

```console
hatch run verify
```

This runs `hatch check`, `hatch test`, and the Colab compatibility assertions. The pre-push hook uses
the same command.

Run only the compatibility assertions with:

```console
hatch run colab:smoke
```

## CI

CI runs `hatch check`, `hatch test`, wheel packaging, and the Sphinx build on standard runners. It
does not download the Colab image or run a separate Colab job. Colab compatibility is verified by
running the test and verification commands inside the dev container.
