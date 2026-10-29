# Colab compatibility

:::{admonition} tl;dr
:class: tldr

The Colab gate runs the official CPU runtime image pinned by digest. Use `scripts/colab-smoke` for
the gate alone or `scripts/verify --skip-colab` for the fast host checks.
:::

## Pinned image

The single source of truth is `scripts/colab-image`. It contains:

```text
us-docker.pkg.dev/colab-images/public/cpu-runtime@sha256:17929077afbae4b805c774450b50ed6945daa122107ac0fb492ce057bc0c3539
```

The image is the official Colab CPU runtime from release
`release-colab-external-images_20261027-060057_RC00`. It runs Python 3.13.18 and ships NetworkX 3.7
and matplotlib 3.10.0. The smoke script mounts the repository read-only and imports `dynkinlab`
from `src` with `PYTHONPATH`, so nothing is installed into the image's system Python.

Update the digest deliberately when Colab changes. Do not replace it with `latest`: the smoke result
must stay reproducible.

## Run the smoke gate

```console
scripts/colab-smoke
```

The script asserts:

1. `sys.version_info[:2] == (3, 13)`.
2. NetworkX is at least 3.7 and matplotlib is at least 3.10.
3. `import dynkinlab` succeeds and `dynkinlab.__version__` is a non-empty string.

It prints the interpreter and dependency versions before exiting.

## Skip Colab for local checks

```console
scripts/verify --skip-colab
```

This runs `hatch check` and `hatch test`, then reports the Colab gate as skipped. The pre-push hook
uses this fast path so a normal push does not pull or start the 28.3 GB runtime image.

Run the full sequence when changing the package boundary, the dependency floors, or the Colab gate:

```console
scripts/verify
```

## CI split

CI has a dedicated `colab` job that reads `scripts/colab-image`, verifies the reference ends in a
`sha256:` digest and does not use `latest`, then runs `scripts/colab-smoke`. The other jobs stay
small and fast:

- `check` runs `hatch check`.
- `test` runs `hatch test` on Python 3.13.
- `package` builds the wheel, installs it in a clean environment, and verifies `py.typed` and the
  license file.
- `docs` builds the Sphinx site with warnings treated as errors.

Keeping the Colab gate separate avoids downloading the image for every unit-test run.
