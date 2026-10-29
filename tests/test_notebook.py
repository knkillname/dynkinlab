# SPDX-FileCopyrightText: 2026-present Mario Abarca <knkillname@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Smoke tests for the script-first thesis companion notebook."""

from pathlib import Path

import jupytext  # pyrefly: ignore[missing-import]
import nbformat  # pyrefly: ignore[missing-import]
from jupyter_client import KernelManager  # pyrefly: ignore[missing-import]
from nbclient import NotebookClient  # pyrefly: ignore[missing-import]
from nbformat import NotebookNode  # pyrefly: ignore[missing-import]

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_NOTEBOOK_DIR = _PROJECT_ROOT / "notebooks"
_SOURCE_PATH = _NOTEBOOK_DIR / "thesis-companion.py"
_NOTEBOOK_PATH = _NOTEBOOK_DIR / "thesis-companion.ipynb"
_TIMEOUT_SECONDS = 180


def _read_source_with_jupytext() -> NotebookNode:
    """Return the percent-format source parsed into notebook cells."""
    return jupytext.reads(_SOURCE_PATH.read_text(encoding="utf-8"), fmt="py:percent")


def _normalized_cells(notebook: NotebookNode) -> list[tuple[str, str]]:
    """Return cell types and stripped sources for a structural comparison."""
    return [(cell.cell_type, cell.source.strip()) for cell in notebook.cells]


def test_generated_notebook_matches_source() -> None:
    """The committed notebook must be regenerated from the Python source."""
    source_notebook = _read_source_with_jupytext()
    generated_notebook = nbformat.read(_NOTEBOOK_PATH, as_version=4)

    assert _normalized_cells(source_notebook) == _normalized_cells(generated_notebook)


def _plain_kernel_manager() -> KernelManager:
    """Return a kernel manager that bypasses the Colab kernel config.

    The dev container installs an IPython configuration that selects
    ``google.colab`` as the kernel class. The test environment has ipykernel
    but not the Colab package, so the smoke execution explicitly selects the
    ordinary IPython kernel and clears the Colab-only extension list.
    """
    manager = KernelManager(kernel_name="python3")
    if manager.kernel_spec is None:
        raise RuntimeError("the python3 kernel spec is unavailable")
    manager.kernel_spec.argv.extend(
        [
            "--IPKernelApp.kernel_class=ipykernel.ipkernel.IPythonKernel",
            "--InteractiveShellApp.extensions=[]",
        ],
    )
    return manager


def test_notebook_executes_cleanly() -> None:
    """Execute the source-derived notebook in a clean kernel."""
    source_notebook = _read_source_with_jupytext()
    client = NotebookClient(
        source_notebook,
        km=_plain_kernel_manager(),
        timeout=_TIMEOUT_SECONDS,
        resources={"metadata": {"path": str(_NOTEBOOK_DIR)}},
    )

    client.execute()

    final_output = "".join(
        output.get("text", "")
        for output in source_notebook.cells[-1].get("outputs", [])
        if output.output_type == "stream"
    )
    assert "DynkinLab" in final_output
    assert "All certificates verified." in final_output
