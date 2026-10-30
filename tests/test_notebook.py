# SPDX-FileCopyrightText: 2026-present Mario Abarca <knkillname@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Smoke tests for the script-first DynkinLab masterclass notebook."""

import ast
from pathlib import Path

import jupytext  # pyrefly: ignore[missing-import]
import nbformat  # pyrefly: ignore[missing-import]
from jupyter_client import KernelManager  # pyrefly: ignore[missing-import]
from nbclient import NotebookClient  # pyrefly: ignore[missing-import]
from nbformat import NotebookNode  # pyrefly: ignore[missing-import]

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_NOTEBOOK_DIR = _PROJECT_ROOT / "notebooks"
_SOURCE_PATH = _NOTEBOOK_DIR / "dynkin-masterclass.py"
_NOTEBOOK_PATH = _NOTEBOOK_DIR / "dynkin-masterclass.ipynb"
_MAX_CELL_LINES = 40
_TIMEOUT_SECONDS = 180


def _read_source_with_jupytext() -> NotebookNode:
    """Return the percent-format source parsed into notebook cells."""
    return jupytext.reads(_SOURCE_PATH.read_text(encoding="utf-8"), fmt="py:percent")


def _normalized_cells(notebook: NotebookNode) -> list[tuple[str, str]]:
    """Return cell types and stripped sources for a structural comparison."""
    return [(cell.cell_type, cell.source.strip()) for cell in notebook.cells]


def _code_cells(notebook: NotebookNode) -> list[tuple[int, NotebookNode]]:
    """Return code cells with their notebook indices."""
    return [(index, cell) for index, cell in enumerate(notebook.cells) if cell.cell_type == "code"]


def _direct_markdown_wrappers(source: str) -> list[int]:
    """Return line numbers of direct display(Markdown(value.to_markdown())) calls."""
    tree = ast.parse(source)
    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "display":
            continue
        if not node.args:
            continue
        markdown = node.args[0]
        if not isinstance(markdown, ast.Call):
            continue
        if not isinstance(markdown.func, ast.Name) or markdown.func.id != "Markdown":
            continue
        if not markdown.args:
            continue
        rendered = markdown.args[0]
        if (
            isinstance(rendered, ast.Call)
            and isinstance(rendered.func, ast.Attribute)
            and rendered.func.attr == "to_markdown"
        ):
            lines.append(node.lineno)
    return lines


def _is_number(node: ast.expr) -> bool:
    """Return whether a node is a numeric literal."""
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    )


def _is_point(node: ast.expr) -> bool:
    """Return whether a node is a literal two-number coordinate pair."""
    return (
        isinstance(node, (ast.Tuple, ast.List))
        and len(node.elts) == 2
        and all(_is_number(element) for element in node.elts)
    )


def test_generated_notebook_matches_source() -> None:
    """The committed notebook must be regenerated from the Python source."""
    source_notebook = _read_source_with_jupytext()
    generated_notebook = nbformat.read(_NOTEBOOK_PATH, as_version=4)

    assert _normalized_cells(source_notebook) == _normalized_cells(generated_notebook)


def test_committed_notebook_contains_rendered_outputs() -> None:
    """The Colab artifact must carry its computed tables and figures."""
    notebook = nbformat.read(_NOTEBOOK_PATH, as_version=4)
    outputs = [output for cell in notebook.cells for output in cell.get("outputs", [])]
    images = sum("image/png" in output.get("data", {}) for output in outputs)

    assert images >= 5
    assert any("text/html" in output.get("data", {}) for output in outputs)
    assert any("text/markdown" in output.get("data", {}) for output in outputs)


def test_every_code_cell_is_within_the_line_budget() -> None:
    """Every code cell must contain at most 40 physical lines."""
    for index, cell in _code_cells(_read_source_with_jupytext()):
        line_count = len(cell.source.splitlines())
        assert line_count <= _MAX_CELL_LINES, f"cell {index} has {line_count} lines"


def test_no_code_cell_defines_a_function_or_class() -> None:
    """Notebook code must compose public values, not define local machinery."""
    for index, cell in _code_cells(_read_source_with_jupytext()):
        tree = ast.parse(cell.source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                raise AssertionError(f"cell {index} defines {type(node).__name__} {node.name}")


def test_no_code_cell_stores_a_coordinate_table() -> None:
    """Literal position tables are forbidden; computed layouts are allowed."""
    for index, cell in _code_cells(_read_source_with_jupytext()):
        tree = ast.parse(cell.source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            point_count = sum(_is_point(value) for value in node.values)
            if point_count >= 2:
                raise AssertionError(f"cell {index} contains a coordinate table")


def test_every_code_cell_is_preceded_by_markdown() -> None:
    """Each code cell must be explained by the Markdown before it."""
    notebook = _read_source_with_jupytext()
    for index, _cell in _code_cells(notebook):
        assert index > 0, f"cell {index} is code without a preceding Markdown cell"
        assert notebook.cells[index - 1].cell_type == "markdown", (
            f"cell {index} is not preceded by Markdown"
        )


def test_markdown_uses_latex_for_math_operators() -> None:
    """Prose equations must use LaTeX rather than programming-language operators."""
    notebook = _read_source_with_jupytext()
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "markdown":
            continue
        for operator in ("!=", "==", ">=", "<="):
            assert operator not in cell.source, (
                f"Markdown cell {index} uses {operator!r}; use LaTeX notation instead"
            )


def test_no_direct_markdown_wrappers() -> None:
    """Hook-backed values must not be wrapped only to call to_markdown()."""
    for index, cell in _code_cells(_read_source_with_jupytext()):
        wrapper_lines = _direct_markdown_wrappers(cell.source)
        assert not wrapper_lines, (
            f"cell {index} wraps to_markdown() in display(Markdown(...)) at lines {wrapper_lines}"
        )


def test_direct_markdown_wrapper_guard_detects_injected_wrapper() -> None:
    """The structural guard must catch the pattern it forbids."""
    injected = "display(Markdown(A9_CERTIFICATE.to_markdown()))"

    assert _direct_markdown_wrappers(injected) == [1]
    assert _direct_markdown_wrappers('display(Markdown("caption" + value.to_markdown()))') == []
    assert _direct_markdown_wrappers("display(value)") == []


def test_notebook_has_no_thesis_framing() -> None:
    """The masterclass must stand alone from any thesis-specific narrative."""
    text = _SOURCE_PATH.read_text(encoding="utf-8").lower()

    assert "thesis" not in text


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
