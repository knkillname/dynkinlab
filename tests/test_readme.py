# SPDX-FileCopyrightText: 2026-present Mario Abarca <knkillname@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Execute the Python examples embedded in the project README."""

import contextlib
import io
import re
import typing
from pathlib import Path

import pytest


class TestReadmeExamples:
    """Verify that every fenced Python block in README.md runs."""

    README: typing.ClassVar[Path] = Path(__file__).resolve().parents[1] / "README.md"

    def test_python_snippets_execute(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Run each README Python snippet in an isolated namespace.

        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            Fixture used to isolate the working directory.
        tmp_path : Path
            Temporary directory used as the working directory.
        """
        monkeypatch.chdir(tmp_path)
        text = self.README.read_text(encoding="utf-8")
        snippets = re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL)
        assert snippets
        for index, snippet in enumerate(snippets):
            namespace: dict[str, object] = {"__name__": "__readme__"}
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(snippet, f"README.md[{index}]", "exec"), namespace)
