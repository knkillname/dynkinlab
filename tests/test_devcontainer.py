# SPDX-FileCopyrightText: 2026-present Mario Abarca <knkillname@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Tests for the repository development-container configuration."""

import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_devcontainer_uses_latest_colab_image() -> None:
    dockerfile = (ROOT / ".devcontainer" / "Dockerfile").read_text(encoding="utf-8")

    assert "FROM us-docker.pkg.dev/colab-images/public/cpu-runtime:latest" in dockerfile
    assert "opencode.ai/v2/install" in dockerfile


def test_devcontainer_configuration_parses() -> None:
    text = (ROOT / ".devcontainer" / "devcontainer.json").read_text(encoding="utf-8")
    config = json.loads(text)

    assert config["remoteUser"] == "ubuntu"
    assert "ghcr.io/devcontainers/features/docker-in-docker:4" in config["features"]


def test_devcontainer_installs_python_and_jupyter_extensions() -> None:
    text = (ROOT / ".devcontainer" / "devcontainer.json").read_text(encoding="utf-8")
    extensions = set(json.loads(text)["customizations"]["vscode"]["extensions"])

    assert {
        "ms-python.debugpy",
        "ms-python.python",
        "ms-python.vscode-pylance",
        "ms-python.vscode-python-envs",
        "ms-toolsai.jupyter",
        "ms-toolsai.jupyter-keymap",
        "ms-toolsai.jupyter-renderers",
        "ms-toolsai.vscode-jupyter-cell-tags",
        "ms-toolsai.vscode-jupyter-slideshow",
    } <= extensions


def test_test_environment_has_a_notebook_kernel() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    extra_dependencies = tomllib.loads(text)["tool"]["hatch"]["envs"]["hatch-test"][
        "extra-dependencies"
    ]

    assert any(dependency.startswith("ipykernel") for dependency in extra_dependencies)
