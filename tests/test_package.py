# SPDX-FileCopyrightText: 2026-present Mario Abarca <knkillname@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Smoke tests for the package skeleton."""

import importlib.metadata

import dynkinlab


class TestPackage:
    """Verify the importable package surface and metadata."""

    def test_version_is_a_non_empty_string(self) -> None:
        assert isinstance(dynkinlab.__version__, str)
        assert dynkinlab.__version__

    def test_public_surface_exposes_only_version(self) -> None:
        assert dynkinlab.__all__ == ["__version__"]
        assert set(dynkinlab.__all__) == {"__version__"}

    def test_installed_metadata_matches_package_version(self) -> None:
        assert importlib.metadata.version("dynkinlab") == dynkinlab.__version__
