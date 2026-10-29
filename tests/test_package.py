# SPDX-FileCopyrightText: 2026-present Mario Abarca <knkillname@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Smoke tests for the package skeleton."""

import importlib.metadata

import dynkinlab
from dynkinlab import Inertia, QuadraticForm, QuasiCartanMatrix


class TestPackage:
    """Verify the importable package surface and metadata."""

    def test_version_is_a_non_empty_string(self) -> None:
        assert isinstance(dynkinlab.__version__, str)
        assert dynkinlab.__version__

    def test_public_surface(self) -> None:
        assert dynkinlab.__all__ == [
            "Inertia",
            "QuadraticForm",
            "QuasiCartanMatrix",
            "__version__",
        ]
        assert set(dynkinlab.__all__) == {
            "Inertia",
            "QuadraticForm",
            "QuasiCartanMatrix",
            "__version__",
        }

    def test_quasi_cartan_matrix_is_importable(self) -> None:
        assert QuasiCartanMatrix(((2,),)).dimension == 1

    def test_quadratic_form_is_importable(self) -> None:
        matrix = QuasiCartanMatrix(((2, -1), (-1, 2)))

        assert isinstance(matrix.quadratic_form, QuadraticForm)
        assert matrix.quadratic_form.evaluate((1, 1)) == 1

    def test_inertia_is_importable(self) -> None:
        matrix = QuasiCartanMatrix(((2, -1), (-1, 2)))

        assert isinstance(matrix.inertia, Inertia)
        assert matrix.inertia == Inertia(2, 0, 0)
        assert Inertia.from_matrix(matrix) == Inertia(2, 0, 0)

    def test_installed_metadata_matches_package_version(self) -> None:
        assert importlib.metadata.version("dynkinlab") == dynkinlab.__version__
