# SPDX-FileCopyrightText: 2026-present Mario Abarca <knkillname@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Tests for integer quadratic forms."""

import pytest

from dynkinlab.quadratic_form import QuadraticForm
from dynkinlab.quasi_cartan import QuasiCartanMatrix


class TestQuadraticForm:
    """Verify exact evaluation and vector validation."""

    def test_known_values(self) -> None:
        form = QuadraticForm(QuasiCartanMatrix(((2, -1), (-1, 2))))

        assert form.evaluate((0, 0)) == 0
        assert form.evaluate((1, 0)) == 1
        assert form.evaluate((0, 1)) == 1
        assert form.evaluate((1, 1)) == 1
        assert form.evaluate((2, 1)) == 3

    def test_negative_coordinates(self) -> None:
        form = QuadraticForm(QuasiCartanMatrix(((2, -1), (-1, 2))))

        assert form.evaluate((-1, 1)) == 3
        assert form.evaluate((-1, -1)) == 1

    def test_positive_off_diagonal_entry(self) -> None:
        form = QuadraticForm(QuasiCartanMatrix(((2, 3), (3, 2))))

        assert form.evaluate((1, 1)) == 5

    def test_three_dimensional_form(self) -> None:
        matrix = QuasiCartanMatrix(((2, -1, -1), (-1, 2, -1), (-1, -1, 2)))
        form = QuadraticForm(matrix)

        assert form.evaluate((1, 1, 0)) == 1
        assert form.evaluate((1, 1, 1)) == 0
        assert form.evaluate((2, 0, 0)) == 4

    def test_result_is_an_integer(self) -> None:
        form = QuadraticForm(QuasiCartanMatrix(((2, -1), (-1, 2))))

        result = form.evaluate((1, 1))

        assert isinstance(result, int)

    def test_form_preserves_matrix_identity(self) -> None:
        matrix = QuasiCartanMatrix(((2, -1), (-1, 2)))

        assert QuadraticForm(matrix).matrix is matrix

    @pytest.mark.parametrize(
        "vector",
        [
            (),
            (1,),
            (1, 1, 1),
            (1.5, 1),
            ("1", 1),
            (True, 1),
            42,
        ],
    )
    def test_invalid_vectors_raise_value_error(self, vector: object) -> None:
        form = QuadraticForm(QuasiCartanMatrix(((2, -1), (-1, 2))))

        with pytest.raises(ValueError, match="vector"):
            form.evaluate(vector)  # type: ignore[arg-type]
