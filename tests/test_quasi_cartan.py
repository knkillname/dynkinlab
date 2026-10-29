# SPDX-FileCopyrightText: 2026-present Mario Abarca <knkillname@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Tests for symmetric quasi-Cartan matrices."""

import pytest

from dynkinlab.quasi_cartan import QuasiCartanMatrix


class TestQuasiCartanMatrix:
    """Verify construction, copying, equality, and validation."""

    def test_valid_matrix_reports_dimension_and_entries(self) -> None:
        matrix = QuasiCartanMatrix(((2, -1), (-1, 2)))

        assert matrix.dimension == 2
        assert matrix.entries == ((2, -1), (-1, 2))

    def test_input_is_copied(self) -> None:
        rows = [[2, -1], [-1, 2]]

        matrix = QuasiCartanMatrix(rows)
        rows[0][0] = 99

        assert matrix.entries == ((2, -1), (-1, 2))

    def test_accepts_generators(self) -> None:
        rows = (row for row in ((2, 0), (0, 2)))

        matrix = QuasiCartanMatrix(rows)

        assert matrix.entries == ((2, 0), (0, 2))

    def test_single_entry_matrix_is_valid(self) -> None:
        matrix = QuasiCartanMatrix(((2,),))

        assert matrix.dimension == 1
        assert matrix.entries == ((2,),)

    def test_value_equality_and_hashing(self) -> None:
        first = QuasiCartanMatrix(((2, -1), (-1, 2)))
        second = QuasiCartanMatrix(((2, -1), (-1, 2)))

        assert first == second
        assert hash(first) == hash(second)
        assert len({first, second}) == 1

    @pytest.mark.parametrize(
        "entries",
        [
            (),
            ((), ()),
            ((2, -1, 0), (-1, 2)),
            ((2, -1), (-1, 2), (0, 0)),
            ((2, "x"), ("x", 2)),
            ((2, True), (True, 2)),
            ((2, 1), (1, 3)),
            ((2, 1), (0, 2)),
            ((2.0, -1), (-1, 2.0)),
            ((2, -1), 5),
            42,
        ],
    )
    def test_invalid_input_raises_value_error(
        self,
        entries: object,
    ) -> None:
        with pytest.raises(ValueError, match=r"matrix|diagonal"):
            QuasiCartanMatrix(entries)  # type: ignore[arg-type]
