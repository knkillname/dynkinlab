# SPDX-FileCopyrightText: 2026-present Mario Abarca <knkillname@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Tests for exact inertia."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.strategies import DataObject, DrawFn

from dynkinlab.inertia import Inertia
from dynkinlab.quasi_cartan import QuasiCartanMatrix


@st.composite
def quasi_cartan_matrices(
    draw: DrawFn,
    min_size: int = 1,
    max_size: int = 4,
) -> QuasiCartanMatrix:
    """Generate a small symmetric integer matrix with diagonal 2."""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    entries = [[2] * size for _ in range(size)]
    for row in range(size):
        for column in range(row + 1, size):
            value = draw(st.integers(min_value=-3, max_value=3))
            entries[row][column] = value
            entries[column][row] = value
    return QuasiCartanMatrix(entries)


class TestExactInertia:
    """Verify exact inertia on known matrices and pivot paths."""

    def test_a2_is_positive_definite(self) -> None:
        matrix = QuasiCartanMatrix(((2, -1), (-1, 2)))

        assert Inertia.from_matrix(matrix) == Inertia(2, 0, 0)

    def test_a3_is_positive_definite(self) -> None:
        matrix = QuasiCartanMatrix(
            (
                (2, -1, 0),
                (-1, 2, -1),
                (0, -1, 2),
            ),
        )

        assert Inertia.from_matrix(matrix) == Inertia(3, 0, 0)

    def test_d4_is_positive_definite(self) -> None:
        matrix = QuasiCartanMatrix(
            (
                (2, -1, 0, 0),
                (-1, 2, -1, -1),
                (0, -1, 2, 0),
                (0, -1, 0, 2),
            ),
        )

        assert Inertia.from_matrix(matrix) == Inertia(4, 0, 0)

    def test_e8_is_positive_definite(self) -> None:
        matrix = QuasiCartanMatrix(
            (
                (2, -1, 0, 0, 0, 0, 0, 0),
                (-1, 2, -1, 0, 0, 0, 0, 0),
                (0, -1, 2, -1, 0, 0, 0, -1),
                (0, 0, -1, 2, -1, 0, 0, 0),
                (0, 0, 0, -1, 2, -1, 0, 0),
                (0, 0, 0, 0, -1, 2, -1, 0),
                (0, 0, 0, 0, 0, -1, 2, 0),
                (0, 0, -1, 0, 0, 0, 0, 2),
            ),
        )

        assert Inertia.from_matrix(matrix) == Inertia(8, 0, 0)

    def test_affine_a2_is_degenerate(self) -> None:
        matrix = QuasiCartanMatrix(
            (
                (2, -1, -1),
                (-1, 2, -1),
                (-1, -1, 2),
            ),
        )

        assert Inertia.from_matrix(matrix) == Inertia(2, 0, 1)

    def test_indefinite_matrix(self) -> None:
        matrix = QuasiCartanMatrix(((2, 3), (3, 2)))

        assert Inertia.from_matrix(matrix) == Inertia(1, 1, 0)

    def test_zero_pivot_without_off_diagonal(self) -> None:
        matrix = QuasiCartanMatrix(
            (
                (2, 2, 0),
                (2, 2, 0),
                (0, 0, 2),
            ),
        )

        assert Inertia.from_matrix(matrix) == Inertia(2, 0, 1)

    def test_two_by_two_pivot(self) -> None:
        matrix = QuasiCartanMatrix(
            (
                (2, 2, 0),
                (2, 2, 2),
                (0, 2, 2),
            ),
        )

        assert Inertia.from_matrix(matrix) == Inertia(2, 1, 0)

    def test_two_by_two_pivot_with_swap_and_remaining_block(self) -> None:
        matrix = QuasiCartanMatrix(
            (
                (2, 2, 0, 0),
                (2, 2, 0, 2),
                (0, 0, 2, 0),
                (0, 2, 0, 2),
            ),
        )

        assert Inertia.from_matrix(matrix) == Inertia(3, 1, 0)

    def test_reordering_indices_preserves_inertia(self) -> None:
        matrix = QuasiCartanMatrix(
            (
                (2, -1, 0),
                (-1, 2, -1),
                (0, -1, 2),
            ),
        )
        permuted = QuasiCartanMatrix(
            tuple(tuple(matrix.entries[row][column] for column in (2, 1, 0)) for row in (2, 1, 0)),
        )

        assert Inertia.from_matrix(permuted) == Inertia.from_matrix(matrix)


class TestInertiaInvariants:
    """Verify the derived properties of an inertia result."""

    def test_positive_definite_result(self) -> None:
        inertia = Inertia(2, 0, 0)

        assert inertia.rank == 2
        assert inertia.signature == 2
        assert inertia.is_positive_definite is True
        assert inertia.is_negative_definite is False
        assert inertia.is_degenerate is False

    def test_degenerate_result(self) -> None:
        inertia = Inertia(2, 0, 1)

        assert inertia.rank == 2
        assert inertia.signature == 2
        assert inertia.is_positive_definite is False
        assert inertia.is_negative_definite is False
        assert inertia.is_degenerate is True

    def test_negative_definite_result(self) -> None:
        inertia = Inertia(0, 2, 0)

        assert inertia.rank == 2
        assert inertia.signature == -2
        assert inertia.is_positive_definite is False
        assert inertia.is_negative_definite is True
        assert inertia.is_degenerate is False

    def test_indefinite_result(self) -> None:
        inertia = Inertia(1, 1, 0)

        assert inertia.rank == 2
        assert inertia.signature == 0
        assert inertia.is_positive_definite is False
        assert inertia.is_negative_definite is False
        assert inertia.is_degenerate is False

    @pytest.mark.parametrize(
        "inertia",
        [
            (-1, 0, 0),
            (0, -1, 0),
            (0, 0, -1),
            (1.5, 0, 0),
            (True, 0, 0),
            ("1", 0, 0),
        ],
    )
    def test_invalid_counts_raise_value_error(self, inertia: tuple[object, object, object]) -> None:
        with pytest.raises(ValueError, match=r"positive|negative|zero"):
            Inertia(*inertia)  # type: ignore[arg-type]


class TestInertiaProperties:
    """Verify algebraic invariants with generated matrices."""

    @settings(max_examples=50, deadline=None)
    @given(matrix=quasi_cartan_matrices(), data=st.data())
    def test_permutation_preserves_inertia(
        self,
        matrix: QuasiCartanMatrix,
        data: DataObject,
    ) -> None:
        permutation = data.draw(st.permutations(range(matrix.dimension)))
        permuted = QuasiCartanMatrix(
            tuple(
                tuple(matrix.entries[row][column] for column in permutation) for row in permutation
            ),
        )

        assert Inertia.from_matrix(permuted) == Inertia.from_matrix(matrix)

    @settings(max_examples=50, deadline=None)
    @given(matrix=quasi_cartan_matrices(), data=st.data())
    def test_diagonal_sign_congruence_preserves_inertia(
        self,
        matrix: QuasiCartanMatrix,
        data: DataObject,
    ) -> None:
        signs = data.draw(
            st.lists(
                st.sampled_from((-1, 1)),
                min_size=matrix.dimension,
                max_size=matrix.dimension,
            ),
        )
        congruent = QuasiCartanMatrix(
            tuple(
                tuple(
                    signs[row] * signs[column] * matrix.entries[row][column]
                    for column in range(matrix.dimension)
                )
                for row in range(matrix.dimension)
            ),
        )

        assert Inertia.from_matrix(congruent) == Inertia.from_matrix(matrix)

    @settings(max_examples=50, deadline=None)
    @given(
        first=quasi_cartan_matrices(max_size=3),
        second=quasi_cartan_matrices(max_size=3),
    )
    def test_block_diagonal_adds_inertia(
        self,
        first: QuasiCartanMatrix,
        second: QuasiCartanMatrix,
    ) -> None:
        size = first.dimension + second.dimension
        entries = [[0] * size for _ in range(size)]
        for row in range(first.dimension):
            for column in range(first.dimension):
                entries[row][column] = first.entries[row][column]
        for row in range(second.dimension):
            for column in range(second.dimension):
                entries[first.dimension + row][first.dimension + column] = second.entries[row][
                    column
                ]
        matrix = QuasiCartanMatrix(entries)
        first_inertia = Inertia.from_matrix(first)
        second_inertia = Inertia.from_matrix(second)
        expected = Inertia(
            first_inertia.positive + second_inertia.positive,
            first_inertia.negative + second_inertia.negative,
            first_inertia.zero + second_inertia.zero,
        )

        assert Inertia.from_matrix(matrix) == expected

    @settings(max_examples=50, deadline=None)
    @given(matrix=quasi_cartan_matrices(), data=st.data())
    def test_mutated_diagonal_is_rejected(
        self,
        matrix: QuasiCartanMatrix,
        data: DataObject,
    ) -> None:
        row = data.draw(st.integers(min_value=0, max_value=matrix.dimension - 1))
        replacement = data.draw(
            st.integers(min_value=-3, max_value=5).filter(lambda value: value != 2),
        )
        entries = [list(row_values) for row_values in matrix.entries]
        entries[row][row] = replacement

        with pytest.raises(ValueError, match="diagonal"):
            QuasiCartanMatrix(entries)

    @settings(max_examples=50, deadline=None)
    @given(matrix=quasi_cartan_matrices(min_size=2, max_size=4), data=st.data())
    def test_asymmetric_mutation_is_rejected(
        self,
        matrix: QuasiCartanMatrix,
        data: DataObject,
    ) -> None:
        row = data.draw(st.integers(min_value=0, max_value=matrix.dimension - 1))
        column = data.draw(
            st.integers(min_value=0, max_value=matrix.dimension - 1).filter(
                lambda value: value != row,
            ),
        )
        entries = [list(row_values) for row_values in matrix.entries]
        entries[row][column] += 1

        with pytest.raises(ValueError, match="symmetric"):
            QuasiCartanMatrix.symmetric(entries)
