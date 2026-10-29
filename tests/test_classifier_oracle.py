# SPDX-FileCopyrightText: 2026-present Mario Abarca <knkillname@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Property-test the optimized classifier against a naive reference."""

from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.strategies import DrawFn

from dynkinlab import ClassificationStatus, DynkinType, QuasiCartanMatrix
from dynkinlab.symmetrizer import Symmetrizer


@st.composite
def small_symmetric_matrices(draw: DrawFn) -> QuasiCartanMatrix:
    """Generate a small symmetric quasi-Cartan matrix with entries in {-1, 0, 1}."""
    size = draw(st.integers(min_value=1, max_value=4))
    rows = [[2] * size for _ in range(size)]
    for row in range(size):
        for column in range(row + 1, size):
            value = draw(st.integers(min_value=-1, max_value=1))
            rows[row][column] = value
            rows[column][row] = value
    return QuasiCartanMatrix(rows)


def _naive_component(matrix: QuasiCartanMatrix, bound: int = 500) -> DynkinType | None:
    """Reference: naive generalized inflations with a conservative bound."""
    current = QuasiCartanMatrix(matrix.entries)
    for _ in range(bound):
        pair = None
        for row in range(current.dimension):
            for column in range(current.dimension):
                if row != column and current.entries[row][column] > 0:
                    pair = (row, column)
                    break
            if pair is not None:
                break
        if pair is None:
            return DynkinType.from_matrix(current)
        current = current.flation(*pair)
    return None


def _naive_types(matrix: QuasiCartanMatrix) -> tuple[DynkinType, ...] | None:
    """Reference classification for every connected component."""
    result: list[DynkinType] = []
    for component in Symmetrizer.components(matrix.entries):
        submatrix = QuasiCartanMatrix(
            tuple(tuple(matrix.entries[row][column] for column in component) for row in component),
        )
        dynkin_type = _naive_component(submatrix)
        if dynkin_type is None:
            return None
        result.append(dynkin_type)
    return tuple(result)


class TestNaiveOracle:
    """Compare the optimized classifier with the naive inflations reference."""

    @settings(max_examples=40, deadline=None)
    @given(matrix=small_symmetric_matrices())
    def test_classified_result_matches_naive(self, matrix: QuasiCartanMatrix) -> None:
        result = matrix.classify()
        naive = _naive_types(matrix)

        if result.status is ClassificationStatus.CLASSIFIED:
            assert naive == tuple(component.dynkin_type for component in result.components)
        else:
            assert naive is None
