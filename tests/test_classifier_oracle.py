# SPDX-FileCopyrightText: 2026-present Mario Abarca <knkillname@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Property-test the optimized classifier against a naive reference."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.strategies import DrawFn

from dynkinlab import ClassificationStatus, DynkinType, QuasiCartanMatrix
from dynkinlab.symmetrizer import Symmetrizer

_TYPES = (
    DynkinType("A", 1),
    DynkinType("A", 13),
    DynkinType("B", 4),
    DynkinType("C", 4),
    DynkinType("D", 13),
    DynkinType("E", 6),
    DynkinType("F", 4),
    DynkinType("G", 2),
)


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


class TestDomainEquivalence:
    """Verify the int64 classifier across the supported input families."""

    @pytest.mark.parametrize("dynkin_type", _TYPES)
    def test_canonical_types_classify(self, dynkin_type: DynkinType) -> None:
        result = dynkin_type.cartan_matrix().classify()

        assert result.status is ClassificationStatus.CLASSIFIED
        assert result.components[0].dynkin_type == dynkin_type
        assert result.steps == len(result.witness)
        if dynkin_type != DynkinType("G", 2):
            assert result.steps <= result.bound
        assert result.verify() is True

    @pytest.mark.parametrize("dynkin_type", _TYPES)
    def test_flated_types_classify(self, dynkin_type: DynkinType) -> None:
        bigraph = dynkin_type.bigraph()
        for edge in bigraph.edges:
            bigraph = bigraph.flation(edge.source, edge.target)

        result = bigraph.classify()

        assert result.status is ClassificationStatus.CLASSIFIED
        assert result.components[0].dynkin_type == dynkin_type
        assert result.steps == len(result.witness)
        if dynkin_type != DynkinType("G", 2):
            assert result.steps <= result.bound
        assert result.verify() is True

    @pytest.mark.parametrize("dynkin_type", _TYPES)
    def test_permuted_types_classify(self, dynkin_type: DynkinType) -> None:
        matrix = dynkin_type.cartan_matrix()
        permutation = tuple(reversed(range(dynkin_type.rank)))
        permuted = QuasiCartanMatrix(
            tuple(
                tuple(matrix.entries[row][column] for column in permutation) for row in permutation
            ),
        )

        result = permuted.classify()

        assert result.status is ClassificationStatus.CLASSIFIED
        assert result.components[0].dynkin_type == dynkin_type
        assert result.steps == len(result.witness)
        if dynkin_type != DynkinType("G", 2):
            assert result.steps <= result.bound
        assert result.verify() is True

    @pytest.mark.parametrize("size", [3, 4, 6, 13])
    def test_affine_cycles_terminate(self, size: int) -> None:
        rows = [[2 if row == column else 0 for column in range(size)] for row in range(size)]
        for row in range(size):
            column = (row + 1) % size
            rows[row][column] = -1
            rows[column][row] = -1

        result = QuasiCartanMatrix(rows).classify()

        assert result.status is ClassificationStatus.NOT_POSITIVE_DEFINITE
        assert result.inertia.is_degenerate
        assert result.steps <= result.bound
