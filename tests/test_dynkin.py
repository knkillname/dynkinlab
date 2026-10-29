# SPDX-FileCopyrightText: 2026-present Mario Abarca <knkillname@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Tests for finite Dynkin types and Stage 3 graph classification."""

import pytest

from dynkinlab import Bigraph, DynkinType, QuasiCartanMatrix

_TYPES = (
    DynkinType("A", 1),
    DynkinType("A", 2),
    DynkinType("A", 3),
    DynkinType("B", 2),
    DynkinType("B", 3),
    DynkinType("B", 4),
    DynkinType("C", 3),
    DynkinType("C", 4),
    DynkinType("D", 4),
    DynkinType("D", 5),
    DynkinType("E", 6),
    DynkinType("E", 7),
    DynkinType("E", 8),
    DynkinType("F", 4),
    DynkinType("G", 2),
)


def _matrix(size: int, edges: tuple[tuple[int, int], ...]) -> QuasiCartanMatrix:
    """Build a symmetric solid bigraph matrix from edges."""
    rows = [[2 if row == column else 0 for column in range(size)] for row in range(size)]
    for row, column in edges:
        rows[row][column] = -1
        rows[column][row] = -1
    return QuasiCartanMatrix(rows)


class TestDynkinType:
    """Verify finite type values and canonical names."""

    @pytest.mark.parametrize("dynkin_type", _TYPES)
    def test_name_and_rank(self, dynkin_type: DynkinType) -> None:
        assert dynkin_type.rank >= 1
        assert dynkin_type.name == f"{dynkin_type.family}_{dynkin_type.rank}"
        assert str(dynkin_type) == dynkin_type.name

    @pytest.mark.parametrize(
        ("family", "rank"),
        [("A", 0), ("B", 1), ("C", 2), ("D", 3), ("E", 5), ("F", 3), ("G", 3), ("H", 2)],
    )
    def test_invalid_types_are_rejected(self, family: str, rank: int) -> None:
        with pytest.raises(ValueError):
            DynkinType(family, rank)

    def test_values_are_hashable_and_comparable(self) -> None:
        first = DynkinType("A", 3)
        second = DynkinType("A", 3)

        assert first == second
        assert hash(first) == hash(second)
        assert len({first, second}) == 1


class TestCanonicalTypes:
    """Verify canonical matrices and bigraphs round-trip through Stage 3."""

    @pytest.mark.parametrize("dynkin_type", _TYPES)
    def test_canonical_matrix_round_trip(self, dynkin_type: DynkinType) -> None:
        matrix = dynkin_type.cartan_matrix()
        bigraph = Bigraph(matrix)

        assert DynkinType.from_matrix(matrix) == dynkin_type
        assert DynkinType.from_bigraph(bigraph) == dynkin_type

    @pytest.mark.parametrize("dynkin_type", _TYPES)
    def test_canonical_bigraph_has_expected_size(self, dynkin_type: DynkinType) -> None:
        assert dynkin_type.bigraph().dimension == dynkin_type.rank

    def test_graph_classification_is_repeatable(self) -> None:
        matrix = DynkinType("B", 5).cartan_matrix()
        first = DynkinType.from_matrix(matrix)

        assert first == DynkinType("B", 5)
        assert all(DynkinType.from_matrix(matrix) == first for _ in range(5))


class TestStageThree:
    """Verify non-Dynkin graphs are rejected."""

    def test_star_with_four_leaves_is_rejected(self) -> None:
        matrix = _matrix(5, ((0, 1), (0, 2), (0, 3), (0, 4)))

        assert DynkinType.from_bigraph(Bigraph(matrix)) is None

    def test_disconnected_graph_is_rejected(self) -> None:
        matrix = _matrix(4, ((0, 1), (2, 3)))

        assert DynkinType.from_bigraph(Bigraph(matrix)) is None

    def test_two_branch_vertices_are_rejected(self) -> None:
        matrix = _matrix(6, ((0, 1), (0, 2), (0, 3), (3, 4), (3, 5)))

        assert DynkinType.from_bigraph(Bigraph(matrix)) is None

    def test_dotted_graph_is_rejected(self) -> None:
        matrix = QuasiCartanMatrix(((2, 1), (1, 2)))

        assert DynkinType.from_bigraph(Bigraph(matrix)) is None
