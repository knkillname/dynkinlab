# SPDX-FileCopyrightText: 2026-present Mario Abarca <knkillname@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Golden classification tests from the paper and the thesis."""

from dynkinlab import ClassificationStatus, DynkinType, QuasiCartanMatrix


def _matrix(size: int, edges: dict[tuple[int, int], int]) -> QuasiCartanMatrix:
    """Build a symmetric matrix from one-based weighted edges."""
    rows = [[2 if row == column else 0 for column in range(size)] for row in range(size)]
    for (source, target), weight in edges.items():
        rows[source - 1][target - 1] = weight
        rows[target - 1][source - 1] = weight
    return QuasiCartanMatrix(rows)


class TestPaperExamples:
    """Verify the paper's Figure 1 and Example 3.1."""

    def test_figure_one_is_b6(self) -> None:
        matrix = QuasiCartanMatrix(
            (
                (2, -1, 1, 0, 0, 0),
                (-1, 2, -1, 0, 0, 0),
                (1, -1, 2, -1, -1, 1),
                (0, 0, -1, 2, 1, -1),
                (0, 0, -1, 1, 2, -1),
                (0, 0, 2, -2, -2, 2),
            ),
        )

        result = matrix.classify()

        assert result.status is ClassificationStatus.CLASSIFIED
        assert result.components[0].dynkin_type == DynkinType("B", 6)
        assert result.verify() is True

    def test_example_three_one_exhausts_the_bound(self) -> None:
        matrix = QuasiCartanMatrix(
            (
                (2, -1, -1, 1),
                (-1, 2, -1, 0),
                (-1, -1, 2, -1),
                (1, 0, -1, 2),
            ),
        )

        result = matrix.classify()

        assert result.status is ClassificationStatus.NOT_POSITIVE_DEFINITE
        assert result.reason == "bound exhausted"
        assert result.steps == result.bound == 12
        assert result.normal_form is None


class TestThesisExamples:
    """Verify the thesis's D-core and A-block constructions."""

    def test_d_core_gluing_is_d13(self) -> None:
        matrix = _matrix(
            13,
            {
                (1, 2): 1,
                (1, 8): 1,
                (2, 8): 1,
                (2, 3): -1,
                (2, 6): 1,
                (3, 6): -1,
                (6, 7): -1,
                (3, 4): 1,
                (4, 5): 1,
                (4, 9): 1,
                (4, 10): -1,
                (5, 9): 1,
                (5, 10): -1,
                (9, 10): -1,
                (9, 12): -1,
                (9, 13): -1,
                (10, 11): -1,
                (12, 13): 1,
                (1, 5): -1,
            },
        )

        result = matrix.classify()

        assert result.status is ClassificationStatus.CLASSIFIED
        assert result.components[0].dynkin_type == DynkinType("D", 13)
        assert result.verify() is True

    def test_a_block_tree_is_a9(self) -> None:
        matrix = _matrix(
            9,
            {
                (1, 2): -1,
                (1, 3): -1,
                (2, 3): 1,
                (3, 5): -1,
                (3, 9): -1,
                (4, 5): -1,
                (4, 9): -1,
                (3, 4): 1,
                (5, 9): 1,
                (5, 6): 1,
                (5, 8): 1,
                (6, 8): 1,
                (4, 7): -1,
            },
        )

        result = matrix.classify()

        assert result.status is ClassificationStatus.CLASSIFIED
        assert result.components[0].dynkin_type == DynkinType("A", 9)
        assert result.verify() is True
