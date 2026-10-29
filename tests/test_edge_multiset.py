# SPDX-FileCopyrightText: 2026-present Mario Abarca <knkillname@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Tests for labeled edge multisets."""

import pytest

from dynkinlab import Bigraph, BigraphEdge, EdgeStyle, QuasiCartanMatrix
from dynkinlab.edge_multiset import EdgeMultiset


def _edge(
    source: str,
    target: str,
    style: EdgeStyle,
    multiplicity: int = 1,
    directed: bool = False,
) -> BigraphEdge:
    """Build an edge for multiset tests."""
    return BigraphEdge(source, target, directed, style, multiplicity)


class TestEdgeMultisetConstruction:
    """Verify validation and normalization."""

    def test_opposite_parallel_edges_are_preserved(self) -> None:
        multiset = EdgeMultiset(
            ("a", "b"),
            (
                _edge("a", "b", EdgeStyle.SOLID),
                _edge("a", "b", EdgeStyle.DOTTED),
            ),
        )

        assert len(multiset.edges) == 2
        assert {(edge.style, edge.multiplicity) for edge in multiset.edges} == {
            (EdgeStyle.SOLID, 1),
            (EdgeStyle.DOTTED, 1),
        }

    def test_same_style_parallel_edges_combine(self) -> None:
        multiset = EdgeMultiset(
            ("a", "b"),
            (
                _edge("a", "b", EdgeStyle.SOLID),
                _edge("a", "b", EdgeStyle.SOLID),
            ),
        )

        assert multiset.edges == (_edge("a", "b", EdgeStyle.SOLID, 2),)

    def test_directed_edges_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="undirected"):
            EdgeMultiset(("a", "b"), (_edge("a", "b", EdgeStyle.SOLID, directed=True),))

    def test_unknown_labels_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown label"):
            EdgeMultiset(("a",), (_edge("a", "b", EdgeStyle.SOLID),))

    def test_empty_labels_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            EdgeMultiset((), ())

    def test_duplicate_labels_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            EdgeMultiset(("a", "a"), ())

    def test_from_bigraph_preserves_labels_and_edges(self) -> None:
        bigraph = Bigraph(QuasiCartanMatrix(((2, -1), (-1, 2))), ("a", "b"))

        multiset = EdgeMultiset.from_bigraph(bigraph)

        assert multiset.labels == ("a", "b")
        assert multiset.edges == (_edge("a", "b", EdgeStyle.SOLID),)

    def test_from_mixed_bigraph_is_rejected(self) -> None:
        bigraph = Bigraph(QuasiCartanMatrix(((2, -1), (-2, 2))), ("a", "b"))

        with pytest.raises(ValueError, match="undirected"):
            EdgeMultiset.from_bigraph(bigraph)


class TestEdgeMultisetArithmetic:
    """Verify addition and simplification."""

    def test_addition_unions_labels_in_first_appearance_order(self) -> None:
        first = EdgeMultiset(("a", "b"), (_edge("a", "b", EdgeStyle.SOLID),))
        second = EdgeMultiset(("b", "c"), (_edge("b", "c", EdgeStyle.DOTTED),))

        result = first + second

        assert result.labels == ("a", "b", "c")
        assert set(result.edges) == {
            _edge("a", "b", EdgeStyle.SOLID),
            _edge("b", "c", EdgeStyle.DOTTED),
        }

    def test_addition_keeps_opposite_edges_until_simplification(self) -> None:
        first = EdgeMultiset(("a", "b"), (_edge("a", "b", EdgeStyle.SOLID),))
        second = EdgeMultiset(("a", "b"), (_edge("a", "b", EdgeStyle.DOTTED),))

        result = first + second

        assert len(result.edges) == 2

    def test_simplification_cancels_equal_opposite_pairs(self) -> None:
        multiset = EdgeMultiset(
            ("a", "b"),
            (
                _edge("a", "b", EdgeStyle.SOLID),
                _edge("a", "b", EdgeStyle.DOTTED),
            ),
        )

        bigraph = multiset.simplify()

        assert bigraph.labels == ("a", "b")
        assert bigraph.edges == ()

    def test_simplification_leaves_the_excess_style(self) -> None:
        multiset = EdgeMultiset(
            ("a", "b"),
            (
                _edge("a", "b", EdgeStyle.SOLID, 3),
                _edge("a", "b", EdgeStyle.DOTTED),
            ),
        )

        bigraph = multiset.simplify()

        assert bigraph.edges == (_edge("a", "b", EdgeStyle.SOLID, 2),)

    def test_simplification_retains_isolated_labels(self) -> None:
        multiset = EdgeMultiset(("a", "b"), ())

        bigraph = multiset.simplify()

        assert bigraph.labels == ("a", "b")
        assert bigraph.edges == ()

    def test_identify_non_adjacent_vertices(self) -> None:
        multiset = EdgeMultiset(
            ("a", "b", "c"),
            (
                _edge("a", "b", EdgeStyle.DOTTED),
                _edge("c", "b", EdgeStyle.DOTTED),
            ),
        )

        result = multiset.identify("a", "c")

        assert result.labels == ("a", "b")
        assert result.edges == (_edge("a", "b", EdgeStyle.DOTTED, 2),)

    def test_identify_adjacent_vertices_drops_the_loop(self) -> None:
        multiset = EdgeMultiset(
            ("a", "b", "c"),
            (
                _edge("a", "b", EdgeStyle.DOTTED),
                _edge("b", "c", EdgeStyle.SOLID),
            ),
        )

        result = multiset.identify("a", "b")

        assert result.labels == ("a", "c")
        assert result.edges == (_edge("a", "c", EdgeStyle.SOLID),)

    def test_identify_cancels_opposite_edges(self) -> None:
        multiset = EdgeMultiset(
            ("a", "b", "c"),
            (
                _edge("a", "b", EdgeStyle.DOTTED),
                _edge("c", "b", EdgeStyle.SOLID),
            ),
        )

        result = multiset.identify("a", "c")

        assert result.simplify().edges == ()

    def test_identify_retains_the_source_position(self) -> None:
        multiset = EdgeMultiset(("a", "b", "c"), ())

        result = multiset.identify("c", "a")

        assert result.labels == ("b", "c")

    def test_identify_rejects_equal_or_unknown_labels(self) -> None:
        multiset = EdgeMultiset(("a", "b"), ())

        with pytest.raises(ValueError, match="distinct"):
            multiset.identify("a", "a")
        with pytest.raises(ValueError, match="unknown label"):
            multiset.identify("a", "z")
