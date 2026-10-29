# SPDX-FileCopyrightText: 2026-present Mario Abarca <knkillname@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Tests for bigraph sums and vertex identification."""

import pytest

from dynkinlab import Bigraph, EdgeStyle, QuasiCartanMatrix


def _bigraph(labels: tuple[str, ...], edges: dict[tuple[str, str], int]) -> Bigraph:
    """Build a symmetric bigraph from string labels and signed edges."""
    order = {label: index for index, label in enumerate(labels)}
    rows = [
        [2 if row == column else 0 for column in range(len(labels))] for row in range(len(labels))
    ]
    for (source, target), weight in edges.items():
        source_index = order[source]
        target_index = order[target]
        rows[source_index][target_index] = weight
        rows[target_index][source_index] = weight
    return Bigraph(QuasiCartanMatrix(rows), labels)


class TestBigraphSum:
    """Verify shared-label sums and cancellation."""

    def test_disjoint_pieces_form_a_disjoint_union(self) -> None:
        first = _bigraph(("a", "b"), {("a", "b"): -1})
        second = _bigraph(("c", "d", "e"), {("c", "d"): -1, ("d", "e"): -1})

        result = Bigraph.sum(first, second)

        assert result.labels == ("a", "b", "c", "d", "e")
        assert len(result.edges) == 3

    def test_shared_labels_are_identified(self) -> None:
        first = _bigraph(("a", "b"), {("a", "b"): -1})
        second = _bigraph(("b", "c"), {("b", "c"): 1})

        result = Bigraph.sum(first, second)

        assert result.labels == ("a", "b", "c")
        assert {(edge.source, edge.target, edge.style) for edge in result.edges} == {
            ("a", "b", EdgeStyle.SOLID),
            ("b", "c", EdgeStyle.DOTTED),
        }

    def test_opposite_edges_cancel_in_the_sum(self) -> None:
        first = _bigraph(("a", "b"), {("a", "b"): -1})
        second = _bigraph(("a", "b"), {("a", "b"): 1})

        result = Bigraph.sum(first, second)

        assert result.edges == ()
        assert result.matrix.entries == ((2, 0), (0, 2))

    def test_explicit_label_order_is_preserved(self) -> None:
        first = _bigraph(("a", "b"), {("a", "b"): -1})
        second = _bigraph(("b", "c"), {("b", "c"): 1})

        result = Bigraph.sum(first, second, labels=("c", "a", "b"))

        assert result.labels == ("c", "a", "b")

    def test_incomplete_explicit_order_is_rejected(self) -> None:
        first = _bigraph(("a", "b"), {("a", "b"): -1})
        second = _bigraph(("b", "c"), {("b", "c"): 1})

        with pytest.raises(ValueError, match="cover exactly"):
            Bigraph.sum(first, second, labels=("a", "b"))

    def test_mixed_piece_is_rejected(self) -> None:
        symmetric = _bigraph(("a", "b"), {("a", "b"): -1})
        mixed = Bigraph(QuasiCartanMatrix(((2, -1), (-2, 2))), ("a", "b"))

        with pytest.raises(ValueError, match="symmetric"):
            Bigraph.sum(symmetric, mixed)

    def test_no_pieces_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            Bigraph.sum()


class TestBigraphIdentify:
    """Verify vertex identification and simplification."""

    def test_non_adjacent_vertices_merge(self) -> None:
        bigraph = _bigraph(
            ("a", "b", "c"),
            {("a", "b"): 1, ("c", "b"): 1},
        )

        result = bigraph.identify("a", "c")

        assert result.labels == ("a", "b")
        assert len(result.edges) == 1
        assert result.edges[0].multiplicity == 2

    def test_adjacent_vertices_drop_the_loop(self) -> None:
        bigraph = _bigraph(
            ("a", "b", "c"),
            {("a", "b"): 1, ("b", "c"): -1},
        )

        result = bigraph.identify("a", "b")

        assert result.labels == ("a", "c")
        assert {(edge.source, edge.target, edge.style) for edge in result.edges} == {
            ("a", "c", EdgeStyle.SOLID),
        }

    def test_opposite_edges_cancel_after_merging(self) -> None:
        bigraph = _bigraph(
            ("a", "b", "c"),
            {("a", "b"): 1, ("c", "b"): -1},
        )

        result = bigraph.identify("a", "c")

        assert result.edges == ()

    def test_label_order_retains_the_source(self) -> None:
        bigraph = _bigraph(("a", "b", "c"), {})

        result = bigraph.identify("c", "a")

        assert result.labels == ("b", "c")

    def test_invalid_labels_are_rejected(self) -> None:
        bigraph = _bigraph(("a", "b"), {})

        with pytest.raises(ValueError, match="distinct"):
            bigraph.identify("a", "a")
        with pytest.raises(ValueError, match="unknown label"):
            bigraph.identify("a", "z")
