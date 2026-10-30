# SPDX-FileCopyrightText: 2026-present Mario Abarca <knkillname@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Tests for D-core cycles and D structural certificates."""

from dataclasses import replace

import pytest

from dynkinlab import Bigraph, BigraphEdge, DynkinType, EdgeStyle, QuasiCartanMatrix
from dynkinlab.edge_multiset import EdgeMultiset
from dynkinlab.structure_d import ACyclizationCertificate, DCore, DCoreGluingCertificate

SOLID = EdgeStyle.SOLID
DOTTED = EdgeStyle.DOTTED


def _edge(source: int, target: int, style: EdgeStyle) -> BigraphEdge:
    """Build a simple undirected edge."""
    return BigraphEdge(source, target, False, style, 1)


def _bigraph(labels: tuple[int, ...], edges: dict[tuple[int, int], int]) -> Bigraph:
    """Build a symmetric bigraph from one-based signed edges."""
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


def _d13_pieces() -> tuple[Bigraph, ...]:
    """Return the five A-type pieces from the thesis D13 example."""
    return (
        _bigraph((1, 2, 8), {(1, 2): 1, (1, 8): 1, (2, 8): 1}),
        _bigraph((2, 3, 6, 7), {(2, 3): -1, (2, 6): 1, (3, 6): -1, (6, 7): -1}),
        _bigraph((3, 4), {(3, 4): 1}),
        _bigraph(
            (4, 5, 9, 10, 11, 12, 13),
            {
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
            },
        ),
        _bigraph((1, 5), {(1, 5): -1}),
    )


def _d13_bigraph() -> Bigraph:
    """Return the expected D13 gluing."""
    edges = {
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
    }
    return _bigraph(tuple(range(1, 14)), edges)


def _d13_core() -> DCore:
    """Return the odd five-cycle core from the thesis."""
    return DCore.cycle(
        (1, 2, 3, 4, 5),
        (DOTTED, SOLID, DOTTED, DOTTED, SOLID),
    )


def _a9_bigraph() -> Bigraph:
    """Return the 9-vertex A-block tree from the thesis."""
    edges = {
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
    }
    return _bigraph(tuple(range(1, 10)), edges)


def _solid_chain(size: int) -> Bigraph:
    """Return a solid A_size path."""
    return _bigraph(
        tuple(range(1, size + 1)),
        {(index, index + 1): -1 for index in range(1, size)},
    )


class TestDCore:
    """Verify D-core cycle validation."""

    def test_odd_five_cycle_is_accepted(self) -> None:
        core = DCore.cycle(
            (1, 2, 3, 4, 5),
            (DOTTED, SOLID, DOTTED, DOTTED, SOLID),
        )

        assert core.labels == (1, 2, 3, 4, 5)
        assert core.cycle_condition() is True
        assert len(core.edges.edges) == 5

    def test_double_link_is_accepted(self) -> None:
        core = DCore.cycle((1, 2), (SOLID, DOTTED))

        assert core.labels == (1, 2)
        assert core.cycle_condition() is True
        assert {edge.style for edge in core.edges.edges} == {SOLID, DOTTED}

    def test_even_dotted_count_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="odd number"):
            DCore.cycle((1, 2, 3), (DOTTED, DOTTED, SOLID))

    def test_chord_is_rejected(self) -> None:
        multiset = EdgeMultiset(
            (1, 2, 3, 4),
            (
                _edge(1, 2, DOTTED),
                _edge(2, 3, SOLID),
                _edge(3, 4, SOLID),
                _edge(4, 1, SOLID),
                _edge(1, 3, SOLID),
            ),
        )

        with pytest.raises(ValueError, match="consecutive"):
            DCore(multiset)

    def test_duplicate_labels_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            DCore.cycle((1, 1, 2), (DOTTED, SOLID, SOLID))

    def test_missing_cycle_edge_is_rejected(self) -> None:
        multiset = EdgeMultiset(
            (1, 2, 3, 4),
            (
                _edge(1, 2, DOTTED),
                _edge(2, 3, SOLID),
                _edge(3, 4, SOLID),
            ),
        )

        with pytest.raises(ValueError, match="consecutive"):
            DCore(multiset)

    def test_double_link_requires_opposite_styles(self) -> None:
        multiset = EdgeMultiset(
            (1, 2),
            (_edge(1, 2, SOLID), _edge(1, 2, SOLID)),
        )

        with pytest.raises(ValueError, match="multiplicity one"):
            DCore(multiset)


class TestDCoreGluing:
    """Verify D-core gluing certificates."""

    def test_d13_gluing_verifies(self) -> None:
        certificate = _d13_core().gluing(_d13_pieces(), labels=tuple(range(1, 14)))

        assert isinstance(certificate, DCoreGluingCertificate)
        assert certificate.verify() is True
        assert certificate.gluing == _d13_bigraph()
        assert certificate.dynkin_type == DynkinType("D", 13)
        assert certificate.gluing.classify().components[0].dynkin_type == DynkinType("D", 13)

    def test_double_link_d4_gluing_verifies(self) -> None:
        core = DCore.cycle((1, 2), (SOLID, DOTTED))
        pieces = (
            _bigraph((1, 2), {(1, 2): -1}),
            _bigraph(
                (1, 2, 3, 4),
                {(1, 2): 1, (1, 3): -1, (2, 3): -1, (3, 4): -1},
            ),
        )

        certificate = core.gluing(pieces, labels=(1, 2, 3, 4))

        assert certificate.verify() is True
        assert certificate.gluing.edges == (
            _edge(1, 3, SOLID),
            _edge(2, 3, SOLID),
            _edge(3, 4, SOLID),
        )
        assert certificate.gluing.classify().components[0].dynkin_type == DynkinType("D", 4)

    def test_non_a_piece_is_rejected(self) -> None:
        pieces = list(_d13_pieces())
        pieces[0] = _bigraph(
            (1, 2, 8, 9),
            {(1, 2): 1, (1, 8): -1, (1, 9): -1, (8, 9): -1},
        )

        with pytest.raises(ValueError, match="A-type"):
            _d13_core().gluing(pieces)

    def test_non_positive_definite_piece_is_rejected(self) -> None:
        pieces = list(_d13_pieces())
        pieces[0] = _bigraph((1, 2), {(1, 2): 2})

        with pytest.raises(ValueError, match="positive definite"):
            _d13_core().gluing(pieces)

    def test_mismatched_core_style_is_rejected(self) -> None:
        core = DCore.cycle(
            (1, 2, 3, 4, 5),
            (SOLID, DOTTED, DOTTED, DOTTED, SOLID),
        )
        pieces = list(_d13_pieces())

        with pytest.raises(ValueError, match="core edge style"):
            core.gluing(pieces)

    def test_articulation_endpoint_is_rejected(self) -> None:
        pieces = list(_d13_pieces())
        pieces[0] = _bigraph(
            (1, 2, 8, 9),
            {(1, 2): 1, (1, 8): -1, (8, 9): -1},
        )

        with pytest.raises(ValueError, match="internal"):
            _d13_core().gluing(pieces)

    def test_unexpected_overlap_is_rejected(self) -> None:
        pieces = list(_d13_pieces())
        pieces[1] = _bigraph(
            (2, 3, 6, 7, 9),
            {
                (2, 3): -1,
                (2, 6): 1,
                (3, 6): -1,
                (6, 7): -1,
                (6, 9): -1,
                (7, 9): 1,
            },
        )

        with pytest.raises(ValueError, match="overlap"):
            _d13_core().gluing(pieces)

    def test_wrong_piece_count_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="number of core edges"):
            _d13_core().gluing(_d13_pieces()[:4])

    def test_markdown_lists_pieces_core_edges_and_a_types(self) -> None:
        certificate = _d13_core().gluing(_d13_pieces(), labels=tuple(range(1, 14)))

        markdown = certificate.to_markdown()

        assert "`D_13`" in markdown
        assert "| F1 | 1 .. 2 | {1, 2, 8} | `A_3` |" in markdown
        assert "| F2 | 2 -- 3 | {2, 3, 6, 7} | `A_4` |" in markdown
        assert "| F5 | 5 -- 1 | {1, 5} | `A_2` |" in markdown
        assert certificate._repr_markdown_() == markdown
        assert "<table" not in repr(certificate)

    def test_tampered_certificate_is_rejected(self) -> None:
        certificate = _d13_core().gluing(_d13_pieces(), labels=tuple(range(1, 14)))

        tampered_piece = replace(
            certificate,
            pieces=(_bigraph((1, 2), {(1, 2): 1}), *certificate.pieces[1:]),
        )
        tampered_gluing = replace(
            certificate,
            gluing=Bigraph(
                QuasiCartanMatrix(
                    tuple(
                        tuple(2 if row == column else 0 for column in range(13))
                        for row in range(13)
                    ),
                ),
                certificate.gluing.labels,
            ),
        )
        tampered_core = replace(
            certificate,
            core=DCore.cycle((1, 2, 3, 4, 5), (SOLID, DOTTED, DOTTED, DOTTED, SOLID)),
        )

        assert tampered_piece.verify() is False
        assert tampered_gluing.verify() is False
        assert tampered_core.verify() is False


class TestACyclization:
    """Verify A-cyclization certificates."""

    def test_a9_to_d8_verifies(self) -> None:
        certificate = _a9_bigraph().a_cyclization(1, 6)

        assert isinstance(certificate, ACyclizationCertificate)
        assert certificate.verify() is True
        assert certificate.path == (1, 3, 5, 6)
        assert certificate.dynkin_type == DynkinType("D", 8)
        assert certificate.result.labels == (1, 2, 3, 4, 5, 7, 8, 9)
        assert certificate.result.classify().components[0].dynkin_type == DynkinType("D", 8)

    def test_adjacent_endpoints_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="length"):
            _a9_bigraph().a_cyclization(1, 2)

    def test_even_dotted_parity_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="odd number"):
            _solid_chain(5).a_cyclization(1, 5)

    def test_articulation_endpoint_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="internal"):
            _a9_bigraph().a_cyclization(1, 3)

    def test_non_a_source_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="A-type"):
            _d13_bigraph().a_cyclization(1, 2)

    def test_disconnected_source_is_rejected(self) -> None:
        source = _bigraph(
            (1, 2, 3, 4, 5),
            {(1, 2): -1, (3, 4): -1, (4, 5): -1},
        )

        with pytest.raises(ValueError, match="connected"):
            source.a_cyclization(1, 2)

    def test_too_few_vertices_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="five"):
            _solid_chain(4).a_cyclization(1, 4)

    def test_markdown_reports_path_result_labels_and_type(self) -> None:
        certificate = _a9_bigraph().a_cyclization(1, 6)

        markdown = certificate.to_markdown()

        assert "| identified | 1 = 6 |" in markdown
        assert "| path | 1 -> 3 -> 5 -> 6 |" in markdown
        assert "| result labels | {1, 2, 3, 4, 5, 7, 8, 9} |" in markdown
        assert "| type | `D_8` |" in markdown
        assert certificate._repr_markdown_() == markdown
        assert "|" not in repr(certificate)

    def test_tampered_certificate_is_rejected(self) -> None:
        certificate = _a9_bigraph().a_cyclization(1, 6)
        identity = Bigraph(
            QuasiCartanMatrix(
                tuple(tuple(2 if row == column else 0 for column in range(8)) for row in range(8)),
            ),
            certificate.result.labels,
        )
        tampered_result = replace(certificate, result=identity)
        tampered_path = replace(certificate, path=(1, 2, 3, 5, 6))

        assert tampered_result.verify() is False
        assert tampered_path.verify() is False
