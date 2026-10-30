# SPDX-FileCopyrightText: 2026-present Mario Abarca <knkillname@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Tests for label-generic mixed bigraphs and flation witnesses."""

import pytest

from dynkinlab import Bigraph, BigraphEdge, EdgeStyle, FlationSequence, FlationStep
from dynkinlab.quasi_cartan import QuasiCartanMatrix

_A2 = QuasiCartanMatrix(((2, -1), (-1, 2)))
_A3 = QuasiCartanMatrix(((2, -1, 0), (-1, 2, -1), (0, -1, 2)))
_B2 = QuasiCartanMatrix(((2, -1), (-2, 2)))
_DOTTED_A2 = QuasiCartanMatrix(((2, 1), (1, 2)))
_A9_TEXT = """1--2..3--1
3..4--5--3
3--9--4
5..9
5..6..8..5
4--7"""


class TestPathNotation:
    """Verify the compact path notation constructor."""

    def test_a9_path_text_is_parsed(self) -> None:
        bigraph = Bigraph.from_paths(_A9_TEXT, labels=tuple(range(1, 10)))

        assert bigraph.labels == tuple(range(1, 10))
        assert bigraph.matrix.entries[0][1] == -1
        assert bigraph.matrix.entries[1][2] == 1
        assert len(bigraph.edges) == 13
        assert bigraph.a_type_certificate() is not None

    def test_explicit_label_order_is_preserved(self) -> None:
        bigraph = Bigraph.from_paths("b--a", labels=("a", "b"))

        assert bigraph.labels == ("a", "b")
        assert bigraph.matrix.entries == ((2, -1), (-1, 2))

    def test_labels_outside_the_pinned_order_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="outside the explicit label order"):
            Bigraph.from_paths("a--c", labels=("a", "b"))

    def test_labels_are_inferred_in_first_appearance_order(self) -> None:
        bigraph = Bigraph.from_paths("2--3--6..2\n6--7")

        assert bigraph.labels == (2, 3, 6, 7)
        assert bigraph.matrix.dimension == 4

    def test_isolated_vertices_comments_and_separators(self) -> None:
        bigraph = Bigraph.from_paths("solo\n1, --, 2  # a solid edge")

        assert bigraph.labels == ("solo", 1, 2)
        assert bigraph.matrix.entries == ((2, 0, 0), (0, 2, -1), (0, -1, 2))

    @pytest.mark.parametrize(
        ("labels", "match"),
        [(("a", "a"), "unique"), ((["a"], ["b"]), "hashable")],
    )
    def test_invalid_explicit_labels_are_rejected(
        self,
        labels: object,
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            Bigraph.from_paths("a--b", labels=labels)  # type: ignore[arg-type]

    def test_conflicting_declarations_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="conflicting declarations"):
            Bigraph.from_paths("1--2\n1..2")

    @pytest.mark.parametrize(
        "text",
        ["", "1--", "1--1", "1~~2"],
    )
    def test_empty_or_malformed_text_is_rejected(self, text: str) -> None:
        with pytest.raises(ValueError):
            Bigraph.from_paths(text)

    @pytest.mark.parametrize("multiplicity", [0, -1, True, 1.5])
    def test_invalid_arrow_multiplicity_is_rejected(self, multiplicity: object) -> None:
        with pytest.raises(ValueError, match="arrow_multiplicity"):
            Bigraph.from_paths("a->b", arrow_multiplicity=multiplicity)  # type: ignore[arg-type]

    def test_arrow_multiplicity_is_configurable(self) -> None:
        bigraph = Bigraph.from_paths("a->b", arrow_multiplicity=3)

        assert bigraph.matrix.entries == ((2, -1), (-3, 2))
        assert bigraph.matrix.has_faithful_bigraph is True
        assert bigraph.edges == (BigraphEdge("a", "b", True, EdgeStyle.SOLID, 3),)


class TestFaithfulPredicate:
    """Verify the faithful-bigraph entry condition."""

    def test_positive_definite_examples_are_faithful(self) -> None:
        assert _A2.has_faithful_bigraph is True
        assert _A3.has_faithful_bigraph is True

    def test_two_over_four_is_not_faithful(self) -> None:
        matrix = QuasiCartanMatrix(((2, 2), (4, 2)))

        assert matrix.has_faithful_bigraph is False
        bigraph = Bigraph(matrix, (1, 2))
        with pytest.raises(ValueError, match="no faithful bigraph"):
            tuple(bigraph.edges)

    def test_unit_arrow_is_faithful(self) -> None:
        matrix = QuasiCartanMatrix(((2, 1), (3, 2)))

        assert matrix.has_faithful_bigraph is True
        assert Bigraph(matrix, ("a", "b")).edges == (
            BigraphEdge("a", "b", True, EdgeStyle.DOTTED, 3),
        )

    def test_symmetric_dotted_pair_is_faithful(self) -> None:
        matrix = QuasiCartanMatrix(((2, 1), (1, 2)))

        assert matrix.has_faithful_bigraph is True
        assert Bigraph(matrix, ("a", "b")).edges == (
            BigraphEdge("a", "b", False, EdgeStyle.DOTTED, 1),
        )

    def test_zero_pair_is_faithful(self) -> None:
        matrix = QuasiCartanMatrix(((2, 0), (0, 2)))

        assert matrix.has_faithful_bigraph is True
        assert Bigraph(matrix).edges == ()


class TestBigraphEdge:
    """Verify public edge values and validation."""

    def test_edge_is_inspectable_and_hashable(self) -> None:
        edge = BigraphEdge("a", "b", True, EdgeStyle.SOLID, 2)

        assert edge.source == "a"
        assert edge.target == "b"
        assert edge.directed is True
        assert edge.style is EdgeStyle.SOLID
        assert edge.multiplicity == 2
        assert edge.weight == -2
        assert len({edge, edge}) == 1

    @pytest.mark.parametrize(
        "edge",
        [
            ("a", "a", False, EdgeStyle.SOLID, 1),
            ("a", "b", False, "solid", 1),
            ("a", "b", False, EdgeStyle.SOLID, 0),
            ("a", "b", False, EdgeStyle.SOLID, True),
            (["a"], "b", False, EdgeStyle.SOLID, 1),
        ],
    )
    def test_invalid_edges_are_rejected(self, edge: tuple[object, ...]) -> None:
        with pytest.raises(ValueError):
            BigraphEdge(*edge)  # type: ignore[arg-type]


class TestBigraph:
    """Verify labels, edges, and flation delegation."""

    def test_default_labels_follow_indices(self) -> None:
        bigraph = Bigraph(_A2)

        assert bigraph.labels == (0, 1)

    def test_string_labels_are_preserved_and_matrix_is_shared(self) -> None:
        bigraph = Bigraph(_A2, ("left", "right"))

        assert bigraph.labels == ("left", "right")
        assert bigraph.matrix is _A2

    @pytest.mark.parametrize(
        ("labels", "match"),
        [
            (("a",), "entries"),
            (("a", "a"), "unique"),
            (("a", ["b"]), "hashable"),
        ],
    )
    def test_invalid_labels_are_rejected(
        self,
        labels: tuple[object, ...],
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            Bigraph(_A2, labels)  # type: ignore[arg-type]

    def test_matrix_type_is_validated(self) -> None:
        with pytest.raises(ValueError, match="QuasiCartanMatrix"):
            Bigraph(object())  # type: ignore[arg-type]

    def test_symmetric_pair_derives_solid_edge(self) -> None:
        bigraph = Bigraph(_A2)

        assert bigraph.edges == (BigraphEdge(0, 1, False, EdgeStyle.SOLID, 1),)

    def test_symmetric_positive_pair_derives_dotted_edge(self) -> None:
        bigraph = Bigraph(_DOTTED_A2)

        assert bigraph.edges == (BigraphEdge(0, 1, False, EdgeStyle.DOTTED, 1),)

    def test_asymmetric_negative_pair_derives_solid_arrow(self) -> None:
        bigraph = Bigraph(_B2, ("a", "b"))

        assert bigraph.edges == (BigraphEdge("a", "b", True, EdgeStyle.SOLID, 2),)

    def test_asymmetric_positive_pair_derives_dotted_arrow(self) -> None:
        matrix = QuasiCartanMatrix(((2, 1), (2, 2)))
        bigraph = Bigraph(matrix, ("a", "b"))

        assert bigraph.edges == (BigraphEdge("a", "b", True, EdgeStyle.DOTTED, 2),)

    def test_zero_pair_has_no_edge(self) -> None:
        bigraph = Bigraph(QuasiCartanMatrix(((2, 0), (0, 2))))

        assert bigraph.edges == ()

    def test_mixed_pair_uses_smaller_to_larger_direction(self) -> None:
        # The mixed pair (3, 6): A_36 = 1, A_63 = 2.
        bigraph = Bigraph(QuasiCartanMatrix(((2, 1), (2, 2))), (3, 6))

        assert bigraph.edges == (BigraphEdge(3, 6, True, EdgeStyle.DOTTED, 2),)

    def test_flation_flips_style_and_preserves_labels(self) -> None:
        bigraph = Bigraph(_DOTTED_A2, ("left", "right"))

        flated = bigraph.flation("left", "right")

        assert flated.labels == ("left", "right")
        assert flated.edges == (BigraphEdge("left", "right", False, EdgeStyle.SOLID, 1),)

    def test_sign_inversion_is_available(self) -> None:
        bigraph = Bigraph(_DOTTED_A2, ("left", "right"))

        flated = bigraph.flation("left", "left")

        assert flated.edges == (BigraphEdge("left", "right", False, EdgeStyle.SOLID, 1),)

    def test_unknown_label_is_rejected(self) -> None:
        bigraph = Bigraph(_A2, ("a", "b"))

        with pytest.raises(ValueError, match="unknown label"):
            bigraph.flation("a", "c")

    def test_flation_preserves_inertia(self) -> None:
        bigraph = Bigraph(_B2)

        assert bigraph.flation(0, 1).matrix.inertia == bigraph.matrix.inertia

    def test_index_lookup(self) -> None:
        bigraph = Bigraph(_A2, ("a", "b"))

        assert bigraph.index("a") == 0
        assert bigraph.index("b") == 1


class TestFlationSequence:
    """Verify first-class flation witnesses."""

    def test_empty_sequence_is_a_noop(self) -> None:
        bigraph = Bigraph(_A2)
        sequence = FlationSequence()

        assert len(sequence) == 0
        assert sequence.apply(bigraph) == bigraph

    def test_sequence_replays_and_verifies(self) -> None:
        start = Bigraph(_DOTTED_A2, ("a", "b"))
        sequence = FlationSequence((FlationStep("a", "b"), FlationStep("b", "b")))

        final = sequence.apply(start)

        assert len(sequence) == 2
        assert sequence.verify(start, final) is True

    def test_inverse_recovers_start(self) -> None:
        start = Bigraph(_DOTTED_A2, ("a", "b"))
        sequence = FlationSequence((FlationStep("a", "b"), FlationStep("b", "b")))
        final = sequence.apply(start)

        assert sequence.inverse().apply(final) == start

    def test_verification_rejects_wrong_final(self) -> None:
        start = Bigraph(_DOTTED_A2, ("a", "b"))
        sequence = FlationSequence((FlationStep("a", "b"),))

        assert sequence.verify(start, start) is False

    def test_invalid_step_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="FlationStep"):
            FlationSequence((("a", "b"),))  # type: ignore[arg-type]

    def test_unhashable_step_label_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="hashable"):
            FlationStep(["a"], "b")  # type: ignore[arg-type]

    def test_markdown_lists_steps_in_application_order(self) -> None:
        sequence = FlationSequence((FlationStep("a", "b"), FlationStep("b", "b")))

        markdown = sequence.to_markdown()

        assert "| 1 | a | b | flation |" in markdown
        assert "| 2 | b | b | sign inversion |" in markdown
        assert sequence._repr_markdown_() == markdown

    def test_empty_sequence_markdown(self) -> None:
        markdown = FlationSequence().to_markdown()

        assert "| - | - | - | none |" in markdown
