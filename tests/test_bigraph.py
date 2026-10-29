# SPDX-FileCopyrightText: 2026-present Mario Abarca <knkillname@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Tests for label-generic mixed bigraphs and flation witnesses."""

import pytest

from dynkinlab import Bigraph, BigraphEdge, EdgeStyle, FlationSequence, FlationStep
from dynkinlab.quasi_cartan import QuasiCartanMatrix

_A2 = QuasiCartanMatrix(((2, -1), (-1, 2)))
_B2 = QuasiCartanMatrix(((2, -1), (-2, 2)))
_DOTTED_A2 = QuasiCartanMatrix(((2, 1), (1, 2)))


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
