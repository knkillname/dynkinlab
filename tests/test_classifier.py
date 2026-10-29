# SPDX-FileCopyrightText: 2026-present Mario Abarca <knkillname@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Tests for the exact cubic Dynkin classifier."""

import pytest

from dynkinlab import (
    Bigraph,
    ClassificationStatus,
    ComponentClassification,
    DynkinType,
    QuasiCartanMatrix,
)

_TYPES = (
    DynkinType("A", 1),
    DynkinType("A", 2),
    DynkinType("A", 3),
    DynkinType("B", 2),
    DynkinType("B", 3),
    DynkinType("B", 4),
    DynkinType("B", 5),
    DynkinType("C", 3),
    DynkinType("C", 4),
    DynkinType("C", 5),
    DynkinType("D", 4),
    DynkinType("D", 5),
    DynkinType("E", 6),
    DynkinType("E", 7),
    DynkinType("E", 8),
    DynkinType("F", 4),
    DynkinType("G", 2),
)

_INDEFINITE = (
    (2, 1, 1, 0, 1, -1),
    (1, 2, -1, 1, -1, 1),
    (1, -1, 2, 1, -1, 0),
    (0, 1, 1, 2, 0, -1),
    (1, -1, -1, 0, 2, 0),
    (-1, 1, 0, -1, 0, 2),
)


class TestCanonicalClassification:
    """Verify every canonical finite type."""

    @pytest.mark.parametrize("dynkin_type", _TYPES)
    def test_canonical_matrix_is_classified(self, dynkin_type: DynkinType) -> None:
        matrix = dynkin_type.cartan_matrix()
        result = matrix.classify()

        assert result.status is ClassificationStatus.CLASSIFIED
        assert result.components == (
            ComponentClassification(tuple(range(dynkin_type.rank)), dynkin_type),
        )
        assert result.inertia.is_positive_definite
        assert result.normal_form is not None
        assert result.verify() is True
        assert result.steps == len(result.witness)
        assert result.bound >= result.steps

    @pytest.mark.parametrize("dynkin_type", _TYPES)
    def test_witness_inverse_recovers_input(self, dynkin_type: DynkinType) -> None:
        result = dynkin_type.cartan_matrix().classify()

        assert result.normal_form is not None
        assert result.witness.inverse().apply(result.normal_form) == result.input


class TestG2:
    """Verify the G2 exception and normalization."""

    def test_standard_orientation(self) -> None:
        result = QuasiCartanMatrix(((2, -1), (-3, 2))).classify()

        assert result.status is ClassificationStatus.CLASSIFIED
        assert result.components[0].dynkin_type == DynkinType("G", 2)
        assert result.steps == 0

    def test_transposed_orientation(self) -> None:
        result = QuasiCartanMatrix(((2, -3), (-1, 2))).classify()

        assert result.status is ClassificationStatus.CLASSIFIED
        assert result.components[0].dynkin_type == DynkinType("G", 2)

    def test_positive_g2_is_normalized(self) -> None:
        result = QuasiCartanMatrix(((2, 1), (3, 2))).classify()

        assert result.status is ClassificationStatus.CLASSIFIED
        assert result.components[0].dynkin_type == DynkinType("G", 2)
        assert result.steps == 1
        assert result.verify() is True


class TestNonFinite:
    """Verify admissibility and positivity failures do not hang."""

    def test_entry_magnitude_three_is_not_admissible(self) -> None:
        result = QuasiCartanMatrix(((2, 3), (3, 2))).classify()

        assert result.status is ClassificationStatus.NOT_ADMISSIBLE
        assert "magnitude" in result.reason

    def test_parallel_edges_are_not_admissible(self) -> None:
        result = QuasiCartanMatrix(((2, 2), (2, 2))).classify()

        assert result.status is ClassificationStatus.NOT_ADMISSIBLE
        assert "parallel" in result.reason

    def test_affine_a2_is_not_positive_definite(self) -> None:
        result = QuasiCartanMatrix(
            ((2, -1, -1), (-1, 2, -1), (-1, -1, 2)),
        ).classify()

        assert result.status is ClassificationStatus.NOT_POSITIVE_DEFINITE
        assert result.inertia.is_degenerate
        assert result.normal_form is None
        assert result.verify() is False

    def test_indefinite_component_is_rejected_before_flations(self) -> None:
        result = QuasiCartanMatrix(_INDEFINITE).classify()

        assert result.status is ClassificationStatus.NOT_POSITIVE_DEFINITE
        assert result.reason == "indefinite"
        assert result.steps == 0
        assert len(result.witness) == 0
        assert result.inertia.negative > 0

    def test_negative_definite_is_not_admissible(self) -> None:
        result = QuasiCartanMatrix(((2, 3), (3, 2))).classify()

        assert result.status is ClassificationStatus.NOT_ADMISSIBLE

    def test_no_distinguished_vertex_is_reported(self) -> None:
        matrix = QuasiCartanMatrix(
            (
                (2, -1, 0, 0, 0),
                (-1, 2, -2, 0, 0),
                (0, -1, 2, -1, 0),
                (0, 0, -1, 2, -1),
                (0, 0, 0, -1, 2),
            ),
        )

        result = matrix.classify()

        assert result.status is ClassificationStatus.NOT_POSITIVE_DEFINITE
        assert result.reason == "no distinguished vertex"


class TestDisconnected:
    """Verify component splitting and labels."""

    def test_a2_plus_e6(self) -> None:
        first = DynkinType("A", 2).cartan_matrix()
        second = DynkinType("E", 6).cartan_matrix()
        size = first.dimension + second.dimension
        rows = [[0] * size for _ in range(size)]
        for row in range(first.dimension):
            for column in range(first.dimension):
                rows[row][column] = first.entries[row][column]
        for row in range(second.dimension):
            for column in range(second.dimension):
                rows[first.dimension + row][first.dimension + column] = second.entries[row][column]
        matrix = QuasiCartanMatrix(rows)

        result = matrix.classify()

        assert result.status is ClassificationStatus.CLASSIFIED
        assert [component.dynkin_type for component in result.components] == [
            DynkinType("A", 2),
            DynkinType("E", 6),
        ]
        assert result.components[0].labels == (0, 1)
        assert result.components[1].labels == (2, 3, 4, 5, 6, 7)
        assert result.verify() is True

    def test_labels_are_preserved(self) -> None:
        first = DynkinType("A", 2).cartan_matrix()
        second = DynkinType("A", 2).cartan_matrix()
        size = first.dimension + second.dimension
        rows = [[0] * size for _ in range(size)]
        for row in range(first.dimension):
            for column in range(first.dimension):
                rows[row][column] = first.entries[row][column]
        for row in range(second.dimension):
            for column in range(second.dimension):
                rows[first.dimension + row][first.dimension + column] = second.entries[row][column]
        bigraph = Bigraph(QuasiCartanMatrix(rows), ("a", "b", "c", "d"))

        result = bigraph.classify()

        assert result.components[0].labels == ("a", "b")
        assert result.components[1].labels == ("c", "d")

    def test_duplicate_types_and_isolated_vertex(self) -> None:
        first = DynkinType("A", 2).cartan_matrix()
        rows = [[0] * 4 for _ in range(4)]
        for offset in (0, 2):
            for row in range(2):
                for column in range(2):
                    rows[offset + row][offset + column] = first.entries[row][column]

        result = QuasiCartanMatrix(rows).classify()

        assert [component.dynkin_type for component in result.components] == [
            DynkinType("A", 2),
            DynkinType("A", 2),
        ]
        assert QuasiCartanMatrix(((2,),)).classify().components[0].dynkin_type == DynkinType("A", 1)

    def test_indefinite_component_preserves_earlier_classification(self) -> None:
        first = DynkinType("A", 2).cartan_matrix()
        second = QuasiCartanMatrix(_INDEFINITE)
        size = first.dimension + second.dimension
        rows = [[0] * size for _ in range(size)]
        for row in range(first.dimension):
            for column in range(first.dimension):
                rows[row][column] = first.entries[row][column]
        for row in range(second.dimension):
            for column in range(second.dimension):
                rows[first.dimension + row][first.dimension + column] = second.entries[row][column]

        result = QuasiCartanMatrix(rows).classify()

        assert result.status is ClassificationStatus.NOT_POSITIVE_DEFINITE
        assert result.reason == "indefinite"
        assert result.components == (ComponentClassification((0, 1), DynkinType("A", 2)),)
        assert result.components[0].labels == (0, 1)
        assert result.steps == 0


class TestLabelsAndWitness:
    """Verify label-preserving witnesses."""

    def test_dotted_bigraph_uses_labels_in_witness(self) -> None:
        bigraph = Bigraph(QuasiCartanMatrix(((2, 1), (1, 2))), ("left", "right"))

        result = bigraph.classify()

        assert result.status is ClassificationStatus.CLASSIFIED
        assert result.components[0].labels == ("left", "right")
        assert result.witness.steps[0].source == "left"
        assert result.witness.steps[0].target == "right"
        assert result.verify() is True


class TestFlationGenerated:
    """Verify positive definite inputs generated by flations keep their type."""

    @pytest.mark.parametrize(
        "dynkin_type",
        [
            DynkinType("A", 3),
            DynkinType("D", 5),
            DynkinType("E", 6),
            DynkinType("B", 5),
            DynkinType("C", 5),
            DynkinType("F", 4),
            DynkinType("G", 2),
        ],
    )
    def test_flated_canonical_type_is_recovered(self, dynkin_type: DynkinType) -> None:
        bigraph = dynkin_type.bigraph()
        for edge in bigraph.edges:
            bigraph = bigraph.flation(edge.source, edge.target)

        result = bigraph.classify()

        assert result.status is ClassificationStatus.CLASSIFIED
        assert result.components[0].dynkin_type == dynkin_type
        assert result.verify() is True

    def test_b5_witness_contains_sign_inversion(self) -> None:
        bigraph = DynkinType("B", 5).bigraph()
        for edge in bigraph.edges:
            bigraph = bigraph.flation(edge.source, edge.target)

        result = bigraph.classify()

        assert any(step.source == step.target for step in result.witness.steps)

    def test_permuted_matrix_keeps_type(self) -> None:
        matrix = DynkinType("B", 5).cartan_matrix()
        permutation = (2, 0, 4, 1, 3)
        permuted = QuasiCartanMatrix(
            tuple(
                tuple(matrix.entries[permutation[row]][permutation[column]] for column in range(5))
                for row in range(5)
            ),
        )

        result = permuted.classify()

        assert result.status is ClassificationStatus.CLASSIFIED
        assert result.components[0].dynkin_type == DynkinType("B", 5)
