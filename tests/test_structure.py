# SPDX-FileCopyrightText: 2026-present Mario Abarca <knkillname@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Tests for structural bigraph inspection."""

import pytest

from dynkinlab import (
    ABlockPartition,
    Bigraph,
    Block,
    BlockCutTree,
    DynkinType,
    QuasiCartanMatrix,
)


def _matrix(size: int, edges: dict[tuple[int, int], int]) -> QuasiCartanMatrix:
    """Build a symmetric matrix from one-based weighted edges."""
    rows = [[2 if row == column else 0 for column in range(size)] for row in range(size)]
    for (source, target), weight in edges.items():
        rows[source - 1][target - 1] = weight
        rows[target - 1][source - 1] = weight
    return QuasiCartanMatrix(rows)


def _a9_bigraph() -> Bigraph:
    """Return the 9-vertex A-block tree from the thesis examples."""
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
    return Bigraph(_matrix(9, edges), tuple(range(1, 10)))


def _d13_bigraph() -> Bigraph:
    """Return the 13-vertex D-core gluing from the thesis examples."""
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
    return Bigraph(_matrix(13, edges), tuple(range(1, 14)))


def _disjoint_a2_a3() -> Bigraph:
    """Return a disconnected A2 plus A3 bigraph with labels 1 through 5."""
    rows = [[2 if row == column else 0 for column in range(5)] for row in range(5)]
    rows[0][1] = rows[1][0] = -1
    rows[2][3] = rows[3][2] = -1
    rows[3][4] = rows[4][3] = -1
    return Bigraph(QuasiCartanMatrix(rows), (1, 2, 3, 4, 5))


class TestNetworkXConversion:
    """Verify the undirected NetworkX view of a bigraph."""

    def test_mixed_bigraph_preserves_labels_and_edge_attributes(self) -> None:
        bigraph = Bigraph(
            QuasiCartanMatrix(((2, -1), (-2, 2))),
            ("a", "b"),
        )

        graph = bigraph.to_networkx()

        assert set(graph.nodes) == {"a", "b"}
        assert graph.number_of_edges() == 1
        attributes = graph.edges["a", "b"]
        assert attributes["style"] == "solid"
        assert attributes["directed"] is True
        assert attributes["multiplicity"] == 2
        assert attributes["weight"] == -2

    def test_isolated_vertices_are_retained(self) -> None:
        bigraph = Bigraph(
            QuasiCartanMatrix(((2, 0), (0, 2))),
            ("a", "b"),
        )

        graph = bigraph.to_networkx()

        assert set(graph.nodes) == {"a", "b"}
        assert graph.number_of_edges() == 0


class TestBlockCutForest:
    """Verify biconnected blocks and component forests."""

    def test_a9_blocks_and_articulation_points(self) -> None:
        bigraph = _a9_bigraph()

        forest = bigraph.block_cut_forest()

        assert len(forest.trees) == 1
        tree = forest.trees[0]
        assert {frozenset(block.labels) for block in tree.blocks} == {
            frozenset({1, 2, 3}),
            frozenset({3, 4, 5, 9}),
            frozenset({4, 7}),
            frozenset({5, 6, 8}),
        }
        assert tree.articulation_points == (3, 4, 5)
        assert tree.verify(bigraph) is True
        assert forest.verify(bigraph) is True

    def test_bridge_is_a_block(self) -> None:
        edges = {
            (1, 2): -1,
            (2, 3): -1,
            (1, 3): -1,
            (3, 4): -1,
            (4, 5): -1,
            (5, 6): -1,
            (4, 6): -1,
        }
        bigraph = Bigraph(_matrix(6, edges), tuple(range(1, 7)))

        tree = bigraph.block_cut_forest().trees[0]

        assert {frozenset(block.labels) for block in tree.blocks} == {
            frozenset({1, 2, 3}),
            frozenset({3, 4}),
            frozenset({4, 5, 6}),
        }
        assert tree.articulation_points == (3, 4)

    def test_single_vertex_component(self) -> None:
        bigraph = Bigraph(QuasiCartanMatrix(((2,),)), ("solo",))

        tree = bigraph.block_cut_forest().trees[0]

        assert tree.labels == ("solo",)
        assert tree.blocks == ()
        assert tree.articulation_points == ()
        assert tree.verify(bigraph) is True
        assert set(tree.to_networkx().nodes) == {"solo"}

    def test_single_edge_block(self) -> None:
        bigraph = Bigraph(QuasiCartanMatrix(((2, -1), (-1, 2))), (1, 2))

        tree = bigraph.block_cut_forest().trees[0]

        assert [block.labels for block in tree.blocks] == [(1, 2)]
        assert tree.articulation_points == ()

    def test_disconnected_components_have_separate_trees(self) -> None:
        bigraph = _disjoint_a2_a3()

        forest = bigraph.block_cut_forest()

        assert {tree.labels for tree in forest.trees} == {(1, 2), (3, 4, 5)}
        assert forest.verify(bigraph) is True

    def test_tree_networkx_view_has_block_and_vertex_nodes(self) -> None:
        tree = _a9_bigraph().block_cut_forest().trees[0]

        graph = tree.to_networkx()

        block_nodes = [node for node in graph.nodes if graph.nodes[node]["kind"] == "block"]
        vertex_nodes = [node for node in graph.nodes if graph.nodes[node]["kind"] == "vertex"]
        assert len(block_nodes) == 4
        assert len(vertex_nodes) == 9
        assert graph.number_of_edges() == 12

    def test_tree_rejects_a_different_bigraph(self) -> None:
        tree = _a9_bigraph().block_cut_forest().trees[0]

        assert tree.verify(_d13_bigraph()) is False

    def test_block_cut_conversion_rejects_block_labels(self) -> None:
        block = Block((1, 2), Bigraph(QuasiCartanMatrix(((2, -1), (-1, 2))), (1, 2)))
        tree = BlockCutTree((block,), (), ())

        with pytest.raises(ValueError, match="non-Block"):
            tree.to_networkx()


class TestABlockPartition:
    """Verify A-block partitions and the solid-graph bipartition rule."""

    def test_standard_a_block_is_accepted(self) -> None:
        labels = ("a", "b", "c", "d")
        rows = [[2 if row == column else 0 for column in range(4)] for row in range(4)]
        for row in range(4):
            for column in range(row + 1, 4):
                cross = (row < 2) != (column < 2)
                weight = -1 if cross else 1
                rows[row][column] = rows[column][row] = weight
        bigraph = Bigraph(QuasiCartanMatrix(rows), labels)

        partition = bigraph.is_a_block(labels)

        assert partition is not None
        assert {partition.x, partition.y} == {("a", "b"), ("c", "d")}
        assert partition.verify(bigraph, labels) is True

    def test_all_dotted_block_has_empty_second_side(self) -> None:
        labels = ("a", "b", "c")
        rows = [[2 if row == column else 1 for column in range(3)] for row in range(3)]
        bigraph = Bigraph(QuasiCartanMatrix(rows), labels)

        partition = bigraph.is_a_block(labels)

        assert partition is not None
        assert partition.x == labels
        assert partition.y == ()
        assert partition.verify(bigraph, labels) is True

    def test_single_solid_edge_is_an_a_block(self) -> None:
        bigraph = Bigraph(QuasiCartanMatrix(((2, -1), (-1, 2))), ("a", "b"))

        partition = bigraph.is_a_block(("a", "b"))

        assert partition is not None
        assert {partition.x, partition.y} == {("a",), ("b",)}
        assert partition.verify(bigraph, ("a", "b")) is True

    def test_solid_triangle_is_rejected(self) -> None:
        labels = ("a", "b", "c")
        rows = [[2 if row == column else -1 for column in range(3)] for row in range(3)]
        bigraph = Bigraph(QuasiCartanMatrix(rows), labels)

        assert bigraph.is_a_block(labels) is None

    def test_disconnected_solid_graph_is_rejected(self) -> None:
        labels = ("a", "b", "c", "d")
        rows = [[2 if row == column else 1 for column in range(4)] for row in range(4)]
        for row, column in ((0, 1), (2, 3)):
            rows[row][column] = rows[column][row] = -1
        bigraph = Bigraph(QuasiCartanMatrix(rows), labels)

        assert bigraph.is_a_block(labels) is None

    def test_arrow_is_not_an_a_block(self) -> None:
        bigraph = Bigraph(QuasiCartanMatrix(((2, -1), (-2, 2))), ("a", "b"))

        assert bigraph.is_a_block(("a", "b")) is None

    def test_unknown_labels_are_rejected(self) -> None:
        bigraph = Bigraph(QuasiCartanMatrix(((2, -1), (-1, 2))), ("a", "b"))

        with pytest.raises(ValueError, match="unknown label"):
            bigraph.is_a_block(("a", "c"))

    def test_overlapping_partition_sides_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="disjoint"):
            ABlockPartition(("a",), ("a",))


class TestATypeCertificate:
    """Verify componentwise A-type certificates."""

    def test_a9_certificate_verifies(self) -> None:
        bigraph = _a9_bigraph()

        certificate = bigraph.a_type_certificate()

        assert certificate is not None
        assert certificate.verify(bigraph) is True
        assert len(certificate.components) == 1
        component = certificate.components[0]
        assert component.labels == tuple(range(1, 10))
        assert component.dynkin_type == DynkinType("A", 9)
        assert len(component.partitions) == 4
        for block, partition in zip(
            component.tree.blocks,
            component.partitions,
            strict=True,
        ):
            assert partition.verify(block.bigraph, block.labels) is True

    def test_d13_is_not_a_type_and_still_classifies_as_d(self) -> None:
        bigraph = _d13_bigraph()

        assert bigraph.a_type_certificate() is None
        assert bigraph.is_type_a() is False
        result = bigraph.classify()
        assert result.components[0].dynkin_type == DynkinType("D", 13)

    def test_a1_and_a2_are_certified(self) -> None:
        for dynkin_type in (DynkinType("A", 1), DynkinType("A", 2)):
            bigraph = dynkin_type.bigraph()

            certificate = bigraph.a_type_certificate()

            assert certificate is not None
            assert certificate.verify(bigraph) is True
            assert certificate.components[0].dynkin_type == dynkin_type

    def test_disjoint_union_is_certified_componentwise(self) -> None:
        bigraph = _disjoint_a2_a3()

        certificate = bigraph.a_type_certificate()

        assert certificate is not None
        assert certificate.verify(bigraph) is True
        assert [component.dynkin_type for component in certificate.components] == [
            DynkinType("A", 2),
            DynkinType("A", 3),
        ]

    def test_certificate_rejects_a_different_bigraph(self) -> None:
        bigraph = _a9_bigraph()
        certificate = bigraph.a_type_certificate()
        assert certificate is not None

        assert certificate.verify(_d13_bigraph()) is False
        assert certificate.components[0].verify(_d13_bigraph()) is False

    def test_structural_and_classifier_results_are_independent(self) -> None:
        a9 = _a9_bigraph()
        d13 = _d13_bigraph()

        assert a9.is_type_a() is True
        assert d13.a_type_certificate() is None
        result = d13.classify()
        assert result.components[0].dynkin_type == DynkinType("D", 13)
