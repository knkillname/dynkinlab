# SPDX-FileCopyrightText: 2026-present Mario Abarca <knkillname@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Structural bigraph inspection: blocks, forests, and A-type certificates."""

from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from itertools import combinations
from typing import cast

import networkx as nx

from dynkinlab.bigraph import Bigraph
from dynkinlab.dynkin import DynkinType
from dynkinlab.quasi_cartan import QuasiCartanMatrix

_CUT_VERTEX_BLOCK_COUNT = 2


def _normalized_labels(bigraph: Bigraph, labels: Iterable[Hashable]) -> tuple[Hashable, ...]:
    """Normalize a label collection to bigraph-index order."""
    try:
        values = tuple(labels)
    except TypeError as error:
        raise ValueError("labels must be an iterable") from error
    if not values:
        raise ValueError("labels must not be empty")
    order = {label: index for index, label in enumerate(bigraph.labels)}
    seen: set[Hashable] = set()
    for label in values:
        if label not in order:
            raise ValueError(f"unknown label: {label!r}")
        if label in seen:
            raise ValueError("labels must be unique")
        seen.add(label)
    return tuple(sorted(values, key=order.__getitem__))


def _induced_bigraph(bigraph: Bigraph, labels: tuple[Hashable, ...]) -> Bigraph:
    """Return the principal sub-bigraph induced by ``labels``."""
    indices = [bigraph.index(label) for label in labels]
    rows = tuple(
        tuple(bigraph.matrix.entries[row][column] for column in indices) for row in indices
    )
    return Bigraph(QuasiCartanMatrix(rows), labels)


@dataclass(frozen=True)
class Block:
    """An immutable biconnected block of a bigraph.

    Parameters
    ----------
    labels : tuple[Hashable, ...]
        Vertex labels in bigraph-index order.
    bigraph : Bigraph
        Induced sub-bigraph over exactly those labels.
    """

    labels: tuple[Hashable, ...]
    bigraph: Bigraph

    def __post_init__(self) -> None:
        """Validate the block labels and induced bigraph."""
        if not self.labels:
            raise ValueError("block labels must not be empty")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("block labels must be unique")
        if self.bigraph.labels != self.labels:
            raise ValueError("block bigraph labels must match block labels")


@dataclass(frozen=True)
class BlockCutTree:
    """The block-cut tree of one connected bigraph component.

    Parameters
    ----------
    labels : tuple[Hashable, ...]
        Component vertex labels in bigraph-index order.
    blocks : tuple[Block, ...]
        Biconnected blocks in deterministic order.
    articulation_points : tuple[Hashable, ...]
        Cut vertices in bigraph-index order.
    """

    labels: tuple[Hashable, ...]
    blocks: tuple[Block, ...]
    articulation_points: tuple[Hashable, ...]

    def __post_init__(self) -> None:
        """Validate the tree against its blocks and labels."""
        if not self.labels:
            raise ValueError("tree labels must not be empty")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("tree labels must be unique")
        label_set = set(self.labels)
        block_labels: set[Hashable] = set()
        incidence = dict.fromkeys(self.labels, 0)
        for block in self.blocks:
            if not set(block.labels) <= label_set:
                raise ValueError("block labels must belong to the tree")
            for label in block.labels:
                block_labels.add(label)
                incidence[label] += 1
        if self.blocks and block_labels != label_set:
            raise ValueError("every tree label must belong to a block")
        expected = tuple(label for label in self.labels if incidence[label] > 1)
        if self.articulation_points != expected:
            raise ValueError("articulation points do not match the block incidence")

    def to_networkx(self) -> "nx.Graph[Hashable]":
        """Return a bipartite NetworkX view of the block-cut tree.

        Block nodes are ``Block`` values with ``kind="block"``; vertex nodes
        are the original labels with ``kind="vertex"``.

        Returns
        -------
        networkx.Graph
            Bipartite graph whose edges join each block to its vertices.
        """
        if any(isinstance(label, Block) for label in self.labels):
            raise ValueError("block-cut conversion requires non-Block labels")
        graph = nx.Graph()
        graph.add_nodes_from(self.labels, kind="vertex")
        for block in self.blocks:
            graph.add_node(block, kind="block")
            for label in block.labels:
                graph.add_edge(block, label)
        return graph

    def verify(self, bigraph: Bigraph) -> bool:
        """Return whether this tree matches the bigraph's block-cut structure."""
        try:
            forest = build_block_cut_forest(bigraph)
        except ValueError:
            return False
        return any(set(tree.labels) == set(self.labels) and tree == self for tree in forest.trees)


@dataclass(frozen=True)
class BlockCutForest:
    """The block-cut forest of a bigraph, one tree per connected component.

    Parameters
    ----------
    trees : tuple[BlockCutTree, ...]
        Component trees in bigraph-index order.
    """

    trees: tuple[BlockCutTree, ...]

    def __post_init__(self) -> None:
        """Validate that component trees are vertex-disjoint."""
        labels: set[Hashable] = set()
        for tree in self.trees:
            overlap = labels & set(tree.labels)
            if overlap:
                raise ValueError("block-cut trees must have disjoint labels")
            labels |= set(tree.labels)

    def to_networkx(self) -> "nx.Graph[Hashable]":
        """Return the disjoint union of the component block-cut trees."""
        graph = nx.Graph()
        for tree in self.trees:
            tree_graph = tree.to_networkx()
            graph.add_nodes_from(tree_graph.nodes(data=True))
            graph.add_edges_from(tree_graph.edges(data=True))
        return graph

    def verify(self, bigraph: Bigraph) -> bool:
        """Return whether this forest matches the bigraph's components."""
        try:
            expected = build_block_cut_forest(bigraph)
        except ValueError:
            return False
        return self == expected


def _component_labels(
    bigraph: Bigraph,
    graph: "nx.Graph[Hashable]",
) -> tuple[tuple[Hashable, ...], ...]:
    """Return connected component labels in deterministic order."""
    order = {label: index for index, label in enumerate(bigraph.labels)}
    components = [
        tuple(sorted(cast(Iterable[Hashable], component), key=order.__getitem__))
        for component in nx.connected_components(graph)
    ]
    return tuple(sorted(components, key=lambda labels: order[labels[0]]))


def _build_tree(bigraph: Bigraph, labels: tuple[Hashable, ...]) -> BlockCutTree:
    """Build one block-cut tree for a connected component."""
    if len(labels) == 1:
        return BlockCutTree(labels=labels, blocks=(), articulation_points=())
    order = {label: index for index, label in enumerate(bigraph.labels)}
    graph = bigraph.to_networkx().subgraph(labels)
    blocks: list[Block] = []
    for raw_block in nx.biconnected_components(graph):
        block_labels = tuple(
            sorted(cast(Iterable[Hashable], raw_block), key=order.__getitem__),
        )
        blocks.append(Block(block_labels, _induced_bigraph(bigraph, block_labels)))
    blocks.sort(key=lambda block: tuple(order[label] for label in block.labels))
    articulation_points = tuple(
        sorted(cast(Iterable[Hashable], nx.articulation_points(graph)), key=order.__getitem__),
    )
    return BlockCutTree(
        labels=labels,
        blocks=tuple(blocks),
        articulation_points=articulation_points,
    )


def build_block_cut_forest(bigraph: Bigraph) -> BlockCutForest:
    """Build the block-cut forest of a bigraph.

    Parameters
    ----------
    bigraph : Bigraph
        Input bigraph.

    Returns
    -------
    BlockCutForest
        One block-cut tree per connected component.
    """
    graph = bigraph.to_networkx()
    return BlockCutForest(
        tuple(_build_tree(bigraph, labels) for labels in _component_labels(bigraph, graph)),
    )


@dataclass(frozen=True)
class ABlockPartition:
    """Two disjoint sides witnessing that a block is an A-block.

    Parameters
    ----------
    x : tuple[Hashable, ...]
        First side of the partition.
    y : tuple[Hashable, ...]
        Second side of the partition.
    """

    x: tuple[Hashable, ...]
    y: tuple[Hashable, ...]

    def __post_init__(self) -> None:
        """Validate disjoint, duplicate-free sides."""
        if len(set(self.x)) != len(self.x) or len(set(self.y)) != len(self.y):
            raise ValueError("A-block sides must be unique")
        if set(self.x) & set(self.y):
            raise ValueError("A-block sides must be disjoint")
        if not self.x and not self.y:
            raise ValueError("A-block partition must not be empty")

    def verify(
        self,
        bigraph: Bigraph,
        labels: Iterable[Hashable] | None = None,
    ) -> bool:
        """Return whether this partition satisfies the A-block rule.

        Parameters
        ----------
        bigraph : Bigraph
            Parent or induced bigraph.
        labels : Iterable[Hashable] | None
            Block labels. Defaults to all bigraph labels.

        Returns
        -------
        bool
            True when every cross pair is solid and every within-side pair is
            dotted.
        """
        try:
            block_labels = bigraph.labels if labels is None else _normalized_labels(bigraph, labels)
        except ValueError:
            return False
        if set(self.x) | set(self.y) != set(block_labels):
            return False
        for first, second in combinations(block_labels, 2):
            first_index = bigraph.index(first)
            second_index = bigraph.index(second)
            forward = bigraph.matrix.entries[first_index][second_index]
            backward = bigraph.matrix.entries[second_index][first_index]
            if forward != backward or forward == 0:
                return False
            same_side = (first in self.x and second in self.x) or (
                first in self.y and second in self.y
            )
            if same_side and forward <= 0:
                return False
            if not same_side and forward >= 0:
                return False
        return True


def a_block_partition(bigraph: Bigraph, labels: Iterable[Hashable]) -> ABlockPartition | None:
    """Return an A-block partition for an induced vertex set, if one exists.

    Parameters
    ----------
    bigraph : Bigraph
        Parent bigraph.
    labels : Iterable[Hashable]
        Candidate block labels.

    Returns
    -------
    ABlockPartition | None
        A verified partition, or ``None`` when the induced set is not an
        A-block.

    Raises
    ------
    ValueError
        If the labels are empty, duplicated, or unknown.
    """
    block_labels = _normalized_labels(bigraph, labels)
    for first, second in combinations(block_labels, 2):
        first_index = bigraph.index(first)
        second_index = bigraph.index(second)
        forward = bigraph.matrix.entries[first_index][second_index]
        backward = bigraph.matrix.entries[second_index][first_index]
        if forward != backward or forward == 0:
            return None

    solid = nx.Graph()
    solid.add_nodes_from(block_labels)
    for first, second in combinations(block_labels, 2):
        if bigraph.matrix.entries[bigraph.index(first)][bigraph.index(second)] < 0:
            solid.add_edge(first, second)

    if solid.number_of_edges() == 0:
        partition = ABlockPartition(block_labels, ())
        return partition if partition.verify(bigraph, block_labels) else None
    if not nx.is_connected(solid):
        return None
    try:
        colors = nx.bipartite.color(solid)
    except nx.NetworkXError:
        return None
    x = tuple(label for label in block_labels if colors[label] == 0)
    y = tuple(label for label in block_labels if colors[label] == 1)
    partition = ABlockPartition(x, y)
    return partition if partition.verify(bigraph, block_labels) else None


@dataclass(frozen=True)
class ATypeComponentCertificate:
    """A verifiable certificate that one connected component is A-type.

    Parameters
    ----------
    labels : tuple[Hashable, ...]
        Component labels.
    dynkin_type : DynkinType
        The ``A_n`` type of the component.
    tree : BlockCutTree
        The component's block-cut tree.
    partitions : tuple[ABlockPartition, ...]
        One A-block partition per block, in tree order.
    """

    labels: tuple[Hashable, ...]
    dynkin_type: DynkinType
    tree: BlockCutTree
    partitions: tuple[ABlockPartition, ...]

    def __post_init__(self) -> None:
        """Validate the component type and partition count."""
        if self.dynkin_type.family != "A" or self.dynkin_type.rank != len(self.labels):
            raise ValueError("component certificate must describe A_n for its labels")
        if self.tree.labels != self.labels:
            raise ValueError("tree labels must match component labels")
        if len(self.partitions) != len(self.tree.blocks):
            raise ValueError("each block must have one partition")

    def verify(self, bigraph: Bigraph) -> bool:
        """Return whether this component certificate matches the bigraph."""
        try:
            expected = build_a_type_certificate(bigraph)
        except ValueError:
            return False
        if expected is None:
            return False
        return any(component == self for component in expected.components)


@dataclass(frozen=True)
class ATypeCertificate:
    """A verifiable certificate that every component is A-type.

    Parameters
    ----------
    components : tuple[ATypeComponentCertificate, ...]
        Per-component certificates.
    """

    components: tuple[ATypeComponentCertificate, ...]

    def __post_init__(self) -> None:
        """Validate that component certificates are vertex-disjoint."""
        if not self.components:
            raise ValueError("certificate must have at least one component")
        labels: set[Hashable] = set()
        for component in self.components:
            overlap = labels & set(component.labels)
            if overlap:
                raise ValueError("component certificates must have disjoint labels")
            labels |= set(component.labels)

    def verify(self, bigraph: Bigraph) -> bool:
        """Return whether every component certificate matches the bigraph."""
        try:
            expected = build_a_type_certificate(bigraph)
        except ValueError:
            return False
        return expected == self


def build_a_type_certificate(bigraph: Bigraph) -> ATypeCertificate | None:
    """Build an A-type certificate, or return ``None`` when a component is not A-type.

    Parameters
    ----------
    bigraph : Bigraph
        Input bigraph.

    Returns
    -------
    ATypeCertificate | None
        A certificate for every component, or ``None`` when at least one
        component fails the A-block or articulation-degree conditions.
    """
    forest = build_block_cut_forest(bigraph)
    components: list[ATypeComponentCertificate] = []
    for tree in forest.trees:
        if len(tree.labels) == 1:
            components.append(
                ATypeComponentCertificate(
                    labels=tree.labels,
                    dynkin_type=DynkinType("A", 1),
                    tree=tree,
                    partitions=(),
                ),
            )
            continue
        partitions: list[ABlockPartition] = []
        for block in tree.blocks:
            partition = a_block_partition(block.bigraph, block.labels)
            if partition is None:
                return None
            partitions.append(partition)
        for label in tree.articulation_points:
            degree = sum(label in block.labels for block in tree.blocks)
            if degree != _CUT_VERTEX_BLOCK_COUNT:
                return None
        components.append(
            ATypeComponentCertificate(
                labels=tree.labels,
                dynkin_type=DynkinType("A", len(tree.labels)),
                tree=tree,
                partitions=tuple(partitions),
            ),
        )
    return ATypeCertificate(tuple(components))
