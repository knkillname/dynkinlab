# SPDX-FileCopyrightText: 2026-present Mario Abarca <knkillname@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Structural D-type certificates: D-core cycles and cyclizations."""

from collections.abc import Hashable, Iterable
from dataclasses import dataclass

import networkx as nx

from dynkinlab.bigraph import Bigraph, BigraphEdge, EdgeStyle
from dynkinlab.dynkin import DynkinType
from dynkinlab.edge_multiset import EdgeMultiset

_MIN_CORE_SIZE = 2
_DOUBLE_LINK_SIZE = 2
_MIN_D_GLUING_SIZE = 4
_MIN_A_CYCLIZATION_SIZE = 5
_MIN_PATH_LENGTH = 2


@dataclass(frozen=True)
class DCore:
    """A cyclic D-core represented as an edge multiset.

    For two labels the core is a two-edge double link: one solid and one
    dotted edge between the same pair. For three or more labels the core is a
    simple cycle with one edge from each label to the next.

    Parameters
    ----------
    edges : EdgeMultiset
        The core edges. Its labels define the cyclic order.

    Raises
    ------
    ValueError
        If the labels are not a valid cyclic sequence, an edge is missing or
        extra, an edge is repeated with multiplicity other than one, or the
        number of dotted edges is even.
    """

    edges: EdgeMultiset

    def __post_init__(self) -> None:
        """Validate the core structure and cycle condition."""
        if not isinstance(self.edges, EdgeMultiset):
            raise ValueError("core edges must be an EdgeMultiset")
        labels = self.edges.labels
        size = len(labels)
        if size < _MIN_CORE_SIZE:
            raise ValueError("a D-core requires at least two labels")
        if any(edge.multiplicity != 1 for edge in self.edges.edges):
            raise ValueError("D-core edges must have multiplicity one")

        if size == _DOUBLE_LINK_SIZE:
            pair = frozenset((labels[0], labels[1]))
            if len(self.edges.edges) != _DOUBLE_LINK_SIZE:
                raise ValueError("a two-label D-core requires exactly two edges")
            if any(frozenset((edge.source, edge.target)) != pair for edge in self.edges.edges):
                raise ValueError("a two-label D-core must join the same two labels")
            if {edge.style for edge in self.edges.edges} != {EdgeStyle.SOLID, EdgeStyle.DOTTED}:
                raise ValueError("a two-label D-core requires one solid and one dotted edge")
        else:
            expected = {
                frozenset((labels[index], labels[(index + 1) % size])) for index in range(size)
            }
            actual = [frozenset((edge.source, edge.target)) for edge in self.edges.edges]
            if len(actual) != size or set(actual) != expected:
                raise ValueError("D-core edges must be exactly the consecutive cycle edges")

        if not self.cycle_condition():
            raise ValueError("D-core requires an odd number of dotted edges")

    @property
    def labels(self) -> tuple[Hashable, ...]:
        """Return the core labels in cyclic order."""
        return self.edges.labels

    @classmethod
    def cycle(cls, labels: Iterable[Hashable], styles: Iterable[EdgeStyle]) -> "DCore":
        """Build a D-core from labels and one style per cycle position.

        Parameters
        ----------
        labels : Iterable[Hashable]
            Distinct labels in cyclic order.
        styles : Iterable[EdgeStyle]
            One style per cycle position. For two labels, the two styles are
            the two parallel edges of the double link.

        Returns
        -------
        DCore
            A validated D-core.

        Raises
        ------
        ValueError
            If the labels or styles do not form a valid D-core.
        """
        try:
            core_labels = tuple(labels)
        except TypeError as error:
            raise ValueError("labels must be an iterable") from error
        try:
            core_styles = tuple(styles)
        except TypeError as error:
            raise ValueError("styles must be an iterable") from error
        if len(core_labels) < _MIN_CORE_SIZE:
            raise ValueError("a D-core requires at least two labels")
        try:
            if len(set(core_labels)) != len(core_labels):
                raise ValueError("labels must be unique")
        except TypeError as error:
            raise ValueError("labels must be hashable") from error
        if len(core_styles) != len(core_labels):
            raise ValueError("each core label requires one cycle style")
        edges: list[BigraphEdge] = []
        size = len(core_labels)
        for index, style in enumerate(core_styles):
            if not isinstance(style, EdgeStyle):
                raise ValueError("styles must contain EdgeStyle values")
            source = core_labels[index]
            target = core_labels[(index + 1) % size]
            edges.append(BigraphEdge(source, target, False, style, 1))
        return cls(EdgeMultiset(core_labels, tuple(edges)))

    @staticmethod
    def _edges_between(
        source: Hashable,
        target: Hashable,
        edges: tuple[BigraphEdge, ...],
    ) -> tuple[BigraphEdge, ...]:
        """Return edges joining two labels, ignoring orientation."""
        pair = frozenset((source, target))
        return tuple(edge for edge in edges if frozenset((edge.source, edge.target)) == pair)

    def _validate_gluing_piece(
        self,
        index: int,
        piece: Bigraph,
        core_style_set: set[EdgeStyle],
    ) -> EdgeStyle | None:
        """Validate one piece and return its double-link style, if any."""
        if not isinstance(piece, Bigraph):
            raise ValueError("pieces must be Bigraph values")
        forest = piece.block_cut_forest()
        if len(forest.trees) != 1:
            raise ValueError("each piece must be connected")
        tree = forest.trees[0]
        if piece.a_type_certificate() is None:
            raise ValueError("each piece must be A-type")
        if not piece.matrix.inertia.is_positive_definite:
            raise ValueError("each piece must be positive definite")

        size = len(self.labels)
        first = self.labels[index]
        second = self.labels[(index + 1) % size]
        candidates = self._edges_between(first, second, piece.edges)
        if size == _DOUBLE_LINK_SIZE:
            matches = tuple(
                edge
                for edge in candidates
                if edge.style in core_style_set and edge.multiplicity == 1
            )
            if len(matches) != 1:
                raise ValueError("piece core edge style mismatch")
            style = matches[0].style
        else:
            expected_style = self._edges_between(first, second, self.edges.edges)[0].style
            matches = tuple(
                edge
                for edge in candidates
                if edge.style is expected_style and edge.multiplicity == 1
            )
            if len(matches) != 1:
                raise ValueError("piece core edge style mismatch")
            style = None

        articulation = set(tree.articulation_points)
        if first in articulation or second in articulation:
            raise ValueError("core endpoints must be internal vertices")
        return style

    def _validate_gluing_overlaps(self, pieces: tuple[Bigraph, ...]) -> None:
        """Require consecutive pieces to meet exactly at their core vertices."""
        size = len(self.labels)
        for first_index in range(size):
            for second_index in range(first_index + 1, size):
                if size == _DOUBLE_LINK_SIZE:
                    expected_overlap = {self.labels[0], self.labels[1]}
                elif second_index == first_index + 1:
                    expected_overlap = {self.labels[second_index]}
                elif first_index == 0 and second_index == size - 1:
                    expected_overlap = {self.labels[0]}
                else:
                    expected_overlap = set()
                actual_overlap = set(pieces[first_index].labels) & set(
                    pieces[second_index].labels,
                )
                if actual_overlap != expected_overlap:
                    raise ValueError("unexpected piece overlap")

    def gluing(
        self,
        pieces: Iterable[Bigraph],
        labels: Iterable[Hashable] | None = None,
    ) -> "DCoreGluingCertificate":
        """Glue one A-type piece per core edge into a D_n bigraph.

        Parameters
        ----------
        pieces : Iterable[Bigraph]
            Connected positive definite A-type pieces in core-edge order.
        labels : Iterable[Hashable] | None
            Optional label order for the glued bigraph. Defaults to first
            appearance across the pieces.

        Returns
        -------
        DCoreGluingCertificate
            The verified construction certificate.

        Raises
        ------
        ValueError
            If the pieces do not satisfy the D-core gluing hypotheses.
        """
        try:
            piece_tuple = tuple(pieces)
        except TypeError as error:
            raise ValueError("pieces must be an iterable") from error
        size = len(self.labels)
        if len(piece_tuple) != size:
            raise ValueError("pieces must match the number of core edges")

        core_style_set = {edge.style for edge in self.edges.edges}
        piece_styles: list[EdgeStyle] = []
        for index, piece in enumerate(piece_tuple):
            style = self._validate_gluing_piece(index, piece, core_style_set)
            if style is not None:
                piece_styles.append(style)

        if size == _DOUBLE_LINK_SIZE and set(piece_styles) != core_style_set:
            raise ValueError("pieces must realize both styles of the double link")
        self._validate_gluing_overlaps(piece_tuple)

        gluing = (
            Bigraph.sum(*piece_tuple)
            if labels is None
            else Bigraph.sum(*piece_tuple, labels=labels)
        )
        if len(gluing.block_cut_forest().trees) != 1:
            raise ValueError("the D-core gluing must be connected")
        if gluing.dimension < _MIN_D_GLUING_SIZE:
            raise ValueError("a D-core gluing requires at least four vertices")
        if not gluing.matrix.inertia.is_positive_definite:
            raise ValueError("the D-core gluing must be positive definite")
        return DCoreGluingCertificate(
            core=self,
            pieces=piece_tuple,
            gluing=gluing,
            dynkin_type=DynkinType("D", gluing.dimension),
        )

    def cycle_condition(self) -> bool:
        """Return whether the core has an odd number of dotted edges."""
        dotted = sum(edge.style is EdgeStyle.DOTTED for edge in self.edges.edges)
        return dotted % 2 == 1


@dataclass(frozen=True)
class DCoreGluingCertificate:
    """A verifiable certificate for a D-core gluing.

    Parameters
    ----------
    core : DCore
        The D-core.
    pieces : tuple[Bigraph, ...]
        One A-type piece per core edge, in core order.
    gluing : Bigraph
        The claimed D_n gluing.
    dynkin_type : DynkinType
        The claimed ``D_n`` type.

    Raises
    ------
    ValueError
        If the certificate fields have inconsistent shapes or types.
    """

    core: DCore
    pieces: tuple[Bigraph, ...]
    gluing: Bigraph
    dynkin_type: DynkinType

    def __post_init__(self) -> None:
        """Validate the certificate's basic shape."""
        if not isinstance(self.core, DCore):
            raise ValueError("certificate core must be a DCore")
        try:
            pieces = tuple(self.pieces)
        except TypeError as error:
            raise ValueError("certificate pieces must be an iterable") from error
        if not all(isinstance(piece, Bigraph) for piece in pieces):
            raise ValueError("certificate pieces must be Bigraph values")
        if not isinstance(self.gluing, Bigraph):
            raise ValueError("certificate gluing must be a Bigraph")
        if not isinstance(self.dynkin_type, DynkinType):
            raise ValueError("certificate type must be a DynkinType")
        if len(pieces) != len(self.core.labels):
            raise ValueError("certificate must have one piece per core edge")
        if self.dynkin_type.family != "D" or self.dynkin_type.rank != self.gluing.dimension:
            raise ValueError("certificate type must be D_n for its gluing")
        object.__setattr__(self, "pieces", pieces)

    def verify(self) -> bool:
        """Return whether the certificate recomputes to the stored construction."""
        try:
            expected = self.core.gluing(self.pieces, labels=self.gluing.labels)
        except (AttributeError, TypeError, ValueError):
            return False
        return expected == self


@dataclass(frozen=True)
class ACyclizationCertificate:
    """A verifiable certificate for an A-cyclization.

    Parameters
    ----------
    source : Bigraph
        The connected positive definite A-type source.
    x : Hashable
        First identified label.
    y : Hashable
        Second identified label.
    path : tuple[Hashable, ...]
        The unique shortest path from ``x`` to ``y``.
    result : Bigraph
        The D_{n-1} bigraph obtained by identifying the endpoints.
    dynkin_type : DynkinType
        The claimed ``D_{n-1}`` type.

    Raises
    ------
    ValueError
        If the certificate fields have inconsistent shapes or types.
    """

    source: Bigraph
    x: Hashable
    y: Hashable
    path: tuple[Hashable, ...]
    result: Bigraph
    dynkin_type: DynkinType

    def __post_init__(self) -> None:
        """Validate the certificate's basic shape."""
        if not isinstance(self.source, Bigraph):
            raise ValueError("certificate source must be a Bigraph")
        if not isinstance(self.result, Bigraph):
            raise ValueError("certificate result must be a Bigraph")
        if not isinstance(self.dynkin_type, DynkinType):
            raise ValueError("certificate type must be a DynkinType")
        if self.x == self.y:
            raise ValueError("cyclization endpoints must be distinct")
        try:
            path = tuple(self.path)
        except TypeError as error:
            raise ValueError("certificate path must be an iterable") from error
        if not path:
            raise ValueError("certificate path must not be empty")
        if path[0] != self.x or path[-1] != self.y:
            raise ValueError("certificate path endpoints must match the identified labels")
        if any(label not in self.source.labels for label in path):
            raise ValueError("certificate path labels must belong to the source")
        if len(path) - 1 < _MIN_PATH_LENGTH:
            raise ValueError("certificate path must have length at least two")
        if self.source.dimension < _MIN_A_CYCLIZATION_SIZE:
            raise ValueError("certificate source must have at least five vertices")
        if self.result.dimension != self.source.dimension - 1:
            raise ValueError("certificate result must have one fewer vertex than the source")
        if self.dynkin_type.family != "D" or self.dynkin_type.rank != self.result.dimension:
            raise ValueError("certificate type must be D_(n-1) for its result")
        object.__setattr__(self, "path", path)

    def verify(self) -> bool:
        """Return whether the certificate recomputes to the stored construction."""
        try:
            expected = self.source.a_cyclization(self.x, self.y)
        except (AttributeError, TypeError, ValueError):
            return False
        return expected == self


def build_a_cyclization(
    bigraph: Bigraph,
    source: Hashable,
    target: Hashable,
) -> ACyclizationCertificate:
    """Build and validate an A-cyclization certificate.

    Parameters
    ----------
    bigraph : Bigraph
        Connected positive definite A-type source.
    source : Hashable
        First internal label.
    target : Hashable
        Second internal label.

    Returns
    -------
    ACyclizationCertificate
        The validated construction certificate.

    Raises
    ------
    ValueError
        If the source, endpoints, path, or result fail the A-cyclization
        hypotheses.
    """
    source_index = bigraph.index(source)
    target_index = bigraph.index(target)
    first = bigraph.labels[source_index]
    second = bigraph.labels[target_index]
    if first == second:
        raise ValueError("identified labels must be distinct")

    forest = bigraph.block_cut_forest()
    if len(forest.trees) != 1:
        raise ValueError("A-cyclization requires a connected source")
    if bigraph.dimension < _MIN_A_CYCLIZATION_SIZE:
        raise ValueError("A-cyclization requires at least five vertices")
    if bigraph.a_type_certificate() is None:
        raise ValueError("A-cyclization requires an A-type source")
    if not bigraph.matrix.inertia.is_positive_definite:
        raise ValueError("A-cyclization requires a positive definite source")

    tree = forest.trees[0]
    if first in tree.articulation_points or second in tree.articulation_points:
        raise ValueError("cyclization endpoints must be internal vertices")

    paths = list(nx.all_shortest_paths(bigraph.to_networkx(), first, second))
    if len(paths) != 1:
        raise ValueError("A-cyclization requires a unique shortest path")
    path = tuple(paths[0])
    if len(path) - 1 < _MIN_PATH_LENGTH:
        raise ValueError("A-cyclization path must have length at least two")

    order = {label: index for index, label in enumerate(bigraph.labels)}
    dotted = sum(
        bigraph.matrix.entries[order[path[index]]][order[path[index + 1]]] > 0
        for index in range(len(path) - 1)
    )
    if dotted % 2 == 0:
        raise ValueError("A-cyclization path requires an odd number of dotted edges")

    result = bigraph.identify(first, second)
    if len(result.block_cut_forest().trees) != 1:
        raise ValueError("A-cyclization result must be connected")
    if result.dimension < _MIN_D_GLUING_SIZE:
        raise ValueError("A-cyclization result requires at least four vertices")
    if not result.matrix.inertia.is_positive_definite:
        raise ValueError("A-cyclization result must be positive definite")

    return ACyclizationCertificate(
        source=bigraph,
        x=first,
        y=second,
        path=path,
        result=result,
        dynkin_type=DynkinType("D", result.dimension),
    )
