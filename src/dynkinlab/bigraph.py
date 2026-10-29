# SPDX-FileCopyrightText: 2026-present Mario Abarca <knkillname@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Label-generic mixed bigraphs over quasi-Cartan matrices."""

import html
from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from enum import StrEnum

import networkx as nx

import dynkinlab
from dynkinlab import display
from dynkinlab.display import DisplayOptions
from dynkinlab.quasi_cartan import QuasiCartanMatrix


class EdgeStyle(StrEnum):
    """Visual style of a bigraph edge or arrow."""

    SOLID = "solid"
    DOTTED = "dotted"


@dataclass(frozen=True)
class BigraphEdge:
    """An immutable mixed-bigraph edge or arrow.

    Parameters
    ----------
    source : Hashable
        Source vertex label.
    target : Hashable
        Target vertex label.
    directed : bool
        Whether the edge is an arrow.
    style : EdgeStyle
        Solid or dotted line style.
    multiplicity : int
        Positive number of parallel edges.

    Raises
    ------
    ValueError
        If the endpoints coincide, a label is unhashable, the style is
        unknown, or the multiplicity is not a positive integer.
    """

    source: Hashable
    target: Hashable
    directed: bool
    style: EdgeStyle
    multiplicity: int

    def __post_init__(self) -> None:
        """Validate the edge value."""
        if self.source == self.target:
            raise ValueError("edge endpoints must be distinct")
        for name, label in (("source", self.source), ("target", self.target)):
            try:
                hash(label)
            except TypeError as error:
                raise ValueError(f"{name} must be hashable") from error
        if not isinstance(self.style, EdgeStyle):
            raise ValueError("style must be an EdgeStyle")
        if isinstance(self.multiplicity, bool) or not isinstance(self.multiplicity, int):
            raise ValueError("multiplicity must be an integer")
        if self.multiplicity <= 0:
            raise ValueError("multiplicity must be positive")

    @property
    def weight(self) -> int:
        """Return the signed edge weight."""
        if self.style is EdgeStyle.SOLID:
            return -self.multiplicity
        return self.multiplicity


@dataclass(frozen=True, init=False)
class Bigraph:
    """A label-generic view over a quasi-Cartan matrix.

    Parameters
    ----------
    matrix : QuasiCartanMatrix
        The algebraic source of truth.
    labels : Iterable[Hashable] | None
        Unique vertex labels in matrix-index order. Defaults to ``0..n-1``.

    Raises
    ------
    ValueError
        If the labels are duplicated, unhashable, or do not match the matrix
        dimension.
    """

    matrix: QuasiCartanMatrix
    labels: tuple[Hashable, ...]

    def __init__(
        self,
        matrix: QuasiCartanMatrix,
        labels: Iterable[Hashable] | None = None,
    ) -> None:
        if not isinstance(matrix, QuasiCartanMatrix):
            raise ValueError("matrix must be a QuasiCartanMatrix")
        if labels is None:
            normalized: tuple[Hashable, ...] = tuple(range(matrix.dimension))
        else:
            try:
                normalized = tuple(labels)
            except TypeError as error:
                raise ValueError("labels must be an iterable") from error
        if len(normalized) != matrix.dimension:
            raise ValueError(
                f"labels must have {matrix.dimension} entries, got {len(normalized)}",
            )
        for label in normalized:
            try:
                hash(label)
            except TypeError as error:
                raise ValueError("labels must be hashable") from error
        if len(set(normalized)) != len(normalized):
            raise ValueError("labels must be unique")
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "labels", normalized)

    @property
    def dimension(self) -> int:
        """Return the number of vertices."""
        return self.matrix.dimension

    @property
    def edges(self) -> tuple[BigraphEdge, ...]:
        """Return the derived edges and arrows in matrix-index order."""
        entries = self.matrix.entries
        edges: list[BigraphEdge] = []
        for row in range(self.dimension):
            for column in range(row + 1, self.dimension):
                forward = entries[row][column]
                backward = entries[column][row]
                if forward == 0 and backward == 0:
                    continue
                if forward == backward:
                    style = EdgeStyle.DOTTED if forward > 0 else EdgeStyle.SOLID
                    edges.append(
                        BigraphEdge(
                            self.labels[row],
                            self.labels[column],
                            False,
                            style,
                            abs(forward),
                        ),
                    )
                    continue
                if forward == 0 or backward == 0:
                    raise ValueError("matrix is not representable as a bigraph")
                if abs(forward) < abs(backward):
                    source, target = row, column
                    weight = forward
                    multiplicity = abs(backward)
                elif abs(forward) > abs(backward):
                    source, target = column, row
                    weight = backward
                    multiplicity = abs(forward)
                else:
                    raise ValueError("matrix is not representable as a bigraph")
                style = EdgeStyle.DOTTED if weight > 0 else EdgeStyle.SOLID
                edges.append(
                    BigraphEdge(
                        self.labels[source],
                        self.labels[target],
                        True,
                        style,
                        multiplicity,
                    ),
                )
        return tuple(edges)

    def to_networkx(self) -> "nx.Graph[Hashable]":
        """Return an undirected NetworkX view of the bigraph.

        Every vertex label becomes a node, including labels with no incident
        edges. Each derived edge or arrow becomes one undirected graph edge
        carrying the following attributes:

        - ``style``: ``"solid"`` or ``"dotted"``
        - ``directed``: whether the original edge is an arrow
        - ``multiplicity``: positive number of parallel edges
        - ``weight``: signed multiplicity

        Returns
        -------
        networkx.Graph
            Graph whose nodes are the bigraph labels and whose edges preserve
            the derived bigraph structure.
        """
        graph = nx.Graph()
        graph.add_nodes_from(self.labels)
        for edge in self.edges:
            graph.add_edge(
                edge.source,
                edge.target,
                style=edge.style.value,
                directed=edge.directed,
                multiplicity=edge.multiplicity,
                weight=edge.weight,
            )
        return graph

    def block_cut_forest(self) -> "dynkinlab.structure.BlockCutForest":
        """Return the block-cut forest of the underlying graph.

        Returns
        -------
        BlockCutForest
            One block-cut tree per connected component, with blocks and
            articulation points in bigraph-index order.
        """
        from dynkinlab.structure import build_block_cut_forest

        return build_block_cut_forest(self)

    def is_a_block(
        self,
        labels: Iterable[Hashable],
    ) -> "dynkinlab.structure.ABlockPartition | None":
        """Return an A-block partition for an induced vertex set.

        Parameters
        ----------
        labels : Iterable[Hashable]
            Candidate block labels.

        Returns
        -------
        ABlockPartition | None
            A verified partition when the induced set is an A-block, otherwise
            ``None``.

        Raises
        ------
        ValueError
            If the labels are empty, duplicated, or unknown.
        """
        from dynkinlab.structure import a_block_partition

        return a_block_partition(self, labels)

    def a_type_certificate(self) -> "dynkinlab.structure.ATypeCertificate | None":
        """Return an A-type certificate when every component is A-type.

        Returns
        -------
        ATypeCertificate | None
            A verifiable certificate, or ``None`` when at least one component
            fails the A-type structural conditions.
        """
        from dynkinlab.structure import build_a_type_certificate

        return build_a_type_certificate(self)

    def is_type_a(self) -> bool:
        """Return whether every connected component is A-type."""
        return self.a_type_certificate() is not None

    @staticmethod
    def _sum_labels(
        pieces: tuple["Bigraph", ...],
        labels: Iterable[Hashable] | None,
    ) -> tuple[Hashable, ...]:
        """Return the validated label order for a sum."""
        union: list[Hashable] = []
        seen: set[Hashable] = set()
        for piece in pieces:
            if not isinstance(piece, Bigraph):
                raise ValueError("pieces must be Bigraph values")
            for label in piece.labels:
                if label not in seen:
                    union.append(label)
                    seen.add(label)
        if labels is None:
            return tuple(union)
        try:
            ordered = tuple(labels)
        except TypeError as error:
            raise ValueError("labels must be an iterable") from error
        try:
            provided = set(ordered)
        except TypeError as error:
            raise ValueError("labels must be hashable") from error
        if len(provided) != len(ordered):
            raise ValueError("labels must be unique")
        if provided != set(union):
            raise ValueError("labels must cover exactly the union of piece labels")
        return ordered

    @classmethod
    def sum(
        cls,
        *pieces: "Bigraph",
        labels: Iterable[Hashable] | None = None,
    ) -> "Bigraph":
        """Sum symmetric bigraphs by identifying shared labels.

        Off-diagonal weights add and cancel, while every diagonal entry stays
        at 2. The result uses first-appearance label order unless an explicit
        order is supplied.

        Parameters
        ----------
        *pieces : Bigraph
            Symmetric bigraphs to combine.
        labels : Iterable[Hashable] | None
            Optional label order covering exactly the union of piece labels.

        Returns
        -------
        Bigraph
            The simplified sum.

        Raises
        ------
        ValueError
            If no pieces are supplied, a piece is not a `Bigraph`, a piece
            contains an arrow, or an explicit label order is invalid.
        """
        if not pieces:
            raise ValueError("at least one bigraph is required")
        ordered = cls._sum_labels(pieces, labels)
        order = {label: index for index, label in enumerate(ordered)}
        entries = [
            [2 if row == column else 0 for column in range(len(ordered))]
            for row in range(len(ordered))
        ]
        for piece in pieces:
            for edge in piece.edges:
                if edge.directed:
                    raise ValueError("sum requires symmetric bigraphs")
                source = order[edge.source]
                target = order[edge.target]
                entries[source][target] += edge.weight
                entries[target][source] += edge.weight
        return cls(QuasiCartanMatrix(entries), ordered)

    def identify(self, source: Hashable, target: Hashable) -> "Bigraph":
        """Merge ``target`` into ``source`` and simplify the result.

        Parameters
        ----------
        source : Hashable
            Label retained in the result.
        target : Hashable
            Label removed from the result.

        Returns
        -------
        Bigraph
            The merged bigraph with opposite parallel edges cancelled.

        Raises
        ------
        ValueError
            If the labels are equal, either label is absent, or the bigraph
            contains an arrow.
        """
        from dynkinlab.edge_multiset import EdgeMultiset

        return EdgeMultiset.from_bigraph(self).identify(source, target).simplify()

    def a_cyclization(
        self,
        source: Hashable,
        target: Hashable,
    ) -> "dynkinlab.structure_d.ACyclizationCertificate":
        """Identify two internal vertices of a connected A-type source.

        Parameters
        ----------
        source : Hashable
            First internal label.
        target : Hashable
            Second internal label.

        Returns
        -------
        ACyclizationCertificate
            A verifiable certificate for the resulting D_{n-1} bigraph.

        Raises
        ------
        ValueError
            If the source, endpoints, shortest path, or result fail the
            A-cyclization hypotheses.
        """
        from dynkinlab.structure_d import build_a_cyclization

        return build_a_cyclization(self, source, target)

    @staticmethod
    def _effective_options(options: DisplayOptions | None) -> DisplayOptions:
        """Return the explicit options or the canonical display options."""
        return display.options if options is None else options

    @staticmethod
    def _markdown_label(label: Hashable) -> str:
        """Escape a label for a Markdown table cell."""
        return str(label).replace("|", r"\|")

    def to_markdown(self, options: DisplayOptions | None = None) -> str:
        """Render the bigraph as a Markdown edge/arrow summary.

        Parameters
        ----------
        options : DisplayOptions | None
            Per-call display options. The canonical options are used when omitted.

        Returns
        -------
        str
            Markdown table listing labels, direction, style, and multiplicity.
        """
        settings = self._effective_options(options)
        show_multiplicity = settings.bigraph_show_multiplicity
        header = "| source | target | kind | style |"
        separator = "|---|---|---|---|"
        if show_multiplicity:
            header += " multiplicity |"
            separator += "---:|"
        lines = ["**Bigraph**", "", header, separator]
        edges = self.edges
        if not edges:
            if show_multiplicity:
                lines.append("| - | - | none | - | - |")
            else:
                lines.append("| - | - | none | - |")
            return "\n".join(lines)

        for edge in edges:
            kind = "arrow" if edge.directed else "edge"
            row = (
                f"| {self._markdown_label(edge.source)} "
                f"| {self._markdown_label(edge.target)} "
                f"| {kind} | {edge.style.value} |"
            )
            if show_multiplicity:
                row += f" {edge.multiplicity} |"
            lines.append(row)
        return "\n".join(lines)

    def to_html(self, options: DisplayOptions | None = None) -> str:
        """Render the bigraph as a self-contained HTML/SVG block.

        Parameters
        ----------
        options : DisplayOptions | None
            Per-call display options. The canonical options are used when omitted.

        Returns
        -------
        str
            HTML block containing an inline SVG rendering.
        """
        from dynkinlab.bigraph_display import BigraphRenderer

        settings = self._effective_options(options)
        svg = BigraphRenderer.render(self, settings)
        summary = ", ".join(
            f"{edge.source} {'->' if edge.directed else '--'} {edge.target} "
            f"({edge.style.value} {edge.multiplicity})"
            for edge in self.edges
        )
        return (
            f'<div class="dynkinlab-bigraph" '
            f'data-edges="{html.escape(summary, quote=True)}">{svg}</div>'
        )

    def _repr_html_(self) -> str:
        """Return the canonical HTML representation for notebook display."""
        return self.to_html()

    def flation(self, source: Hashable, target: Hashable) -> "Bigraph":
        """Apply a flation and return a new bigraph with the same labels.

        Parameters
        ----------
        source : Hashable
            Source vertex label.
        target : Hashable
            Target vertex label. Equal labels perform a sign inversion.

        Returns
        -------
        Bigraph
            The flated bigraph.

        Raises
        ------
        ValueError
            If a label is unknown.
        """
        source_index = self.index(source)
        target_index = self.index(target)
        return Bigraph(
            self.matrix.flation(source_index, target_index),
            self.labels,
        )

    def classify(self) -> "dynkinlab.dynkin.DynkinClassification":
        """Classify the underlying matrix while preserving labels."""
        from dynkinlab.classifier import DynkinClassifier

        return DynkinClassifier.classify(self.matrix, self.labels)

    def index(self, label: Hashable) -> int:
        """Return the matrix index of a label.

        Parameters
        ----------
        label : Hashable
            Vertex label.

        Returns
        -------
        int
            Matrix index.

        Raises
        ------
        ValueError
            If the label is not present.
        """
        for index, candidate in enumerate(self.labels):
            if candidate == label:
                return index
        raise ValueError(f"unknown label: {label!r}")
