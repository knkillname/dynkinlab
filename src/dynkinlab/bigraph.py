# SPDX-FileCopyrightText: 2026-present Mario Abarca <knkillname@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Label-generic mixed bigraphs over quasi-Cartan matrices."""

import html
import re
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum

import networkx as nx
from matplotlib.axes import Axes

import dynkinlab
from dynkinlab import display
from dynkinlab.display import DisplayOptions
from dynkinlab.quasi_cartan import QuasiCartanMatrix

_PATH_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|--|\.\.|->|<-|\.>|<.")
_PATH_OPERATORS = frozenset({"--", "..", "->", "<-", ".>", "<."})
_PATH_STEP = 2


def _path_tokens(line: str, line_number: int) -> list[str]:
    """Tokenize one non-empty path line and reject malformed spacing."""
    tokens: list[str] = []
    position = 0
    for match in _PATH_TOKEN.finditer(line):
        if line[position : match.start()].strip(" \t,"):
            raise ValueError(f"line {line_number}: invalid token in path {line!r}")
        tokens.append(match.group())
        position = match.end()
    if line[position:].strip(" \t,"):
        raise ValueError(f"line {line_number}: invalid token in path {line!r}")
    if not tokens or len(tokens) % _PATH_STEP == 0:
        raise ValueError(f"line {line_number}: invalid path {line!r}")
    return tokens


def _path_values(
    operator: str,
    current: Hashable,
    following: Hashable,
    arrow_multiplicity: int,
) -> dict[tuple[Hashable, Hashable], int]:
    """Return the two matrix entries declared by one path operator."""
    if operator not in _PATH_OPERATORS:
        raise ValueError(f"unknown edge operator {operator!r}")
    if operator == "--":
        return {(current, following): -1, (following, current): -1}
    if operator == "..":
        return {(current, following): 1, (following, current): 1}
    if operator == "->":
        return {(current, following): -1, (following, current): -arrow_multiplicity}
    if operator == "<-":
        return {(current, following): -arrow_multiplicity, (following, current): -1}
    if operator == ".>":
        return {(current, following): 1, (following, current): arrow_multiplicity}
    return {(current, following): arrow_multiplicity, (following, current): 1}


def _pinned_labels(labels: Iterable[Hashable] | None) -> tuple[Hashable, ...] | None:
    """Validate an explicit path label order."""
    if labels is None:
        return None
    try:
        pinned = tuple(labels)
    except TypeError as error:
        raise ValueError("labels must be an iterable") from error
    for label in pinned:
        try:
            hash(label)
        except TypeError as error:
            raise ValueError("labels must be hashable") from error
    if len(set(pinned)) != len(pinned):
        raise ValueError("labels must be unique")
    return pinned


def _path_label(
    token: str,
    pinned: tuple[Hashable, ...] | None,
    order: list[Hashable],
    line_number: int,
) -> Hashable:
    """Convert a token to a label and register it in first-appearance order."""
    label: Hashable = int(token) if token.isdigit() else token
    if pinned is not None and label not in pinned:
        raise ValueError(
            f"line {line_number}: label {label!r} is outside the explicit label order",
        )
    if label not in order:
        order.append(label)
    return label


def _declare_pair(
    declared: dict[frozenset[Hashable], dict[tuple[Hashable, Hashable], int]],
    values: dict[tuple[Hashable, Hashable], int],
    current: Hashable,
    following: Hashable,
    line_number: int,
) -> None:
    """Store one pair declaration and reject an inconsistent repeat."""
    pair = frozenset((current, following))
    previous = declared.get(pair)
    if previous is not None and previous != values:
        raise ValueError(
            f"line {line_number}: conflicting declarations for {current!r} and {following!r}",
        )
    declared[pair] = values


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

    @classmethod
    def from_paths(
        cls,
        *paths: str,
        labels: Iterable[Hashable] | None = None,
        arrow_multiplicity: int = 2,
    ) -> "Bigraph":
        """Construct a labeled bigraph from DynkinLab path notation.

        Each path block is a sequence of lines; blocks are joined with
        newlines. Each line is a path of vertices joined by ``--``, ``..``,
        ``->``, ``<-``, ``.>``, or ``<.``. Whitespace and commas separate
        tokens, ``#`` starts a comment, and a line with one vertex declares an
        isolated label. Labels default to first appearance unless an explicit
        order is supplied.

        Parameters
        ----------
        *paths : str
            One or more path-notation blocks. Blocks are joined with newlines.
        labels : Iterable[Hashable] | None
            Optional explicit label order. Path labels outside this order are
            rejected. Labels are keyword-only.
        arrow_multiplicity : int
            Reverse magnitude assigned to every arrow operator. The default
            is 2. This value is keyword-only.

        Returns
        -------
        Bigraph
            The labeled bigraph built from the parsed matrix.

        Raises
        ------
        ValueError
            If a path argument is not a string, the text is malformed, a pair
            is declared inconsistently, a path label is outside the explicit
            order, or the multiplicity is not a positive integer.
        """
        if any(not isinstance(path, str) for path in paths):
            raise ValueError("path arguments must be strings; pass labels by keyword")
        if isinstance(arrow_multiplicity, bool) or not isinstance(arrow_multiplicity, int):
            raise ValueError("arrow_multiplicity must be a positive integer")
        if arrow_multiplicity <= 0:
            raise ValueError("arrow_multiplicity must be a positive integer")
        text = "\n".join(paths)
        pinned = _pinned_labels(labels)

        order: list[Hashable] = []
        declared: dict[frozenset[Hashable], dict[tuple[Hashable, Hashable], int]] = {}
        for line_number, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            tokens = _path_tokens(line, line_number)

            current = _path_label(tokens[0], pinned, order, line_number)
            for token_index in range(1, len(tokens), _PATH_STEP):
                operator = tokens[token_index]
                following = _path_label(tokens[token_index + 1], pinned, order, line_number)
                if current == following:
                    raise ValueError(f"line {line_number}: an edge requires distinct endpoints")
                try:
                    values = _path_values(
                        operator,
                        current,
                        following,
                        arrow_multiplicity,
                    )
                except ValueError as error:
                    raise ValueError(f"line {line_number}: {error}") from error
                _declare_pair(declared, values, current, following, line_number)
                current = following

        matrix_labels = tuple(order) if pinned is None else pinned
        if not matrix_labels:
            raise ValueError(
                "path text must declare at least one vertex when labels are not pinned",
            )

        index = {label: position for position, label in enumerate(matrix_labels)}
        size = len(matrix_labels)
        rows = [[2 if row == column else 0 for column in range(size)] for row in range(size)]
        for values in declared.values():
            for (source, target), value in values.items():
                rows[index[source]][index[target]] = value
        return cls(QuasiCartanMatrix(rows), matrix_labels)

    @property
    def edges(self) -> tuple[BigraphEdge, ...]:
        """Return the derived edges and arrows in matrix-index order.

        Raises
        ------
        ValueError
            If the underlying matrix has no faithful bigraph representation.
        """
        if not self.matrix.has_faithful_bigraph:
            raise ValueError("matrix has no faithful bigraph representation")
        entries = self.matrix.entries
        edges: list[BigraphEdge] = []
        for row in range(self.dimension):
            for column in range(row + 1, self.dimension):
                forward = entries[row][column]
                backward = entries[column][row]
                if forward == backward:
                    if forward == 0:
                        continue
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
                if abs(forward) < abs(backward):
                    source, target = row, column
                    weight = forward
                    multiplicity = abs(backward)
                else:
                    source, target = column, row
                    weight = backward
                    multiplicity = abs(forward)
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

    def layout(
        self,
        layout: str | None = None,
        *,
        seed: int | None = None,
        iterations: int | None = None,
        options: DisplayOptions | None = None,
    ) -> dict[Hashable, tuple[float, float]]:
        """Compute vertex coordinates with a NetworkX layout.

        Parameters
        ----------
        layout : str | None
            Layout policy. When omitted, the effective display options choose
            ``"auto"`` or the configured policy.
        seed : int | None
            Layout seed. When omitted, the effective display options supply it.
        iterations : int | None
            Spring iteration count. When omitted, the effective display options
            supply it.
        options : DisplayOptions | None
            Per-call display options. The canonical options are used when omitted.

        Returns
        -------
        dict[Hashable, tuple[float, float]]
            One coordinate pair for every vertex label.

        Raises
        ------
        ValueError
            If the layout policy is unknown, the matrix has no faithful
            bigraph, or the explicit planar policy receives a non-planar graph.
        """
        from dynkinlab.bigraph_display import compute_layout

        settings = self._effective_options(options)
        selected_layout = settings.bigraph_layout if layout is None else layout
        selected_seed = settings.bigraph_layout_seed if seed is None else seed
        selected_iterations = (
            settings.bigraph_layout_iterations if iterations is None else iterations
        )
        return compute_layout(
            self,
            selected_layout,
            selected_seed,
            selected_iterations,
        )

    def draw(
        self,
        *,
        ax: Axes | None = None,
        layout: str | None = None,
        seed: int | None = None,
        iterations: int | None = None,
        positions: Mapping[Hashable, tuple[float, float]] | None = None,
        options: DisplayOptions | None = None,
        title: str | None = None,
        highlight_edges: Iterable[frozenset[Hashable]] = (),
        highlight_nodes: Iterable[Hashable] = (),
        display_labels: Mapping[Hashable, str] | None = None,
    ) -> Axes:
        """Draw the bigraph on a matplotlib axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes | None
            Existing axes. A new figure is created when omitted.
        layout : str | None
            Layout policy. When omitted, the effective display options choose it.
        seed : int | None
            Layout seed. When omitted, the effective display options supply it.
        iterations : int | None
            Spring iteration count. When omitted, the effective display options
            supply it.
        positions : Mapping[Hashable, tuple[float, float]] | None
            Computed coordinates to reuse. When supplied, no layout is recomputed.
        options : DisplayOptions | None
            Per-call display options. The canonical options are used when omitted.
        title : str | None
            Optional axes title.
        highlight_edges : Iterable[frozenset[Hashable]]
            Undirected endpoint pairs to draw with the highlight style.
        highlight_nodes : Iterable[Hashable]
            Vertex labels to draw with the highlight style.
        display_labels : Mapping[Hashable, str] | None
            Optional text replacing the raw vertex labels.

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the drawing.

        Raises
        ------
        ValueError
            If the matrix has no faithful bigraph or a position is missing.
        """
        from dynkinlab.bigraph_display import BigraphRenderer

        settings = self._effective_options(options)
        return BigraphRenderer.draw(
            self,
            ax=ax,
            layout=layout,
            seed=seed,
            iterations=iterations,
            positions=positions,
            options=settings,
            title=title,
            highlight_edges=highlight_edges,
            highlight_nodes=highlight_nodes,
            display_labels=display_labels,
        )

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

    def to_matrix_markdown(self, options: DisplayOptions | None = None) -> str:
        """Render the underlying matrix with the bigraph labels as Markdown.

        Parameters
        ----------
        options : DisplayOptions | None
            Per-call display options. The canonical options are used when omitted.

        Returns
        -------
        str
            Markdown table whose row and column headers are the bigraph labels.
        """
        return self.matrix.to_markdown(options=options, labels=self.labels)

    def to_matrix_html(
        self,
        options: DisplayOptions | None = None,
        *,
        collapse: bool = False,
    ) -> str:
        """Render the underlying matrix with the bigraph labels as HTML.

        Parameters
        ----------
        options : DisplayOptions | None
            Per-call display options. The canonical options are used when omitted.
        collapse : bool
            Whether to wrap the labeled table in a collapsed HTML disclosure.

        Returns
        -------
        str
            HTML table whose row and column headers are the bigraph labels.
        """
        return self.matrix.to_html(
            options=options,
            labels=self.labels,
            collapse=collapse,
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
