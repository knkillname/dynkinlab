# SPDX-FileCopyrightText: 2026-present Mario Abarca <knkillname@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Label-generic mixed bigraphs over quasi-Cartan matrices."""

import html
from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from dynkinlab import display
from dynkinlab.display import DisplayOptions
from dynkinlab.quasi_cartan import QuasiCartanMatrix

if TYPE_CHECKING:
    from dynkinlab.dynkin import DynkinClassification


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

    def classify(self) -> "DynkinClassification":
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
