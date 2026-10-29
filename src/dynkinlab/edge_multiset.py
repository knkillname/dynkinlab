# SPDX-FileCopyrightText: 2026-present Mario Abarca <knkillname@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Labeled multisets of undirected bigraph edges."""

from collections.abc import Hashable
from dataclasses import dataclass

from dynkinlab.bigraph import Bigraph, BigraphEdge, EdgeStyle
from dynkinlab.quasi_cartan import QuasiCartanMatrix


@dataclass(frozen=True)
class EdgeMultiset:
    """A labeled multiset of undirected solid and dotted edges.

    Same-style parallel edges are normalized into one edge with a larger
    multiplicity. Opposite-style parallel edges remain separate until
    :meth:`simplify` is called.

    Parameters
    ----------
    labels : Iterable[Hashable]
        Unique vertex labels in deterministic order.
    edges : Iterable[BigraphEdge]
        Undirected edges over those labels. Directed edges are rejected.

    Raises
    ------
    ValueError
        If labels are empty, duplicated, unhashable, or an edge is directed,
        malformed, or uses an unknown label.
    """

    labels: tuple[Hashable, ...]
    edges: tuple[BigraphEdge, ...]

    def __post_init__(self) -> None:
        """Validate and normalize the labels and edges."""
        try:
            labels = tuple(self.labels)
        except TypeError as error:
            raise ValueError("labels must be an iterable") from error
        if not labels:
            raise ValueError("labels must not be empty")
        for label in labels:
            try:
                hash(label)
            except TypeError as error:
                raise ValueError("labels must be hashable") from error
        if len(set(labels)) != len(labels):
            raise ValueError("labels must be unique")

        try:
            edges = tuple(self.edges)
        except TypeError as error:
            raise ValueError("edges must be an iterable") from error
        order = {label: index for index, label in enumerate(labels)}
        counts: dict[tuple[int, int, EdgeStyle], int] = {}
        for edge in edges:
            if not isinstance(edge, BigraphEdge):
                raise ValueError("edges must contain BigraphEdge values")
            if edge.directed:
                raise ValueError("edge multiset requires undirected edges")
            try:
                source = order[edge.source]
                target = order[edge.target]
            except KeyError as error:
                raise ValueError(f"unknown label: {error.args[0]!r}") from error
            if source == target:
                raise ValueError("edge endpoints must be distinct")
            key = (min(source, target), max(source, target), edge.style)
            counts[key] = counts.get(key, 0) + edge.multiplicity

        normalized = tuple(
            BigraphEdge(labels[source], labels[target], False, style, multiplicity)
            for (source, target, style), multiplicity in sorted(
                counts.items(),
                key=lambda item: (
                    item[0][0],
                    item[0][1],
                    item[0][2] is EdgeStyle.DOTTED,
                ),
            )
        )
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "edges", normalized)

    @classmethod
    def from_bigraph(cls, bigraph: Bigraph) -> "EdgeMultiset":
        """Return the edge multiset of a symmetric bigraph.

        Parameters
        ----------
        bigraph : Bigraph
            Source bigraph.

        Returns
        -------
        EdgeMultiset
            The bigraph's labels and undirected edges.

        Raises
        ------
        ValueError
            If the bigraph contains a directed edge.
        """
        return cls(bigraph.labels, bigraph.edges)

    def __add__(self, other: "EdgeMultiset") -> "EdgeMultiset":
        """Return the union of two multisets with shared labels identified."""
        labels = list(self.labels)
        seen = set(labels)
        for label in other.labels:
            if label not in seen:
                labels.append(label)
                seen.add(label)
        return EdgeMultiset(tuple(labels), self.edges + other.edges)

    def identify(self, source: Hashable, target: Hashable) -> "EdgeMultiset":
        """Return the multiset after merging ``target`` into ``source``.

        The edge between the two labels, if any, becomes a loop and is dropped.
        Other incident edges are added and normalized, so opposite parallel
        edges remain present until simplification.

        Parameters
        ----------
        source : Hashable
            Label retained in the result.
        target : Hashable
            Label removed from the result.

        Returns
        -------
        EdgeMultiset
            The merged multiset.

        Raises
        ------
        ValueError
            If the labels are equal or either label is absent.
        """
        if source == target:
            raise ValueError("identified labels must be distinct")
        if source not in self.labels:
            raise ValueError(f"unknown label: {source!r}")
        if target not in self.labels:
            raise ValueError(f"unknown label: {target!r}")
        labels = tuple(label for label in self.labels if label != target)
        edges: list[BigraphEdge] = []
        for edge in self.edges:
            first = source if edge.source == target else edge.source
            second = source if edge.target == target else edge.target
            if first == second:
                continue
            edges.append(BigraphEdge(first, second, False, edge.style, edge.multiplicity))
        return EdgeMultiset(labels, tuple(edges))

    def simplify(self) -> Bigraph:
        """Cancel opposite-style pairs and return the resulting bigraph.

        Returns
        -------
        Bigraph
            The simplified bigraph with the multiset's label order.
        """
        order = {label: index for index, label in enumerate(self.labels)}
        entries = [
            [2 if row == column else 0 for column in range(len(self.labels))]
            for row in range(len(self.labels))
        ]
        for edge in self.edges:
            source = order[edge.source]
            target = order[edge.target]
            entries[source][target] += edge.weight
            entries[target][source] += edge.weight
        return Bigraph(QuasiCartanMatrix(entries), self.labels)
