# SPDX-FileCopyrightText: 2026-present Mario Abarca <knkillname@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Finite Dynkin types and normal-form graph classification."""

from collections.abc import Hashable
from dataclasses import dataclass
from enum import StrEnum
from typing import override

import networkx as nx

from dynkinlab.bigraph import Bigraph, EdgeStyle
from dynkinlab.flation import FlationSequence
from dynkinlab.inertia import Inertia
from dynkinlab.quasi_cartan import QuasiCartanMatrix

_FAMILIES = frozenset({"A", "B", "C", "D", "E", "F", "G"})


class ClassificationStatus(StrEnum):
    """Outcome of a Dynkin classification attempt."""

    CLASSIFIED = "classified"
    NOT_ADMISSIBLE = "not_admissible"
    NOT_POSITIVE_DEFINITE = "not_positive_definite"


@dataclass(frozen=True)
class ComponentClassification:
    """The finite type of one connected input component.

    Parameters
    ----------
    labels : tuple[Hashable, ...]
        Original vertex labels of the component.
    dynkin_type : DynkinType
        The classified finite type.
    """

    labels: tuple[Hashable, ...]
    dynkin_type: "DynkinType"


@dataclass(frozen=True)
class DynkinClassification:
    """A rich, verifiable Dynkin classification result.

    Parameters
    ----------
    status : ClassificationStatus
        Overall outcome.
    reason : str
        Human-readable diagnostic for non-classified outcomes.
    inertia : Inertia
        Exact inertia of the input symmetrized matrix.
    components : tuple[ComponentClassification, ...]
        Classified input components.
    witness : FlationSequence
        First-class flation and sign-inversion sequence.
    normal_form : Bigraph | None
        Final Dynkin normal form when the input was classified.
    steps : int
        Number of witness steps.
    bound : int
        Applied symmetric inflation bound.
    input : Bigraph
        Retained labeled input used by ``verify``.
    """

    status: ClassificationStatus
    reason: str
    inertia: Inertia
    components: tuple[ComponentClassification, ...]
    witness: FlationSequence
    normal_form: Bigraph | None
    steps: int
    bound: int
    input: Bigraph

    def __post_init__(self) -> None:
        """Validate non-negative counters."""
        for name, value in (("steps", self.steps), ("bound", self.bound)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def is_classified(self) -> bool:
        """Return whether the input has a finite Dynkin type."""
        return self.status is ClassificationStatus.CLASSIFIED

    def verify(self) -> bool:
        """Return whether the witness replays to the reported normal form."""
        if self.normal_form is None:
            return False
        return self.witness.verify(self.input, self.normal_form)

    def to_markdown(self) -> str:
        """Render the classification result as Markdown."""
        lines = [
            "**Dynkin classification**",
            "",
            f"**Status:** `{self.status.name}`",
        ]
        if self.reason:
            lines.append(f"**Reason:** {self.reason}")
        lines.extend(("", "**Inertia**", "", self.inertia.to_markdown(), "", "**Components**", ""))
        if self.components:
            lines.extend(("| labels | Dynkin type |", "|---|---|"))
            for component in self.components:
                labels = ", ".join(str(label).replace("|", r"\|") for label in component.labels)
                lines.append(f"| {labels} | `{component.dynkin_type.name}` |")
        else:
            lines.append("None.")
        lines.extend(("", f"**Steps:** {self.steps}  **Bound:** {self.bound}"))
        if self.normal_form is not None and self.components:
            normal = ", ".join(str(component.dynkin_type) for component in self.components)
            lines.append(f"**Normal form:** {normal}")
        else:
            lines.append("**Normal form:** none")
        return "\n".join(lines)

    def _repr_markdown_(self) -> str:
        """Return the canonical Markdown representation for notebook display."""
        return self.to_markdown()


@dataclass(frozen=True)
class DynkinType:
    """A finite Cartan-Killing type.

    Parameters
    ----------
    family : str
        One of ``A``, ``B``, ``C``, ``D``, ``E``, ``F``, or ``G``.
    rank : int
        Rank of the type.

    Raises
    ------
    ValueError
        If the family or rank is not a valid finite Dynkin type.
    """

    family: str
    rank: int

    def __post_init__(self) -> None:
        """Validate the finite Dynkin type."""
        if self.family not in _FAMILIES:
            raise ValueError(f"unknown Dynkin family: {self.family}")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int):
            raise ValueError("rank must be an integer")
        valid = {
            "A": self.rank >= 1,
            "B": self.rank >= 2,
            "C": self.rank >= 3,
            "D": self.rank >= 4,
            "E": self.rank in {6, 7, 8},
            "F": self.rank == 4,
            "G": self.rank == 2,
        }
        if not valid[self.family]:
            raise ValueError(f"invalid rank {self.rank} for Dynkin family {self.family}")

    @property
    def name(self) -> str:
        """Return the canonical name, such as ``A_3``."""
        return f"{self.family}_{self.rank}"

    @override
    def __str__(self) -> str:
        """Return the canonical name."""
        return self.name

    def to_markdown(self) -> str:
        """Render the Dynkin type as Markdown."""
        return f"**Dynkin type:** `{self.name}`"

    def _repr_markdown_(self) -> str:
        """Return the canonical Markdown representation for notebook display."""
        return self.to_markdown()

    def cartan_matrix(self) -> QuasiCartanMatrix:
        """Return a canonical Cartan matrix for this type.

        Returns
        -------
        QuasiCartanMatrix
            A positive definite matrix whose mixed bigraph is the canonical
            Dynkin diagram.
        """
        size = self.rank
        rows = [[2 if row == column else 0 for column in range(size)] for row in range(size)]

        if self.family in {"A", "D", "E"}:
            edges = self._symmetric_edges()
        elif self.family == "B":
            edges = [(index, index + 1) for index in range(size - 1)]
            for row, column in edges:
                rows[row][column] = -1
                rows[column][row] = -1
            rows[0][1] = -2
            rows[1][0] = -1
            return QuasiCartanMatrix(rows)
        elif self.family == "C":
            edges = [(index, index + 1) for index in range(size - 1)]
            for row, column in edges:
                rows[row][column] = -1
                rows[column][row] = -1
            rows[0][1] = -1
            rows[1][0] = -2
            return QuasiCartanMatrix(rows)
        elif self.family == "F":
            edges = [(index, index + 1) for index in range(size - 1)]
            for row, column in edges:
                rows[row][column] = -1
                rows[column][row] = -1
            rows[1][2] = -1
            rows[2][1] = -2
            return QuasiCartanMatrix(rows)
        else:
            rows[0][1] = -1
            rows[1][0] = -3
            return QuasiCartanMatrix(rows)

        for row, column in edges:
            rows[row][column] = -1
            rows[column][row] = -1
        return QuasiCartanMatrix(rows)

    def bigraph(self) -> Bigraph:
        """Return the canonical mixed bigraph for this type."""
        return Bigraph(self.cartan_matrix())

    def _symmetric_edges(self) -> tuple[tuple[int, int], ...]:
        """Return the canonical symmetric Dynkin edges."""
        size = self.rank
        if self.family == "A":
            return tuple((index, index + 1) for index in range(size - 1))
        if self.family == "D":
            edges = [(0, 1), (0, 2), (0, 3)]
            edges.extend((index, index + 1) for index in range(3, size - 1))
            return tuple(edges)
        if self.family == "E":
            arms = {
                6: (1, 2, 2),
                7: (1, 2, 3),
                8: (1, 2, 4),
            }[size]
            edges = []
            next_index = 1
            for length in arms:
                previous = 0
                for _ in range(length):
                    edges.append((previous, next_index))
                    previous = next_index
                    next_index += 1
            return tuple(edges)
        raise ValueError(f"no symmetric edges for {self.name}")

    @classmethod
    def from_bigraph(cls, bigraph: Bigraph) -> "DynkinType | None":
        """Classify a connected solid mixed bigraph.

        Parameters
        ----------
        bigraph : Bigraph
            A connected bigraph whose edges are solid.

        Returns
        -------
        DynkinType | None
            The finite Dynkin type, or ``None`` when the graph is not a
            Dynkin diagram.
        """
        size = bigraph.dimension
        if size == 1:
            return cls("A", 1)

        edges = bigraph.edges
        if any(edge.style is not EdgeStyle.SOLID for edge in edges):
            return None
        if any(not edge.directed and edge.multiplicity != 1 for edge in edges):
            return None

        graph = nx.Graph()
        graph.add_nodes_from(range(size))
        arrows: list[tuple[int, int, int]] = []
        for edge in edges:
            source = bigraph.index(edge.source)
            target = bigraph.index(edge.target)
            graph.add_edge(source, target)
            if edge.directed:
                if edge.multiplicity not in {2, 3}:
                    return None
                arrows.append((source, target, edge.multiplicity))

        if len(arrows) > 1:
            return None
        if not nx.is_tree(graph):
            return None

        degrees = [graph.degree(vertex) for vertex in range(size)]
        if max(degrees) > 3:
            return None
        if sum(degree == 3 for degree in degrees) > 1:
            return None
        leaves = {index for index, degree in enumerate(degrees) if degree == 1}
        branch = next((index for index, degree in enumerate(degrees) if degree == 3), None)

        if branch is not None:
            if arrows:
                return None
            distances = nx.single_source_shortest_path_length(graph, branch)
            at_two = sum(distance == 2 for distance in distances.values())
            at_three = sum(distance == 3 for distance in distances.values())
            if at_two <= 1:
                return cls("D", size)
            if 6 <= size <= 8 and at_two == 2 and at_three <= 1:
                return cls("E", size)
            return None

        if len(leaves) != 2 or max(degrees) > 2:
            return None
        if not arrows:
            return cls("A", size)

        source, target, weight = arrows[0]
        if size == 2 and weight == 3:
            return cls("G", 2)
        if weight != 2:
            return None
        if target in leaves:
            return cls("B", size)
        if source in leaves:
            return cls("C", size) if size >= 3 else cls("B", 2)
        if size == 4:
            return cls("F", 4)
        return None

    @classmethod
    def from_matrix(cls, matrix: QuasiCartanMatrix) -> "DynkinType | None":
        """Classify the mixed bigraph of a quasi-Cartan matrix."""
        return cls.from_bigraph(Bigraph(matrix))
