# SPDX-FileCopyrightText: 2026-present Mario Abarca <knkillname@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Canonical positive diagonal symmetrizers for quasi-Cartan matrices."""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from math import gcd, lcm
from typing import Self, cast, override

import networkx as nx

_EDGE_SIZE = 2
_MIN_CYCLE_SIZE = 3


class SymmetrizerObstruction(StrEnum):
    """The kind of obstruction preventing a positive diagonal symmetrizer."""

    ZERO_ENTRY = "zero_entry"
    OPPOSITE_SIGNS = "opposite_signs"
    INCONSISTENT_CYCLE = "inconsistent_cycle"


@dataclass(frozen=True)
class SymmetrizerCertificate:
    """A verifiable proof that no positive diagonal symmetrizer exists.

    Parameters
    ----------
    kind : SymmetrizerObstruction
        The obstruction kind.
    vertices : tuple[int, ...]
        The edge ``(i, j)`` for an edge obstruction, or the cycle vertices for
        an inconsistent cycle.
    """

    kind: SymmetrizerObstruction
    vertices: tuple[int, ...]

    def verify(self, entries: tuple[tuple[int, ...], ...]) -> bool:
        """Return whether this certificate is valid for the matrix.

        Parameters
        ----------
        entries : tuple[tuple[int, ...], ...]
            Square matrix rows.

        Returns
        -------
        bool
            True when the named obstruction holds.
        """
        size = len(entries)
        if size == 0 or any(len(row) != size for row in entries):
            return False
        if not self.vertices or any(not 0 <= index < size for index in self.vertices):
            return False

        if self.kind in (
            SymmetrizerObstruction.ZERO_ENTRY,
            SymmetrizerObstruction.OPPOSITE_SIGNS,
        ):
            return self._verify_edge(entries)
        if self.kind is SymmetrizerObstruction.INCONSISTENT_CYCLE:
            return self._verify_cycle(entries)
        return False

    def _verify_edge(self, entries: tuple[tuple[int, ...], ...]) -> bool:
        """Verify a zero-entry or opposite-sign edge certificate."""
        if len(self.vertices) != _EDGE_SIZE:
            return False
        first, second = self.vertices
        if self.kind is SymmetrizerObstruction.ZERO_ENTRY:
            return entries[first][second] == 0 and entries[second][first] != 0
        return entries[first][second] * entries[second][first] < 0

    def _verify_cycle(self, entries: tuple[tuple[int, ...], ...]) -> bool:
        """Verify an inconsistent-cycle certificate."""
        cycle = self.vertices
        if len(cycle) < _MIN_CYCLE_SIZE or len(set(cycle)) != len(cycle):
            return False
        product = Fraction(1)
        for first, second in zip(cycle, cycle[1:] + cycle[:1], strict=True):
            forward = entries[first][second]
            backward = entries[second][first]
            if forward == 0 or backward == 0 or forward * backward < 0:
                return False
            product *= Fraction(forward, backward)
        return product != 1

    def describe(self, entries: tuple[tuple[int, ...], ...]) -> str:
        """Return a concise human-readable proof.

        Parameters
        ----------
        entries : tuple[tuple[int, ...], ...]
            Square matrix rows.

        Returns
        -------
        str
            The obstruction explanation.

        Raises
        ------
        ValueError
            If the certificate does not verify against ``entries``.
        """
        if not self.verify(entries):
            raise ValueError("certificate does not verify against the supplied matrix")

        if self.kind is SymmetrizerObstruction.ZERO_ENTRY:
            first, second = self.vertices
            return (
                f"zero entry at edge ({first}, {second}): "
                f"a[{first}][{second}] = 0 but a[{second}][{first}] = "
                f"{entries[second][first]}"
            )

        if self.kind is SymmetrizerObstruction.OPPOSITE_SIGNS:
            first, second = self.vertices
            return (
                f"opposite signs at edge ({first}, {second}): "
                f"a[{first}][{second}] = {entries[first][second]} and "
                f"a[{second}][{first}] = {entries[second][first]}"
            )

        product = Fraction(1)
        for first, second in zip(
            self.vertices,
            self.vertices[1:] + self.vertices[:1],
            strict=True,
        ):
            product *= Fraction(entries[first][second], entries[second][first])
        cycle = " -> ".join(str(index) for index in self.vertices + self.vertices[:1])
        return f"inconsistent cycle {cycle}: ratio product {product} != 1"


class NotSymmetrizableError(ValueError):
    """Raised when no positive diagonal integer symmetrizer exists.

    Parameters
    ----------
    message : str
        Human-readable explanation.
    certificate : SymmetrizerCertificate
        Verifiable obstruction witness.
    """

    def __init__(self, message: str, certificate: SymmetrizerCertificate) -> None:
        super().__init__(message)
        self.certificate = certificate

    @override
    def __reduce__(self) -> tuple[type[Self], tuple[str, SymmetrizerCertificate]]:
        """Return the pickling recipe, including the certificate."""
        return type(self), (str(self), self.certificate)


class Symmetrizer:
    """Compute and normalize positive diagonal symmetrizers.

    A symmetrizer ``D`` for an integer matrix ``A`` is a positive diagonal
    integer matrix such that ``D A`` is symmetric. For a connected matrix it is
    unique up to a positive scalar; the canonical value is the primitive
    integer vector of minimal trace. Disconnected components are normalized
    independently.
    """

    @classmethod
    def canonical(cls, entries: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
        """Return the canonical minimal-trace symmetrizer of a matrix.

        Parameters
        ----------
        entries : tuple[tuple[int, ...], ...]
            Square matrix rows with diagonal entries equal to 2.

        Returns
        -------
        tuple[int, ...]
            Canonical positive diagonal symmetrizer.

        Raises
        ------
        NotSymmetrizableError
            If no positive diagonal integer symmetrizer exists.
        """
        size = len(entries)
        if cls._is_symmetric(entries):
            return (1,) * size

        result = [0] * size
        for component in cls.components(entries):
            values = cls._component_values(entries, component)
            for index in component:
                result[index] = values[index]
        return tuple(result)

    @classmethod
    def normalize(
        cls,
        entries: tuple[tuple[int, ...], ...],
        symmetrizer: Iterable[int],
    ) -> tuple[int, ...]:
        """Validate and canonically normalize a supplied symmetrizer.

        Parameters
        ----------
        entries : tuple[tuple[int, ...], ...]
            Square matrix rows with diagonal entries equal to 2.
        symmetrizer : Iterable[int]
            Proposed positive diagonal symmetrizer.

        Returns
        -------
        tuple[int, ...]
            Canonical normalized symmetrizer.

        Raises
        ------
        ValueError
            If the supplied values are not a valid positive symmetrizer.
        """
        try:
            values = tuple(symmetrizer)
        except TypeError as error:
            raise ValueError("symmetrizer must be an iterable of integers") from error

        size = len(entries)
        if len(values) != size:
            raise ValueError(f"symmetrizer must have {size} entries, got {len(values)}")
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("symmetrizer entries must be integers")
            if value <= 0:
                raise ValueError("symmetrizer entries must be positive")

        for row in range(size):
            for column in range(row + 1, size):
                if values[row] * entries[row][column] != values[column] * entries[column][row]:
                    raise ValueError("matrix and symmetrizer are not compatible")

        result = list(values)
        for component in cls.components(entries):
            divisor = 0
            for index in component:
                divisor = gcd(divisor, result[index])
            for index in component:
                result[index] //= divisor
        return tuple(result)

    @staticmethod
    def components(entries: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
        """Return connected components of the underlying undirected graph.

        Parameters
        ----------
        entries : tuple[tuple[int, ...], ...]
            Square matrix rows.

        Returns
        -------
        tuple[tuple[int, ...], ...]
            Components as sorted vertex-index tuples, ordered by their
            smallest vertex index.
        """
        size = len(entries)
        graph = nx.Graph()
        graph.add_nodes_from(range(size))
        graph.add_edges_from(
            (row, column)
            for row in range(size)
            for column in range(row + 1, size)
            if entries[row][column] or entries[column][row]
        )
        raw_components = cast(Iterable[set[int]], nx.connected_components(graph))
        components = sorted(
            (tuple(sorted(component)) for component in raw_components),
            key=lambda component: component[0],
        )
        return tuple(components)

    @classmethod
    def _component_values(
        cls,
        entries: tuple[tuple[int, ...], ...],
        component: tuple[int, ...],
    ) -> dict[int, int]:
        """Return primitive integer ratios for one connected component.

        Parameters
        ----------
        entries : tuple[tuple[int, ...], ...]
            Square matrix rows.
        component : tuple[int, ...]
            Sorted vertex indices of one connected component.

        Returns
        -------
        dict[int, int]
            Primitive positive integer values keyed by vertex index.

        Raises
        ------
        NotSymmetrizableError
            If the component has no positive diagonal integer symmetrizer.
        """
        # Correctness: a zero side forces the opposite d to 0; opposite signs cannot
        # balance positive d; otherwise every edge fixes d_j/d_i = a_ij/a_ji > 0.
        # A connected graph is consistent iff every fundamental cycle product is 1;
        # clearing denominators and dividing by gcd yields the unique primitive solution.
        root = component[0]
        values: dict[int, Fraction] = {root: Fraction(1)}
        parent: dict[int, int] = {root: root}
        depth: dict[int, int] = {root: 0}
        queue = [root]

        while queue:
            row = queue.pop()
            for column in component:
                if column == row:
                    continue
                forward = entries[row][column]
                backward = entries[column][row]
                if forward == 0 and backward == 0:
                    continue
                if forward == 0 or backward == 0:
                    # 0 = d_j*a_ji forces d_j = 0.
                    first, second = (row, column) if forward == 0 else (column, row)
                    certificate = SymmetrizerCertificate(
                        SymmetrizerObstruction.ZERO_ENTRY,
                        (first, second),
                    )
                    raise cls._failure(certificate, entries)
                if forward * backward < 0:
                    # Positive d cannot balance opposite signs.
                    first, second = sorted((row, column))
                    certificate = SymmetrizerCertificate(
                        SymmetrizerObstruction.OPPOSITE_SIGNS,
                        (first, second),
                    )
                    raise cls._failure(certificate, entries)

                candidate = values[row] * Fraction(forward, backward)
                if column in values:
                    if values[column] != candidate:
                        # A positive solution forces every cycle product to 1.
                        cycle = cls._fundamental_cycle(row, column, parent, depth)
                        certificate = SymmetrizerCertificate(
                            SymmetrizerObstruction.INCONSISTENT_CYCLE,
                            cycle,
                        )
                        raise cls._failure(certificate, entries)
                    continue
                values[column] = candidate
                parent[column] = row
                depth[column] = depth[row] + 1
                queue.append(column)

        denominator = 1
        for value in values.values():
            denominator = lcm(denominator, value.denominator)
        integers = {
            index: value.numerator * (denominator // value.denominator)
            for index, value in values.items()
        }
        divisor = 0
        for value in integers.values():
            divisor = gcd(divisor, value)
        return {index: value // divisor for index, value in integers.items()}

    @staticmethod
    def _failure(
        certificate: SymmetrizerCertificate,
        entries: tuple[tuple[int, ...], ...],
    ) -> NotSymmetrizableError:
        """Build the exception carrying a verified certificate."""
        return NotSymmetrizableError(
            f"matrix is not symmetrizable: {certificate.describe(entries)}",
            certificate,
        )

    @staticmethod
    def _fundamental_cycle(
        first: int,
        second: int,
        parent: dict[int, int],
        depth: dict[int, int],
    ) -> tuple[int, ...]:
        """Return the simple cycle formed by the tree paths and the closing edge."""
        left: list[int] = []
        right: list[int] = []
        while depth[first] > depth[second]:
            left.append(first)
            first = parent[first]
        while depth[second] > depth[first]:
            right.append(second)
            second = parent[second]
        while first != second:
            left.append(first)
            first = parent[first]
            right.append(second)
            second = parent[second]
        return (*left, first, *right[::-1])

    @staticmethod
    def _is_symmetric(entries: tuple[tuple[int, ...], ...]) -> bool:
        """Return whether the matrix is symmetric."""
        size = len(entries)
        for row in range(size):
            for column in range(row + 1, size):
                if entries[row][column] != entries[column][row]:
                    return False
        return True
