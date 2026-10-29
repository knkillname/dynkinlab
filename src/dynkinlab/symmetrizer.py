# SPDX-FileCopyrightText: 2026-present Mario Abarca <knkillname@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Canonical positive diagonal symmetrizers for quasi-Cartan matrices."""

from collections.abc import Iterable
from fractions import Fraction
from math import gcd, lcm


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
        ValueError
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
            Components as sorted vertex-index tuples.
        """
        size = len(entries)
        unseen = set(range(size))
        components: list[tuple[int, ...]] = []
        while unseen:
            root = min(unseen)
            unseen.remove(root)
            queue = [root]
            component = [root]
            while queue:
                row = queue.pop()
                for column in range(size):
                    if column == row or column not in unseen:
                        continue
                    if entries[row][column] == 0 and entries[column][row] == 0:
                        continue
                    unseen.remove(column)
                    queue.append(column)
                    component.append(column)
            components.append(tuple(sorted(component)))
        return tuple(components)

    @classmethod
    def _component_values(
        cls,
        entries: tuple[tuple[int, ...], ...],
        component: tuple[int, ...],
    ) -> dict[int, int]:
        """Return primitive integer ratios for one connected component."""
        root = component[0]
        values: dict[int, Fraction] = {root: Fraction(1)}
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
                    raise ValueError("matrix is not symmetrizable")
                if forward * backward < 0:
                    raise ValueError("matrix is not symmetrizable")

                candidate = values[row] * Fraction(forward, backward)
                if column in values:
                    if values[column] != candidate:
                        raise ValueError("matrix is not symmetrizable")
                    continue
                values[column] = candidate
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
    def _is_symmetric(entries: tuple[tuple[int, ...], ...]) -> bool:
        """Return whether the matrix is symmetric."""
        size = len(entries)
        for row in range(size):
            for column in range(row + 1, size):
                if entries[row][column] != entries[column][row]:
                    return False
        return True
