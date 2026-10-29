# SPDX-FileCopyrightText: 2026-present Mario Abarca <knkillname@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Exact inertia of symmetric quasi-Cartan matrices."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from dynkinlab._matrix import MatrixLike


@dataclass(frozen=True)
class Inertia:
    """The positive, negative, and zero eigenvalue counts of a symmetric matrix.

    Parameters
    ----------
    positive : int
        Number of positive eigenvalues.
    negative : int
        Number of negative eigenvalues.
    zero : int
        Number of zero eigenvalues.

    Raises
    ------
    ValueError
        If a count is negative or is not an integer.
    """

    positive: int
    negative: int
    zero: int

    def __post_init__(self) -> None:
        """Validate that all counts are non-negative integers."""
        for name, value in (
            ("positive", self.positive),
            ("negative", self.negative),
            ("zero", self.zero),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def rank(self) -> int:
        """Return the number of nonzero eigenvalues."""
        return self.positive + self.negative

    @property
    def signature(self) -> int:
        """Return the difference between the positive and negative counts."""
        return self.positive - self.negative

    @property
    def is_positive_definite(self) -> bool:
        """Return whether every eigenvalue is positive."""
        return self.positive > 0 and self.negative == 0 and self.zero == 0

    @property
    def is_negative_definite(self) -> bool:
        """Return whether every eigenvalue is negative."""
        return self.negative > 0 and self.positive == 0 and self.zero == 0

    @property
    def is_degenerate(self) -> bool:
        """Return whether the matrix has a zero eigenvalue."""
        return self.zero > 0

    @classmethod
    def from_matrix(cls, matrix: MatrixLike) -> Inertia:
        """Compute inertia exactly by symmetric congruence reduction.

        Parameters
        ----------
        matrix : MatrixLike
            Matrix-like value, usually a `QuasiCartanMatrix`, whose inertia is computed.

        Returns
        -------
        Inertia
            Exact positive, negative, and zero eigenvalue counts.
        """
        size = matrix.dimension
        values = [[Fraction(entry) for entry in row] for row in matrix.entries]
        positive = 0
        negative = 0
        zero = 0
        index = 0

        while index < size:
            pivot = values[index][index]

            if pivot != 0:
                if pivot > 0:
                    positive += 1
                else:
                    negative += 1
                cls._eliminate_one(values, index, size)
                index += 1
                continue

            next_index = cls._next_off_diagonal(values, index, size)
            if next_index is None:
                zero += 1
                index += 1
                continue

            cls._prepare_two_by_two(values, index, next_index)
            positive += 1
            negative += 1
            cls._eliminate_two(values, index, size)
            index += 2

        return cls(positive, negative, zero)

    @staticmethod
    def _eliminate_one(values: list[list[Fraction]], index: int, size: int) -> None:
        """Eliminate one pivot row and column by congruence."""
        pivot = values[index][index]
        for row in range(index + 1, size):
            for column in range(index + 1, size):
                values[row][column] -= values[row][index] * values[index][column] / pivot

        for row in range(index + 1, size):
            values[row][index] = Fraction(0)
            values[index][row] = Fraction(0)

    @staticmethod
    def _next_off_diagonal(values: list[list[Fraction]], index: int, size: int) -> int | None:
        """Return the first row below ``index`` with a nonzero coupling."""
        return next(
            (row for row in range(index + 1, size) if values[row][index] != 0),
            None,
        )

    @staticmethod
    def _prepare_two_by_two(
        values: list[list[Fraction]],
        index: int,
        next_index: int,
    ) -> None:
        """Bring a nonzero off-diagonal entry into the 2x2 pivot block."""
        if next_index == index + 1:
            return

        values[index + 1], values[next_index] = values[next_index], values[index + 1]
        for row in values:
            row[index + 1], row[next_index] = row[next_index], row[index + 1]

    @staticmethod
    def _eliminate_two(values: list[list[Fraction]], index: int, size: int) -> None:
        """Eliminate the remaining block with a 2x2 pivot."""
        coupling = values[index + 1][index]
        diagonal = values[index + 1][index + 1]
        for row in range(index + 2, size):
            for column in range(index + 2, size):
                row_coupling = values[row][index]
                diagonal_row_coupling = values[row][index + 1]
                column_coupling = values[index][column]
                diagonal_column_coupling = values[index + 1][column]
                values[row][column] -= (
                    -diagonal * row_coupling * column_coupling / (coupling * coupling)
                    + row_coupling * diagonal_column_coupling / coupling
                    + diagonal_row_coupling * column_coupling / coupling
                )

        for row in range(index + 2, size):
            values[row][index] = Fraction(0)
            values[index][row] = Fraction(0)
            values[row][index + 1] = Fraction(0)
            values[index + 1][row] = Fraction(0)
