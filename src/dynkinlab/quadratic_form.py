# SPDX-FileCopyrightText: 2026-present Mario Abarca <knkillname@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Integer quadratic forms associated with quasi-Cartan matrices."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from dynkinlab._matrix import MatrixLike


@dataclass(frozen=True)
class QuadraticForm:
    """The integer quadratic form of a quasi-Cartan matrix.

    Parameters
    ----------
    matrix : MatrixLike
        Matrix-like value, usually a `QuasiCartanMatrix`, whose quadratic form is evaluated.
    """

    matrix: MatrixLike

    def evaluate(self, vector: Sequence[int]) -> int:
        """Evaluate the form at an integer vector.

        Parameters
        ----------
        vector : Sequence[int]
            Integer coordinates with one entry per matrix dimension.

        Returns
        -------
        int
            The exact value ``sum(x_i**2) + sum(a_ij * x_i * x_j for i < j)``.

        Raises
        ------
        ValueError
            If the vector has the wrong length or contains a non-integer entry.
        """
        try:
            values = tuple(vector)
        except TypeError as error:
            raise ValueError("vector must be an iterable of integers") from error

        if len(values) != self.matrix.dimension:
            raise ValueError(
                f"vector must have {self.matrix.dimension} entries, got {len(values)}",
            )

        for value in values:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("vector entries must be integers")

        total = sum(value * value for value in values)
        rows = self.matrix.entries
        for row_index in range(self.matrix.dimension):
            for column_index in range(row_index + 1, self.matrix.dimension):
                total += rows[row_index][column_index] * values[row_index] * values[column_index]
        return total
