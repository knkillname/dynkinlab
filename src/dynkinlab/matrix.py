# SPDX-FileCopyrightText: 2026-present Mario Abarca <knkillname@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Structural matrix inputs and the explicit symmetrized matrix adapter."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol


class MatrixLike(Protocol):
    """The matrix interface required by quadratic forms and inertia.

    Attributes
    ----------
    dimension : int
        Number of rows and columns.
    entries : tuple[tuple[int, ...], ...]
        Matrix rows.
    """

    @property
    def dimension(self) -> int:
        """Return the number of rows and columns."""
        ...

    @property
    def entries(self) -> tuple[tuple[int, ...], ...]:
        """Return the matrix rows."""
        ...


@dataclass(frozen=True, init=False)
class SymmetrizedMatrix:
    """An immutable square integer matrix used for form and inertia evaluation.

    Parameters
    ----------
    entries : Iterable[Iterable[int]]
        Square integer rows. The input is copied and validated.

    Raises
    ------
    ValueError
        If the input is empty, ragged, non-square, or contains a non-integer.
    """

    entries: tuple[tuple[int, ...], ...]

    def __init__(self, entries: Iterable[Iterable[int]]) -> None:
        try:
            rows = tuple(tuple(row) for row in entries)
        except TypeError as error:
            raise ValueError("matrix entries must be an iterable of rows") from error

        if not rows:
            raise ValueError("matrix must have at least one row")
        width = len(rows[0])
        if width == 0:
            raise ValueError("matrix rows must not be empty")
        if any(len(row) != width for row in rows):
            raise ValueError("matrix must be square and rectangular")
        if len(rows) != width:
            raise ValueError("matrix must be square")
        for row in rows:
            for value in row:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError("matrix entries must be integers")

        object.__setattr__(self, "entries", rows)

    @property
    def dimension(self) -> int:
        """Return the number of rows and columns."""
        return len(self.entries)
