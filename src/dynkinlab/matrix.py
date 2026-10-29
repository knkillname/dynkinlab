# SPDX-FileCopyrightText: 2026-present Mario Abarca <knkillname@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Structural matrix inputs for quadratic forms and inertia."""

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
