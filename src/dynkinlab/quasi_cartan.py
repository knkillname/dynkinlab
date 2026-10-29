# SPDX-FileCopyrightText: 2026-present Mario Abarca <knkillname@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Immutable symmetric quasi-Cartan matrices."""

from collections.abc import Iterable
from dataclasses import dataclass

from dynkinlab import display
from dynkinlab.display import DisplayOptions
from dynkinlab.inertia import Inertia
from dynkinlab.quadratic_form import QuadraticForm

_DIAGONAL_VALUE = 2


@dataclass(frozen=True, init=False)
class QuasiCartanMatrix:
    """An immutable symmetric integer matrix with diagonal entries equal to 2.

    Parameters
    ----------
    entries : Iterable[Iterable[int]]
        Square matrix rows. The input is copied and must be non-empty, symmetric,
        and integer-valued, with every diagonal entry equal to 2.

    Raises
    ------
    ValueError
        If the input is empty, ragged, non-square, non-integer, asymmetric, or has
        a diagonal entry other than 2.
    """

    entries: tuple[tuple[int, ...], ...]

    def __init__(self, entries: Iterable[Iterable[int]]) -> None:
        object.__setattr__(self, "entries", self._validated_entries(entries))

    @classmethod
    def _validated_entries(cls, entries: Iterable[Iterable[int]]) -> tuple[tuple[int, ...], ...]:
        """Copy and validate the matrix rows."""
        rows = cls._copy_rows(entries)
        cls._validate_shape(rows)
        cls._validate_values(rows)
        cls._validate_diagonal(rows)
        cls._validate_symmetry(rows)
        return rows

    @staticmethod
    def _copy_rows(entries: Iterable[Iterable[int]]) -> tuple[tuple[int, ...], ...]:
        """Copy the input rows and normalize them to tuples."""
        try:
            return tuple(tuple(row) for row in entries)
        except TypeError as error:
            raise ValueError("matrix entries must be an iterable of rows") from error

    @staticmethod
    def _validate_shape(rows: tuple[tuple[int, ...], ...]) -> None:
        """Require a non-empty square matrix with non-empty rows."""
        if not rows:
            raise ValueError("matrix must have at least one row")
        width = len(rows[0])
        if width == 0:
            raise ValueError("matrix rows must not be empty")
        if any(len(row) != width for row in rows):
            raise ValueError("matrix must be square and rectangular")
        if len(rows) != width:
            raise ValueError("matrix must be square")

    @staticmethod
    def _validate_values(rows: tuple[tuple[int, ...], ...]) -> None:
        """Require every entry to be an integer and reject booleans."""
        for row in rows:
            for value in row:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError("matrix entries must be integers")

    @staticmethod
    def _validate_diagonal(rows: tuple[tuple[int, ...], ...]) -> None:
        """Require 2 on every diagonal entry."""
        for index, row in enumerate(rows):
            if row[index] != _DIAGONAL_VALUE:
                raise ValueError("diagonal entries must be 2")

    @staticmethod
    def _validate_symmetry(rows: tuple[tuple[int, ...], ...]) -> None:
        """Require a symmetric matrix."""
        width = len(rows)
        for row_index, row in enumerate(rows):
            for column_index in range(row_index + 1, width):
                if row[column_index] != rows[column_index][row_index]:
                    raise ValueError("matrix must be symmetric")

    @property
    def dimension(self) -> int:
        """Return the number of rows and columns."""
        return len(self.entries)

    @property
    def quadratic_form(self) -> QuadraticForm:
        """Return the integer quadratic form associated with this matrix."""
        return QuadraticForm(self)

    @property
    def inertia(self) -> Inertia:
        """Return the exact inertia of this matrix."""
        return Inertia.from_matrix(self)

    @staticmethod
    def _effective_options(options: DisplayOptions | None) -> DisplayOptions:
        """Return the explicit options or the canonical display options."""
        return display.options if options is None else options

    def to_markdown(self, options: DisplayOptions | None = None) -> str:
        """Render the matrix as a Markdown table.

        Parameters
        ----------
        options : DisplayOptions | None
            Per-call display options. The canonical options are used when omitted.

        Returns
        -------
        str
            Markdown table with row and column headers using the configured index base.
        """
        settings = self._effective_options(options)
        indices = tuple(
            range(
                settings.matrix_index_base,
                settings.matrix_index_base + self.dimension,
            ),
        )
        header = "|   | " + " | ".join(str(index) for index in indices) + " |"
        separator = "|---|" + "|".join("---:" for _ in indices) + "|"
        rows = [
            f"| {index} | " + " | ".join(str(value) for value in row) + " |"
            for index, row in zip(indices, self.entries, strict=True)
        ]
        return "\n".join((header, separator, *rows))

    def to_html(self, options: DisplayOptions | None = None) -> str:
        """Render the matrix as an HTML table.

        Parameters
        ----------
        options : DisplayOptions | None
            Per-call display options. The canonical options are used when omitted.

        Returns
        -------
        str
            HTML table with row and column headers using the configured index base.
        """
        settings = self._effective_options(options)
        indices = tuple(
            range(
                settings.matrix_index_base,
                settings.matrix_index_base + self.dimension,
            ),
        )
        header_cells = "".join(f"<th>{index}</th>" for index in indices)
        rows = []
        for index, row in zip(indices, self.entries, strict=True):
            body_cells = "".join(f"<td>{value}</td>" for value in row)
            rows.append(f"<tr><th>{index}</th>{body_cells}</tr>")
        return (
            '<table class="dynkinlab-matrix">'
            f"<thead><tr><th></th>{header_cells}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody>"
            "</table>"
        )

    def _repr_html_(self) -> str:
        """Return the canonical HTML representation for notebook display."""
        return self.to_html()
