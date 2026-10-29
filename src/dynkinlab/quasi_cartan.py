# SPDX-FileCopyrightText: 2026-present Mario Abarca <knkillname@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Immutable quasi-Cartan matrices and their flation operation."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from dynkinlab import display
from dynkinlab.display import DisplayOptions
from dynkinlab.inertia import Inertia
from dynkinlab.quadratic_form import QuadraticForm
from dynkinlab.symmetrizer import Symmetrizer

if TYPE_CHECKING:
    from dynkinlab.dynkin import DynkinClassification

_DIAGONAL_VALUE = 2


@dataclass(frozen=True, init=False)
class QuasiCartanMatrix:
    """An immutable integer matrix with diagonal entries equal to 2.

    A quasi-Cartan matrix admits a positive diagonal integer symmetrizer `D`
    such that `D A` is symmetric. Symmetric matrices are the special case
    `D = (1, ..., 1)`.

    Parameters
    ----------
    entries : Iterable[Iterable[int]]
        Square matrix rows. The input is copied and must be non-empty, square,
        integer-valued, and have every diagonal entry equal to 2.
    symmetrizer : Iterable[int] | None
        Optional positive diagonal symmetrizer. When omitted, the canonical
        minimal-trace symmetrizer is computed. When supplied, it is validated
        and normalized to the canonical value.

    Raises
    ------
    ValueError
        If the input is empty, ragged, non-square, non-integer, has a diagonal
        entry other than 2, or is not symmetrizable.
    """

    entries: tuple[tuple[int, ...], ...]
    symmetrizer: tuple[int, ...]

    def __init__(
        self,
        entries: Iterable[Iterable[int]],
        symmetrizer: Iterable[int] | None = None,
    ) -> None:
        rows = self._copy_rows(entries)
        self._validate_shape(rows)
        self._validate_values(rows)
        self._validate_diagonal(rows)

        if symmetrizer is None:
            if self._is_symmetric_rows(rows):
                canonical = (1,) * len(rows)
            else:
                canonical = Symmetrizer.canonical(rows)
        else:
            canonical = Symmetrizer.normalize(rows, symmetrizer)

        object.__setattr__(self, "entries", rows)
        object.__setattr__(self, "symmetrizer", canonical)

    @classmethod
    def symmetric(cls, entries: Iterable[Iterable[int]]) -> "QuasiCartanMatrix":
        """Construct a matrix that is required to be symmetric.

        Parameters
        ----------
        entries : Iterable[Iterable[int]]
            Square matrix rows.

        Returns
        -------
        QuasiCartanMatrix
            The validated symmetric matrix.

        Raises
        ------
        ValueError
            If the matrix is not symmetric or is not a valid quasi-Cartan matrix.
        """
        rows = cls._copy_rows(entries)
        cls._validate_shape(rows)
        cls._validate_values(rows)
        cls._validate_diagonal(rows)
        if not cls._is_symmetric_rows(rows):
            raise ValueError("matrix must be symmetric")
        return cls(rows)

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
    def _is_symmetric_rows(rows: tuple[tuple[int, ...], ...]) -> bool:
        """Return whether the rows form a symmetric matrix."""
        width = len(rows)
        for row_index, row in enumerate(rows):
            for column_index in range(row_index + 1, width):
                if row[column_index] != rows[column_index][row_index]:
                    return False
        return True

    @property
    def dimension(self) -> int:
        """Return the number of rows and columns."""
        return len(self.entries)

    @property
    def is_symmetric(self) -> bool:
        """Return whether the original matrix is symmetric."""
        return self._is_symmetric_rows(self.entries)

    @property
    def symmetrized_entries(self) -> tuple[tuple[int, ...], ...]:
        """Return the symmetrized matrix ``S = D A``."""
        return tuple(
            tuple(self.symmetrizer[row] * value for value in entries)
            for row, entries in enumerate(self.entries)
        )

    @property
    def quadratic_form(self) -> QuadraticForm:
        """Return the integer quadratic form associated with this matrix."""
        return QuadraticForm(self)

    @property
    def inertia(self) -> Inertia:
        """Return the exact inertia of the symmetrized matrix."""
        return Inertia.from_matrix(self)

    def classify(self) -> "DynkinClassification":
        """Classify this matrix using the cubic Dynkin classifier."""
        from dynkinlab.classifier import DynkinClassifier

        return DynkinClassifier.classify(self)

    @staticmethod
    def _validate_index(index: int, size: int, name: str) -> None:
        """Validate a vertex index."""
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError(f"{name} must be an integer")
        if not 0 <= index < size:
            raise ValueError(f"{name} must be between 0 and {size - 1}")

    def flation(self, source: int, target: int) -> "QuasiCartanMatrix":
        """Apply the flation ``T_source,target`` and return a new matrix.

        Parameters
        ----------
        source : int
            Source vertex index.
        target : int
            Target vertex index. Equal source and target performs a sign
            inversion of every edge incident to the target.

        Returns
        -------
        QuasiCartanMatrix
            The flated matrix.

        Raises
        ------
        ValueError
            If either index is invalid.
        """
        size = self.dimension
        self._validate_index(source, size, "source")
        self._validate_index(target, size, "target")
        old = self.entries

        if source == target:
            target_index = target
            rows = [list(row) for row in old]
            for index in range(size):
                if index == target_index:
                    continue
                rows[target_index][index] = -old[target_index][index]
                rows[index][target_index] = -old[index][target_index]
            return QuasiCartanMatrix(tuple(tuple(row) for row in rows))

        source_index = source
        target_index = target
        source_row = old[source_index]
        target_row = old[target_index]
        row_coefficient = old[target_index][source_index]
        column_coefficient = old[source_index][target_index]
        rows = [list(row) for row in old]

        for column in range(size):
            if column == target_index:
                continue
            rows[target_index][column] = target_row[column] - row_coefficient * source_row[column]
        for row in range(size):
            if row == target_index:
                continue
            rows[row][target_index] = (
                old[row][target_index] - column_coefficient * old[row][source_index]
            )
        rows[target_index][target_index] = _DIAGONAL_VALUE
        return QuasiCartanMatrix(tuple(tuple(row) for row in rows))

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
