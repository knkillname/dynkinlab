"""Infinite matrix class.

An infinite matrix is a matrix that has no fixed size. It can be
indexed with any non-negative integer, and will automatically
grow to accommodate the index. The matrix is stored as a sparse
matrix, so it is efficient to store and access elements that
are not explicitly set.
"""
import io
import itertools
import json
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np
from scipy import sparse

Indexer = int | slice | Sequence[int]
ArrayLike = Sequence[Sequence[Any]] | Sequence[Any] | Any


class InfMatrix:
    """Infinite matrix class.

    An infinite matrix is a matrix that has no fixed size. It can be
    indexed with any non-negative integer, and will automatically
    grow to accommodate the index. The matrix is stored as a sparse
    matrix, so it is efficient to store and access elements that
    are not explicitly set.

    Parameters
    ----------
    data : Iterable[tuple[int, int, Any]], optional
        An iterable of tuples of the form (i, j, v), where i and j
        are the indices of the element and v is the value of the
        element. If provided, the matrix will be initialized with
        the given data. If not provided, the matrix will be
        initialized with all zeros.
    dtype : type, optional
        The data type of the matrix. If not provided, the data type
        will be inferred from the data. If no data is provided, a
        default data type of float will be used.

    Attributes
    ----------
    dtype : type
        The data type of the matrix.
    """

    class _CoordinateRecord(TypedDict):
        """A record of a matrix element."""

        i: int
        j: int
        value: Any

    class _DictRepresentation(TypedDict):
        """Dictionary representation of the matrix."""

        data: list["InfMatrix._CoordinateRecord"]

    _InfMatrixData = (
        Iterable[tuple[int, int, Any]] | Iterable["InfMatrix._CoordinateRecord"]
    )

    def __init__(
        self,
        data: "_InfMatrixData" | None = None,
        dtype: type | None = None,
    ):
        """Infinite matrix class.

        An infinite matrix is a matrix that has no fixed size. It can be
        indexed with any non-negative integer, and will automatically
        grow to accommodate the index. The matrix is stored as a sparse
        matrix, so it is efficient to store and access elements that
        are not explicitly set.

        Parameters
        ----------
        data : Iterable[tuple[int, int, Any]], optional
            An iterable of tuples of the form (i, j, v), where i and j
            are the indices of the element and v is the value of the
            element. If provided, the matrix will be initialized with
            the given data. If not provided, the matrix will be
            initialized with all zeros.
        dtype : type, optional
            The data type of the matrix. If not provided, the data type
            will be inferred from the data. If no data is provided, a
            ValueError will be raised.

        Raises
        ------
        ValueError
            If neither data nor dtype is provided.
        ValueError
            If the data is empty and the dtype cannot be inferred.
        """
        self._data: dict[int, dict[int, Any]] = {}

        # Infer the dtype from the data if it is not provided
        if dtype is None:
            if data is None:
                raise ValueError("Either data or dtype must be provided.")
            data, data_aux = itertools.tee(iter(data))  # type: ignore
            dtype = self._infer_dtype(data_aux)  # type: ignore

        self._dtype = dtype
        self._zero = dtype(0)

        if data is not None:
            data = cast(InfMatrix._InfMatrixData, data)
            self._build_data(data)

    @property
    def dtype(self) -> type:
        """The data type of the matrix entries."""
        return self._dtype

    def to_sparse(self) -> sparse.coo_array:
        """Convert the matrix to a sparse matrix.

        Returns
        -------
        sparse.coo_matrix
            The sparse matrix.
        """
        i, j, data = zip(*self._iter_data())
        return sparse.coo_array((data, (i, j)), dtype=self._dtype)

    def to_dense(self) -> np.ndarray:
        """Convert the matrix to a dense array form."""
        return self.to_sparse().todense()

    def to_dict(self) -> "InfMatrix._DictRepresentation":
        """Convert the matrix to a dictionary representation.

        Returns
        -------
        dict
            A dictionary representation of the matrix.
        """
        as_record = self._CoordinateRecord
        data = (as_record(i=i, j=j, value=v) for i, j, v in self._iter_data())
        return {"data": list(data)}

    def to_json(self, file: str | io.TextIOBase | Path, **kwargs) -> None:
        """Write the matrix to a JSON file.

        Parameters
        ----------
        file : Path or file object
            The file to write to.
        **kwargs
            Additional keyword arguments to pass to json.dump.
        """
        if isinstance(file, (str, Path)):
            with open(file, "w") as f:
                json.dump(self.to_dict(), f, **kwargs)
        else:
            json.dump(self.to_dict(), file, **kwargs)

    def from_json(self, file: str | io.TextIOBase | Path) -> "InfMatrix":
        """Read the matrix from a JSON file.

        Parameters
        ----------
        file : Path or file object
            The file to read from.
        """
        if isinstance(file, (str, Path)):
            with open(file, "r") as f:
                data = json.load(f)
        else:
            data = json.load(file)
        return InfMatrix(data["data"])

    def __repr__(self) -> str:
        """Return a string representation of the matrix."""
        data = list(self._iter_data())
        if not data:
            return f"InfiniteMatrix(dtype={self.dtype.__name__})"
        return f"InfiniteMatrix({data}, dtype={self.dtype.__name__})"

    def __getitem__(self, index: tuple[Indexer, Indexer]) -> Any | np.ndarray:
        """Get an element or submatrix of the matrix."""
        i, j = index
        if isinstance(i, int) and isinstance(j, int):
            return self._get_item(i, j)
        elif isinstance(i, int):
            assert not isinstance(j, int)
            return self._get_row(i, j)
        elif isinstance(j, int):
            assert not isinstance(i, int)
            return self._get_col(i, j)
        return self._get_submatrix(i, j)

    def __setitem__(
        self,
        index: tuple[Indexer, Indexer],
        value: ArrayLike,
    ) -> None:
        """Set an element or submatrix of the matrix."""
        i, j = index
        if isinstance(i, int) and isinstance(j, int):
            self._set_item(i, j, value)
        elif isinstance(i, int):
            assert not isinstance(j, int)
            self._set_row(i, j, value)
        elif isinstance(j, int):
            assert not isinstance(i, int)
            self._set_column(i, j, value)
        else:
            self._set_submatrix(i, j, value)

    # Elementary row and column operations
    # ------------------------------------
    def swap_rows(self, r: int, s: int) -> None:
        """Swap the rows r and s.

        Parameters
        ----------
        r : int
            The index of the first row.
        s : int
            The index of the second row.
        """
        has_r, has_s = r in self._data, s in self._data
        if has_r and has_s:
            self._data[r], self._data[s] = self._data[s], self._data[r]
        elif has_r:
            self._data[s] = self._data.pop(r)
        elif has_s:
            self._data[r] = self._data.pop(s)

    def swap_columns(self, r: int, s: int) -> None:
        """Swap the columns r and s.

        Parameters
        ----------
        r : int
            The index of the first column.
        s : int
            The index of the second column.
        """
        for row in self._data.values():
            has_r, has_s = r in row, s in row
            if has_r and has_s:
                row[r], row[s] = row[s], row[r]
            elif has_r:
                row[s] = row.pop(r)
            elif has_s:
                row[r] = row.pop(s)

    def scale_row(self, r: int, alpha: Any) -> None:
        """Scale the row r by alpha.

        Parameters
        ----------
        r : int
            The index of the row to scale.
        alpha : Any
            The scaling factor.
        """
        try:
            column_indices = list(self._data[r])
        except KeyError:
            pass
        else:
            self[r, column_indices] *= alpha

    def scale_column(self, r: int, alpha: Any) -> None:
        """Scale the column r by alpha.

        Parameters
        ----------
        r : int
            The index of the column to scale.
        alpha : Any
            The scaling factor.
        """
        try:
            # Get the row indices of the non-zero elements in column r.
            column_indices = [i for i in self._data if r in self._data[i]]
        except KeyError:
            # If column r is zero-filled, then nothing needs to be done.
            pass
        else:
            # Scale column r by alpha.
            self[column_indices, r] *= alpha

    def add_multiple_of_row(self, r: int, s: int, alpha: Any) -> None:
        """Add alpha times row r to row s.

        Parameters
        ----------
        r : int
            The index of the row to add.
        s : int
            The index of the row to add to.
        alpha : Any
            The scaling factor.
        """
        try:
            # Get the column indices of the non-zero elements in row r.
            col_indexes = list(self._data[r])
        except KeyError:
            # If row r is zero-filled, then nothing needs to be done.
            pass
        else:
            # Add alpha times row r to row s.
            self[s, col_indexes] += alpha * self[r, col_indexes]

    def add_multiple_of_column(self, r: int, s: int, alpha: Any) -> None:
        """Add alpha times column r to column s.

        Parameters
        ----------
        r : int
            The index of the column to add.
        s : int
            The index of the column to add to.
        alpha : Any
            The scaling factor.
        """
        row_indexes = list(self._data)
        self[row_indexes, s] += alpha * self[row_indexes, r]

    # Private methods
    # ---------------
    def _build_data(self, data: "_InfMatrixData"):
        """Build the data dictionary from the given data.

        Parameters
        ----------
        data : Iterable[tuple[int, int, Any]]
            The data to build the dictionary from.
        """
        data, data_aux = itertools.tee(iter(data))  # type: ignore
        first_element = next(data_aux, None)
        if isinstance(first_element, dict):
            data = cast(Iterator["InfMatrix._CoordinateRecord"], data)
            data = ((d["i"], d["j"], d["value"]) for d in data)
        data = cast(Iterator[tuple[int, int, Any]], data)
        for i, j, v in data:
            self[i, j] = v

    def _infer_dtype(self, data: Iterator[dict | tuple]) -> type:
        first_element = next(data, None)
        if first_element is None:
            raise ValueError("Cannot infer dtype from empty data.")
        if isinstance(first_element, dict):
            dtype = cast(type, type(first_element["value"]))
        else:
            dtype = cast(type, type(first_element[2]))
        return dtype

    def _iter_data(self) -> Iterator[tuple[int, int, Any]]:
        for i, row in self._data.items():
            for j, v in row.items():
                yield i, j, v

    def _get_item(self, i: int, j: int) -> Any:
        try:
            return self._data[i][j]
        except KeyError:
            return self._zero

    def _get_row(self, i: int, j: slice | Sequence[int]) -> np.ndarray:
        if isinstance(j, slice):
            if j.stop is None:
                raise IndexError("Cannot use infinite slices")
            j = range(j.start or 0, j.stop, j.step or 1)
        return np.array([self._get_item(i, k) for k in j], dtype=self._dtype)

    def _get_col(self, i: slice | Sequence[int], j: int) -> np.ndarray:
        if isinstance(i, slice):
            if i.stop is None:
                raise IndexError("Cannot use infinite slices")
            i = range(i.start or 0, i.stop, i.step or 1)
        return np.array([self._get_item(k, j) for k in i], dtype=self._dtype)

    def _get_submatrix(
        self, i: slice | Sequence[int], j: slice | Sequence[int]
    ) -> np.ndarray:
        if isinstance(i, slice):
            if i.stop is None:
                raise IndexError("Cannot use infinite slices")
            i = range(i.start or 0, i.stop, i.step or 1)
        if isinstance(j, slice):
            if j.stop is None:
                raise IndexError("Cannot use infinite slices")
            j = range(j.start or 0, j.stop, j.step or 1)
        return np.array(
            [[self._get_item(k, ell) for ell in j] for k in i], dtype=self._dtype
        )

    def _set_item(self, i: int, j: int, value: Any) -> None:
        value = self._dtype(value)
        if value == self._zero:
            try:
                del self._data[i][j]
            except KeyError:
                pass
            try:
                if not self._data[i]:
                    del self._data[i]
            except KeyError:
                pass
        else:
            try:
                self._data[i][j] = value
            except KeyError:
                self._data[i] = {j: value}

    def _set_row(self, i: int, j: slice | Sequence[int], value: Sequence[Any]) -> None:
        if isinstance(j, slice):
            if j.stop is None:
                raise IndexError("Cannot use infinite slices")
            j = range(j.start or 0, j.stop, j.step or 1)
        if len(j) != len(value):
            raise ValueError("Number of columns in value does not match j")
        for k, v in zip(j, value):
            self._set_item(i, k, v)

    def _set_column(
        self, i: slice | Sequence[int], j: int, value: Sequence[Any]
    ) -> None:
        if isinstance(i, slice):
            if i.stop is None:
                raise IndexError("Cannot use infinite slices")
            i = range(i.start or 0, i.stop, i.step or 1)
        if len(i) != len(value):
            raise ValueError("Number of rows in value does not match i")
        for k, v in zip(i, value):
            self._set_item(k, j, v)

    def _set_submatrix(
        self,
        i: slice | Sequence[int],
        j: slice | Sequence[int],
        value: Sequence[Sequence[Any]],
    ) -> None:
        if isinstance(i, slice):
            if i.stop is None:
                raise IndexError("Cannot use infinite slices")
            i = range(i.start or 0, i.stop, i.step or 1)
        if isinstance(j, slice):
            if j.stop is None:
                raise IndexError("Cannot use infinite slices")
            j = range(j.start or 0, j.stop, j.step or 1)
        if len(i) != len(value):
            raise ValueError("Number of rows in value does not match i")
        if len(j) != len(value[0]):
            raise ValueError("Number of columns in value does not match j")
        for k, row in zip(i, value):
            for ell, v in zip(j, row):
                self._set_item(k, ell, v)
