"""Infinite matrix class.

An infinite matrix is a matrix that has no fixed size. It can be
indexed with any non-negative integer, and will automatically
grow to accomodate the index. The matrix is stored as a sparse
matrix, so it is efficient to store and access elements that
are not explicitly set.
"""
from collections.abc import Iterable, Sequence, Iterator
from typing import Any
from scipy import sparse
import numpy as np

Indexer = int | slice | Sequence[int]
ArrayLike = Sequence[Sequence[Any]] | Sequence[Any] | Any


class InfiniteMatrix:
    """Infinite matrix class.

    An infinite matrix is a matrix that has no fixed size. It can be
    indexed with any non-negative integer, and will automatically
    grow to accomodate the index. The matrix is stored as a sparse
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

    def __init__(
        self,
        data: Iterable[tuple[int, int, Any]] | None = None,
        dtype: type | None = None,
    ):
        """Infinite matrix class.

        An infinite matrix is a matrix that has no fixed size. It can be
        indexed with any non-negative integer, and will automatically
        grow to accomodate the index. The matrix is stored as a sparse
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
            if isinstance(data, InfiniteMatrix):
                dtype = data.dtype
            elif data is not None:
                data = list(data)
                try:
                    dtype = type(data[0][2])
                except IndexError:
                    raise ValueError("Cannot infer dtype from empty data")
            else:
                raise ValueError("Either data or dtype must be provided")
        self._dtype = dtype

        self._zero = dtype()

        # Set the data
        if data is not None:
            for i, j, v in data:
                self[i, j] = v

    @property
    def dtype(self) -> type:
        """The data type of the matrix entries."""
        return self._dtype

    def to_sparse(self) -> sparse.coo_matrix:
        """Convert the matrix to a sparse matrix.

        Returns
        -------
        sparse.coo_matrix
            The sparse matrix.
        """
        i, j, data = zip(*self._iter_data())
        return sparse.coo_matrix((data, (i, j)), dtype=self._dtype)

    def __repr__(self) -> str:
        """Return a string representation of the matrix."""
        data = list(self._iter_data())
        if not data:
            return f"InfiniteMatrix(dtype={self.dtype.__name__})"
        return f"InfiniteMatrix({data}, dtype={self.dtype.__name__})"

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
