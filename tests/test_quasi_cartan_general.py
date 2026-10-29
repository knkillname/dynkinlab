# SPDX-FileCopyrightText: 2026-present Mario Abarca <knkillname@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Tests for general symmetrizable quasi-Cartan matrices and flation."""

import random

import numpy as np
import pytest

from dynkinlab.inertia import Inertia
from dynkinlab.quasi_cartan import QuasiCartanMatrix
from dynkinlab.symmetrizer import Symmetrizer

_B2 = ((2, -1), (-2, 2))
_B2_SYMMETRIZED = ((4, -2), (-2, 2))
_A2 = ((2, -1), (-1, 2))


class TestGeneralConstruction:
    """Verify symmetrizable construction and canonical symmetrizers."""

    def test_symmetric_matrix_uses_identity_symmetrizer(self) -> None:
        matrix = QuasiCartanMatrix(_A2)

        assert matrix.dimension == 2
        assert matrix.entries == _A2
        assert matrix.symmetrizer == (1, 1)
        assert matrix.symmetrized_entries == _A2
        assert matrix.is_symmetric is True

    def test_non_symmetric_matrix_uses_canonical_symmetrizer(self) -> None:
        matrix = QuasiCartanMatrix(_B2)

        assert matrix.entries == _B2
        assert matrix.symmetrizer == (2, 1)
        assert matrix.symmetrized_entries == _B2_SYMMETRIZED
        assert matrix.is_symmetric is False

    def test_explicit_symmetrizer_is_normalized(self) -> None:
        matrix = QuasiCartanMatrix(_B2, (4, 2))

        assert matrix.symmetrizer == (2, 1)

    def test_invalid_explicit_symmetrizer_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="compatible"):
            QuasiCartanMatrix(_B2, (1, 1))
        with pytest.raises(ValueError, match="positive"):
            QuasiCartanMatrix(_B2, (2, 0))
        with pytest.raises(ValueError, match="entries"):
            QuasiCartanMatrix(_B2, (2,))

    def test_non_symmetrizable_matrix_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="symmetrizable"):
            QuasiCartanMatrix(((2, -1, -2), (-2, 2, -1), (-1, -2, 2)))

    def test_disconnected_components_are_normalized_independently(self) -> None:
        matrix = QuasiCartanMatrix(
            (
                (2, -1, 0, 0),
                (-1, 2, 0, 0),
                (0, 0, 2, -1),
                (0, 0, -2, 2),
            ),
        )

        assert matrix.symmetrizer == (1, 1, 2, 1)

    def test_strict_symmetric_factory_accepts_symmetric(self) -> None:
        matrix = QuasiCartanMatrix.symmetric(_A2)

        assert matrix.is_symmetric is True

    def test_strict_symmetric_factory_rejects_non_symmetric(self) -> None:
        with pytest.raises(ValueError, match="symmetric"):
            QuasiCartanMatrix.symmetric(_B2)


class TestComponents:
    """Verify deterministic connected-component ordering."""

    def test_components_are_sorted_by_smallest_index(self) -> None:
        base = (
            (2, -1, 0, 0),
            (-1, 2, 0, 0),
            (0, 0, 2, -1),
            (0, 0, -1, 2),
        )
        permutation = (0, 2, 1, 3)
        entries = tuple(
            tuple(base[permutation[row]][permutation[column]] for column in range(4))
            for row in range(4)
        )

        assert Symmetrizer.components(entries) == ((0, 2), (1, 3))
        assert Symmetrizer.components(entries) == ((0, 2), (1, 3))


class TestGeneralFormAndInertia:
    """Verify form and inertia use the symmetrized matrix."""

    def test_non_symmetric_form_value(self) -> None:
        matrix = QuasiCartanMatrix(_B2)

        assert matrix.quadratic_form.evaluate((1, 1)) == 1

    def test_non_symmetric_inertia(self) -> None:
        matrix = QuasiCartanMatrix(_B2)

        assert matrix.inertia == Inertia(2, 0, 0)
        assert Inertia.from_matrix(matrix) == Inertia(2, 0, 0)

    def test_affine_inertia_remains_degenerate(self) -> None:
        matrix = QuasiCartanMatrix(((2, -1, -1), (-1, 2, -1), (-1, -1, 2)))

        assert matrix.inertia == Inertia(2, 0, 1)


class TestFlation:
    """Verify generalized flation behavior."""

    def test_dotted_edge_becomes_solid(self) -> None:
        matrix = QuasiCartanMatrix(((2, 1), (1, 2)))

        assert matrix.flation(0, 1).entries == _A2

    def test_sign_inversion_flips_incident_entries(self) -> None:
        matrix = QuasiCartanMatrix(((2, 1), (1, 2)))

        assert matrix.flation(0, 0).entries == _A2

    def test_flation_is_an_involution(self) -> None:
        matrix = QuasiCartanMatrix(_B2)

        assert matrix.flation(0, 1).flation(0, 1) == matrix

    def test_flation_preserves_inertia(self) -> None:
        matrix = QuasiCartanMatrix(_B2)

        assert matrix.flation(0, 1).inertia == matrix.inertia

    def test_flation_changes_only_the_target_row_and_column(self) -> None:
        matrix = QuasiCartanMatrix(
            (
                (2, 1, 0),
                (1, 2, 1),
                (0, 1, 2),
            ),
        )

        flated = matrix.flation(0, 1)

        assert flated.entries[0][2] == matrix.entries[0][2]
        assert flated.entries[2][0] == matrix.entries[2][0]
        assert flated.entries[2][2] == matrix.entries[2][2]

    @pytest.mark.parametrize(("source", "target"), [(-1, 0), (0, 2), (0, True)])
    def test_invalid_indices_are_rejected(self, source: object, target: object) -> None:
        matrix = QuasiCartanMatrix(_A2)

        with pytest.raises(ValueError, match="source|target"):
            matrix.flation(source, target)  # type: ignore[arg-type]


def _reference_flate(
    entries: tuple[tuple[int, ...], ...],
    source: int,
    target: int,
) -> tuple[tuple[int, ...], ...]:
    """Return the paper's flation formula using independent list arithmetic."""
    size = len(entries)
    rows = [list(row) for row in entries]
    if source == target:
        for index in range(size):
            if index == target:
                continue
            rows[target][index] = -entries[target][index]
            rows[index][target] = -entries[index][target]
        return tuple(tuple(row) for row in rows)

    for column in range(size):
        if column == target:
            continue
        rows[target][column] = (
            entries[target][column] - entries[target][source] * entries[source][column]
        )
    for row in range(size):
        if row == target:
            continue
        rows[row][target] = entries[row][target] - entries[source][target] * entries[row][source]
    rows[target][target] = 2
    return tuple(tuple(row) for row in rows)


class TestFlateEntries:
    """Verify the shared flation helper for exact and bounded dtypes."""

    def test_object_and_int64_match_for_small_entries(self) -> None:
        entries = ((2, 1, 0), (1, 2, 1), (0, 1, 2))

        exact = QuasiCartanMatrix.flate_entries(entries, 0, 1, object)
        bounded = QuasiCartanMatrix.flate_entries(entries, 0, 1, np.int64)

        assert tuple(map(tuple, exact.tolist())) == tuple(map(tuple, bounded.tolist()))

    def test_sign_inversion(self) -> None:
        entries = ((2, 1), (1, 2))

        result = QuasiCartanMatrix.flate_entries(entries, 1, 1, object)

        assert tuple(map(tuple, result.tolist())) == ((2, -1), (-1, 2))

    def test_object_dtype_keeps_huge_integers_exact(self) -> None:
        huge = 2**80
        matrix = QuasiCartanMatrix(((2, huge), (huge, 2)))

        flated = matrix.flation(0, 1)

        assert flated.entries == ((2, -huge), (-huge, 2))
        assert flated.flation(0, 1) == matrix

    def test_int64_overflow_guard(self) -> None:
        large = 2**31 + 1
        entries = np.array(((2, large), (large, 2)), dtype=np.int64)

        with pytest.raises(ValueError, match="int64"):
            QuasiCartanMatrix.flate_entries(entries, 0, 1, np.int64)

    def test_invalid_shape_and_indices_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="square"):
            QuasiCartanMatrix.flate_entries(((2, 1),), 0, 1, object)
        with pytest.raises(ValueError, match="source"):
            QuasiCartanMatrix.flate_entries(((2, 1), (1, 2)), -1, 1, object)
        with pytest.raises(ValueError, match="target"):
            QuasiCartanMatrix.flate_entries(((2, 1), (1, 2)), 0, 2, object)

    def test_randomized_reference_matches(self) -> None:
        rng = random.Random(1234)
        for size in range(1, 5):
            for _ in range(20):
                entries = tuple(
                    tuple(2 if row == column else rng.randint(-5, 5) for column in range(size))
                    for row in range(size)
                )
                for source in range(size):
                    for target in range(size):
                        expected = _reference_flate(entries, source, target)
                        actual = QuasiCartanMatrix.flate_entries(entries, source, target, object)

                        assert tuple(map(tuple, actual.tolist())) == expected
