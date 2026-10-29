# SPDX-FileCopyrightText: 2026-present Mario Abarca <knkillname@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Tests for general symmetrizable quasi-Cartan matrices and flation."""

import pytest

from dynkinlab.inertia import Inertia
from dynkinlab.matrix import SymmetrizedMatrix
from dynkinlab.quadratic_form import QuadraticForm
from dynkinlab.quasi_cartan import QuasiCartanMatrix

_B2 = ((2, -1), (-2, 2))
_B2_SYMMETRIZED = ((4, -2), (-2, 2))
_A2 = ((2, -1), (-1, 2))


class TestSymmetrizedMatrix:
    """Verify the explicit symmetrized matrix adapter."""

    def test_dimension_and_entries(self) -> None:
        matrix = SymmetrizedMatrix(((4, -2), (-2, 2)))

        assert matrix.dimension == 2
        assert matrix.entries == ((4, -2), (-2, 2))

    @pytest.mark.parametrize(
        "entries",
        [
            (),
            ((), ()),
            ((4, -2), (-2,)),
            ((4, "x"), ("x", 4)),
            ((4, True), (True, 4)),
        ],
    )
    def test_invalid_adapter_input_is_rejected(self, entries: object) -> None:
        with pytest.raises(ValueError):
            SymmetrizedMatrix(entries)  # type: ignore[arg-type]

    def test_form_uses_half_the_adapter_diagonal(self) -> None:
        form = QuadraticForm(SymmetrizedMatrix(((4, 0), (0, 4))))

        assert form.evaluate((1, 0)) == 2
        assert form.evaluate((1, 1)) == 4


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
