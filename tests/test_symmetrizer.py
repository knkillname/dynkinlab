# SPDX-FileCopyrightText: 2026-present Mario Abarca <knkillname@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Tests for canonical symmetrizers and their nonexistence certificates."""

import pickle

import pytest

from dynkinlab import (
    NotSymmetrizableError,
    QuasiCartanMatrix,
    SymmetrizerCertificate,
    SymmetrizerObstruction,
)

_NON_SYMMETRIZABLE = ((2, -1, -2), (-2, 2, -1), (-1, -2, 2))
_ZERO_ENTRY = ((2, 0), (-1, 2))
_OPPOSITE_SIGNS = ((2, 1), (-1, 2))
_MULTIPLE_FAILURES = (
    (2, 0, 0, 0),
    (-1, 2, 0, 0),
    (0, 0, 2, 1),
    (0, 0, -1, 2),
)


class TestCertificateVerification:
    """Verify each certificate against matching and mismatching matrices."""

    def test_zero_entry_certificate(self) -> None:
        certificate = SymmetrizerCertificate(
            SymmetrizerObstruction.ZERO_ENTRY,
            (0, 1),
        )

        assert certificate.verify(_ZERO_ENTRY)
        assert "zero entry" in certificate.describe(_ZERO_ENTRY)
        assert not certificate.verify(_OPPOSITE_SIGNS)

    def test_reversed_zero_entry_certificate(self) -> None:
        certificate = SymmetrizerCertificate(
            SymmetrizerObstruction.ZERO_ENTRY,
            (1, 0),
        )

        assert certificate.verify(((2, -1), (0, 2)))

    def test_opposite_signs_certificate(self) -> None:
        certificate = SymmetrizerCertificate(
            SymmetrizerObstruction.OPPOSITE_SIGNS,
            (0, 1),
        )

        assert certificate.verify(_OPPOSITE_SIGNS)
        assert "opposite signs" in certificate.describe(_OPPOSITE_SIGNS)
        assert not certificate.verify(_ZERO_ENTRY)

    def test_inconsistent_cycle_certificate(self) -> None:
        certificate = SymmetrizerCertificate(
            SymmetrizerObstruction.INCONSISTENT_CYCLE,
            (0, 1, 2),
        )

        assert certificate.verify(_NON_SYMMETRIZABLE)
        assert "inconsistent cycle" in certificate.describe(_NON_SYMMETRIZABLE)
        assert not certificate.verify(
            ((2, -1, -1), (-1, 2, -1), (-1, -1, 2)),
        )

    def test_malformed_certificates_do_not_verify(self) -> None:
        too_short = SymmetrizerCertificate(
            SymmetrizerObstruction.INCONSISTENT_CYCLE,
            (0, 1),
        )
        repeated = SymmetrizerCertificate(
            SymmetrizerObstruction.INCONSISTENT_CYCLE,
            (0, 1, 1, 2),
        )
        outside = SymmetrizerCertificate(
            SymmetrizerObstruction.ZERO_ENTRY,
            (0, 9),
        )

        assert not too_short.verify(_NON_SYMMETRIZABLE)
        assert not repeated.verify(_NON_SYMMETRIZABLE)
        assert not outside.verify(_ZERO_ENTRY)

    def test_describe_rejects_a_tampered_certificate(self) -> None:
        certificate = SymmetrizerCertificate(
            SymmetrizerObstruction.ZERO_ENTRY,
            (0, 1),
        )

        with pytest.raises(ValueError, match="does not verify"):
            certificate.describe(_OPPOSITE_SIGNS)


class TestNotSymmetrizableError:
    """Verify the exception contract for all three obstructions."""

    @pytest.mark.parametrize(
        ("entries", "kind"),
        [
            (_ZERO_ENTRY, SymmetrizerObstruction.ZERO_ENTRY),
            (_OPPOSITE_SIGNS, SymmetrizerObstruction.OPPOSITE_SIGNS),
            (_NON_SYMMETRIZABLE, SymmetrizerObstruction.INCONSISTENT_CYCLE),
        ],
    )
    def test_construction_attaches_a_verifiable_certificate(
        self,
        entries: tuple[tuple[int, ...], ...],
        kind: SymmetrizerObstruction,
    ) -> None:
        with pytest.raises(NotSymmetrizableError, match="symmetrizable") as info:
            QuasiCartanMatrix(entries)

        error = info.value
        assert isinstance(error, ValueError)
        assert error.certificate.kind is kind
        assert error.certificate.verify(entries)

    def test_pickle_round_trip_preserves_certificate(self) -> None:
        with pytest.raises(NotSymmetrizableError) as info:
            QuasiCartanMatrix(_NON_SYMMETRIZABLE)

        restored = pickle.loads(pickle.dumps(info.value))

        assert isinstance(restored, NotSymmetrizableError)
        assert str(restored) == str(info.value)
        assert restored.certificate == info.value.certificate

    def test_validation_failure_is_not_a_certificate(self) -> None:
        with pytest.raises(ValueError) as info:
            QuasiCartanMatrix(())

        assert not isinstance(info.value, NotSymmetrizableError)

    def test_first_failure_is_deterministic(self) -> None:
        with pytest.raises(NotSymmetrizableError) as info:
            QuasiCartanMatrix(_MULTIPLE_FAILURES)

        assert info.value.certificate.kind is SymmetrizerObstruction.ZERO_ENTRY
        assert info.value.certificate.vertices == (0, 1)
