# SPDX-FileCopyrightText: 2026-present Mario Abarca <knkillname@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Tests for bounded bigraph solidification."""

import pytest

from dynkinlab import (
    Bigraph,
    FlationSequence,
    Inertia,
    NoSolidCertificate,
    QuasiCartanMatrix,
    SolidificationResult,
    SolidificationStatus,
)


def _bigraph(
    labels: tuple[str, ...],
    entries: tuple[tuple[int, ...], ...],
) -> Bigraph:
    """Build a labelled bigraph from symmetric entries."""
    return Bigraph(QuasiCartanMatrix(entries), labels)


def _dotted_a2() -> Bigraph:
    """Return a dotted A2 bigraph."""
    return _bigraph(("a", "b"), ((2, 1), (1, 2)))


def _solid_a2() -> Bigraph:
    """Return a solid A2 bigraph."""
    return _bigraph(("a", "b"), ((2, -1), (-1, 2)))


def _corank_two_psd() -> Bigraph:
    """Return the connected PSD matrix with all off-diagonal entries 2."""
    return _bigraph(
        ("a", "b", "c"),
        ((2, 2, 2), (2, 2, 2), (2, 2, 2)),
    )


def _indefinite_candidate() -> Bigraph:
    """Return the open indefinite candidate from the investigation."""
    return _bigraph(
        ("a", "b", "c", "d"),
        (
            (2, 2, 0, -1),
            (2, 2, -2, 2),
            (0, -2, 2, -2),
            (-1, 2, -2, 2),
        ),
    )


class TestResultValues:
    """Verify result, sequence, and certificate behavior."""

    def test_dotted_a2_solidifies_with_a_witness(self) -> None:
        bigraph = _dotted_a2()

        result = bigraph.solidify()

        assert result.status is SolidificationStatus.SOLIDIFIED
        assert len(result.sequence) == 1
        assert result.final is not None
        assert result.verify()
        assert result.final.matrix.entries[0][1] == -1

    def test_already_solid_uses_the_empty_sequence(self) -> None:
        bigraph = _solid_a2()

        result = bigraph.solidify()

        assert result.status is SolidificationStatus.SOLIDIFIED
        assert result.sequence == FlationSequence()
        assert result.final == bigraph
        assert result.explored_states == 0
        assert result.frame_classes == 0
        assert result.verify()

    def test_reversing_the_witness_recovers_the_input(self) -> None:
        bigraph = _dotted_a2()
        result = bigraph.solidify()
        assert result.final is not None

        recovered = result.sequence.inverse().apply(result.final)

        assert recovered == bigraph

    def test_labels_are_preserved_in_steps_and_final(self) -> None:
        bigraph = _bigraph(("left", "right"), ((2, 1), (1, 2)))

        result = bigraph.solidify()

        assert result.final is not None
        assert result.final.labels == ("left", "right")
        assert result.sequence.steps[0].source in bigraph.labels
        assert result.sequence.steps[0].target in bigraph.labels

    def test_result_markdown_and_notebook_hook_agree(self) -> None:
        result = _dotted_a2().solidify()

        markdown = result.to_markdown()

        assert "solidified" in markdown
        assert result._repr_markdown_() == markdown

    def test_certificate_markdown_and_notebook_hook_agree(self) -> None:
        result = _corank_two_psd().solidify()
        assert result.certificate is not None

        markdown = result.certificate.to_markdown()

        assert "No solid representative" in markdown
        assert result.certificate._repr_markdown_() == markdown

    def test_impossible_result_carries_a_verifying_certificate(self) -> None:
        bigraph = _corank_two_psd()

        result = bigraph.solidify()

        assert result.status is SolidificationStatus.IMPOSSIBLE
        assert result.certificate is not None
        assert result.certificate.verify(bigraph)
        assert result.verify()
        assert result.final is None
        assert len(result.sequence) == 0

    def test_certificate_rejects_a_different_bigraph(self) -> None:
        result = _corank_two_psd().solidify()
        assert result.certificate is not None

        assert not result.certificate.verify(_dotted_a2())

    def test_malformed_result_shapes_are_rejected(self) -> None:
        bigraph = _dotted_a2()
        certificate = NoSolidCertificate(
            ("a", "b"),
            Inertia(1, 0, 2),
            "example",
        )

        with pytest.raises(ValueError, match="final bigraph"):
            SolidificationResult(input=bigraph, status=SolidificationStatus.SOLIDIFIED)
        with pytest.raises(ValueError, match="certificate"):
            SolidificationResult(input=bigraph, status=SolidificationStatus.IMPOSSIBLE)
        with pytest.raises(ValueError, match="cannot carry a final"):
            SolidificationResult(
                input=bigraph,
                status=SolidificationStatus.INCONCLUSIVE,
                final=bigraph,
            )
        with pytest.raises(ValueError, match="cannot carry a sequence"):
            SolidificationResult(
                input=bigraph,
                status=SolidificationStatus.IMPOSSIBLE,
                certificate=certificate,
                sequence=FlationSequence((bigraph.solidify().sequence.steps[0],)),
            )
        with pytest.raises(ValueError, match="frame_classes"):
            SolidificationResult(
                input=bigraph,
                status=SolidificationStatus.INCONCLUSIVE,
                explored_states=1,
                frame_classes=2,
            )

    def test_invalid_certificates_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="corank"):
            NoSolidCertificate(("a",), Inertia(1, 0, 0), "example")
        with pytest.raises(ValueError, match="unique"):
            NoSolidCertificate(("a", "a"), Inertia(1, 0, 2), "example")
        with pytest.raises(ValueError, match="reason"):
            NoSolidCertificate(("a", "b"), Inertia(1, 0, 2), "")


class TestSearch:
    """Verify the exact search behavior and its caps."""

    def test_affine_cycle_solidifies(self) -> None:
        bigraph = _bigraph(
            ("a", "b", "c"),
            ((2, -1, -2), (-1, 2, 1), (-2, 1, 2)),
        )

        result = bigraph.solidify()

        assert result.status is SolidificationStatus.SOLIDIFIED
        assert result.verify()
        assert len(result.sequence) == 1

    def test_positive_definite_component_uses_the_greedy_loop(self) -> None:
        bigraph = _bigraph(
            ("a", "b", "c"),
            ((2, 1, 0), (1, 2, -1), (0, -1, 2)),
        )

        result = bigraph.solidify()

        assert result.status is SolidificationStatus.SOLIDIFIED
        assert result.verify()
        assert len(result.sequence) == 1

    def test_sign_inversion_is_available(self) -> None:
        bigraph = _bigraph(
            ("a", "b", "c", "d"),
            (
                (2, 1, 0, -1),
                (1, 2, -1, 0),
                (0, -1, 2, 1),
                (-1, 0, 1, 2),
            ),
        )

        result = bigraph.solidify()

        assert result.status is SolidificationStatus.SOLIDIFIED
        assert result.verify()
        assert any(step.source == step.target for step in result.sequence.steps)

    def test_indefinite_candidate_is_inconclusive_at_a_small_cap(self) -> None:
        bigraph = _indefinite_candidate()

        result = bigraph.solidify(max_states=25)

        assert result.status is SolidificationStatus.INCONCLUSIVE
        assert result.certificate is None
        assert result.final is None
        assert not result.verify()
        assert result.max_states == 25
        assert result.max_entry == 1_000_000
        assert result.explored_states == 25
        assert result.frame_classes <= result.explored_states
        assert result.reason == "state cap"

    def test_entry_cap_is_reported(self) -> None:
        huge = 2**80
        bigraph = _bigraph(("a", "b"), ((2, huge), (huge, 2)))

        result = bigraph.solidify(max_entry=1)

        assert result.status is SolidificationStatus.INCONCLUSIVE
        assert result.reason == "entry cap"

    def test_large_intermediate_entries_stay_exact(self) -> None:
        huge = 2**80
        bigraph = _bigraph(("a", "b"), ((2, huge), (huge, 2)))

        result = bigraph.solidify(max_entry=10**100)

        assert result.status is SolidificationStatus.SOLIDIFIED
        assert result.final is not None
        assert result.final.matrix.entries[0][1] == -huge
        assert result.verify()

    def test_large_component_uses_frame_bucket_fallback(self) -> None:
        size = 7
        entries = [
            [
                2 if row == column else (-1 if abs(row - column) == 1 else 0)
                for column in range(size)
            ]
            for row in range(size)
        ]
        entries[0][1] = 1
        entries[1][0] = 1
        bigraph = _bigraph(tuple(str(index) for index in range(size)), tuple(map(tuple, entries)))

        result = bigraph.solidify()

        assert result.status is SolidificationStatus.SOLIDIFIED
        assert result.verify()
        assert result.frame_classes >= 1
        assert result.frame_classes <= result.explored_states

    def test_invalid_caps_are_rejected(self) -> None:
        bigraph = _dotted_a2()

        for value in (0, -1, True):
            with pytest.raises(ValueError, match="max_states"):
                bigraph.solidify(max_states=value)  # type: ignore[arg-type]
            with pytest.raises(ValueError, match="max_entry"):
                bigraph.solidify(max_entry=value)  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="max_states"):
            bigraph.solidify(max_states=1.5)  # type: ignore[arg-type]

    def test_non_symmetric_bigraph_is_rejected(self) -> None:
        bigraph = Bigraph(QuasiCartanMatrix(((2, -1), (-2, 2))), ("a", "b"))

        with pytest.raises(ValueError, match="symmetric"):
            bigraph.solidify()


class TestComposition:
    """Verify disconnected composition and permutation behavior."""

    def test_two_dotted_a2_components_compose(self) -> None:
        bigraph = _bigraph(
            ("a", "b", "c", "d"),
            (
                (2, 1, 0, 0),
                (1, 2, 0, 0),
                (0, 0, 2, 1),
                (0, 0, 1, 2),
            ),
        )

        result = bigraph.solidify()

        assert result.status is SolidificationStatus.SOLIDIFIED
        assert len(result.sequence) == 2
        assert result.verify()

    def test_impossible_component_dominates(self) -> None:
        bigraph = _bigraph(
            ("a", "b", "c", "d", "e"),
            (
                (2, 1, 0, 0, 0),
                (1, 2, 0, 0, 0),
                (0, 0, 2, 2, 2),
                (0, 0, 2, 2, 2),
                (0, 0, 2, 2, 2),
            ),
        )

        result = bigraph.solidify()

        assert result.status is SolidificationStatus.IMPOSSIBLE
        assert result.certificate is not None
        assert result.certificate.labels == ("c", "d", "e")
        assert result.verify()

    def test_inconclusive_component_propagates(self) -> None:
        candidate = _indefinite_candidate()
        size = candidate.dimension
        dotted = ((2, 1), (1, 2))
        entries = tuple(
            tuple(
                dotted[row][column]
                if row < 2 and column < 2
                else candidate.matrix.entries[row - 2][column - 2]
                if row >= 2 and column >= 2
                else 0
                for column in range(size + 2)
            )
            for row in range(size + 2)
        )
        bigraph = _bigraph(
            ("a", "b", "c", "d", "e", "f"),
            entries,
        )

        result = bigraph.solidify(max_states=25)

        assert result.status is SolidificationStatus.INCONCLUSIVE
        assert result.certificate is None
        assert len(result.sequence) == 0
        assert not result.verify()

    def test_relabelling_preserves_the_witness_shape(self) -> None:
        first = _dotted_a2()
        second = _bigraph(("b", "a"), ((2, 1), (1, 2)))

        first_result = first.solidify()
        second_result = second.solidify()

        assert first_result.status is second_result.status
        assert first_result.explored_states == second_result.explored_states
        assert first_result.frame_classes == second_result.frame_classes
        assert first_result.verify()
        assert second_result.verify()

    def test_repeated_runs_are_deterministic(self) -> None:
        bigraph = _bigraph(
            ("a", "b", "c"),
            ((2, -1, -2), (-1, 2, 1), (-2, 1, 2)),
        )

        first = bigraph.solidify()
        second = bigraph.solidify()

        assert first.status is second.status
        assert first.sequence == second.sequence
        assert first.explored_states == second.explored_states
        assert first.frame_classes == second.frame_classes
