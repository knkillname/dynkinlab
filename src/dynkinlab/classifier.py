# SPDX-FileCopyrightText: 2026-present Mario Abarca <knkillname@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Paper-faithful cubic Dynkin classifier."""

from collections.abc import Hashable, Iterable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from dynkinlab.bigraph import Bigraph
from dynkinlab.dynkin import (
    ClassificationStatus,
    ComponentClassification,
    DynkinClassification,
    DynkinType,
)
from dynkinlab.flation import FlationSequence, FlationStep
from dynkinlab.inertia import Inertia
from dynkinlab.quasi_cartan import QuasiCartanMatrix
from dynkinlab.symmetrizer import Symmetrizer


@dataclass(frozen=True)
class _ComponentOutcome:
    """Internal outcome for one connected component."""

    status: ClassificationStatus
    reason: str
    dynkin_type: DynkinType | None
    steps: tuple[FlationStep, ...]
    bound: int
    normal_form: tuple[tuple[int, ...], ...] | None


class _SymmetricState:
    """Mutable symmetric state with an O(n) dotted-edge index."""

    def __init__(self, entries: npt.NDArray[np.int64]) -> None:
        self.entries = np.array(entries, dtype=np.int64, copy=True)
        self.size = self.entries.shape[0]
        self.dotted: set[tuple[int, int]] = set()
        self._rebuild_dotted()

    def _rebuild_dotted(self) -> None:
        """Rebuild the dotted-edge set from the current entries."""
        mask = (self.entries > 0) & (self.entries.T > 0)
        self.dotted = {(row, column) for row, column in np.argwhere(np.triu(mask, k=1)).tolist()}

    def next_dotted(self) -> tuple[int, int] | None:
        """Return one dotted edge, or None when the matrix is solid."""
        if not self.dotted:
            return None
        return next(iter(self.dotted))

    def flate(self, source: int, target: int) -> None:
        """Apply a flation and update the dotted-edge index in O(n)."""
        self.entries = QuasiCartanMatrix.flate_entries(
            self.entries,
            source,
            target,
            np.int64,
        )
        for index in range(self.size):
            if index != target:
                self.dotted.discard((min(target, index), max(target, index)))
        for index in range(self.size):
            if index == target:
                continue
            if self.entries[target, index] > 0 and self.entries[index, target] > 0:
                self.dotted.add((min(target, index), max(target, index)))


class DynkinClassifier:
    """Classify positive definite quasi-Cartan matrices.

    The implementation follows the paper's Stage 1 filter/preprocessing, Stage
    2 bounded inflations, and Stage 3 graph classification. The class is a
    namespace for the algorithm; use ``QuasiCartanMatrix.classify`` or
    ``Bigraph.classify`` for the public entry points.
    """

    G2_BOUND = 24

    @classmethod
    def classify(
        cls,
        matrix: QuasiCartanMatrix,
        labels: Iterable[Hashable] | None = None,
    ) -> DynkinClassification:
        """Classify a quasi-Cartan matrix.

        Parameters
        ----------
        matrix : QuasiCartanMatrix
            Input matrix.
        labels : Iterable[Hashable] | None
            Optional labels for the retained bigraph and result components.

        Returns
        -------
        DynkinClassification
            Rich classification result.
        """
        bigraph = Bigraph(matrix, labels)
        inertia = Inertia.from_matrix(matrix)
        components = Symmetrizer.components(matrix.entries)
        all_steps: list[FlationStep] = []
        classified_components: list[ComponentClassification] = []
        total_bound = 0
        failure: _ComponentOutcome | None = None

        for component in components:
            submatrix = cls._submatrix(matrix, component)
            outcome = cls._classify_component(submatrix)
            total_bound += outcome.bound
            for step in outcome.steps:
                if not isinstance(step.source, int) or not isinstance(step.target, int):
                    raise ValueError("component steps must use integer indices")
                all_steps.append(
                    FlationStep(
                        bigraph.labels[component[step.source]],
                        bigraph.labels[component[step.target]],
                    ),
                )
            if outcome.status is ClassificationStatus.CLASSIFIED:
                if outcome.dynkin_type is None:
                    raise ValueError("classified component is missing its Dynkin type")
                classified_components.append(
                    ComponentClassification(
                        tuple(bigraph.labels[index] for index in component),
                        outcome.dynkin_type,
                    ),
                )
            else:
                failure = outcome
                break

        witness = FlationSequence(tuple(all_steps))
        if failure is None:
            normal_form = witness.apply(bigraph)
            return DynkinClassification(
                status=ClassificationStatus.CLASSIFIED,
                reason="",
                inertia=inertia,
                components=tuple(classified_components),
                witness=witness,
                normal_form=normal_form,
                steps=len(witness),
                bound=total_bound,
                input=bigraph,
            )
        return DynkinClassification(
            status=failure.status,
            reason=failure.reason,
            inertia=inertia,
            components=tuple(classified_components),
            witness=witness,
            normal_form=None,
            steps=len(witness),
            bound=total_bound,
            input=bigraph,
        )

    @classmethod
    def _classify_component(cls, matrix: QuasiCartanMatrix) -> _ComponentOutcome:
        """Run the three stages on one connected component."""
        entries = [list(row) for row in matrix.entries]
        size = matrix.dimension

        if size == 2:
            forward = entries[0][1]
            backward = entries[1][0]
            if forward * backward > 0 and {abs(forward), abs(backward)} == {1, 3}:
                if forward < 0:
                    return _ComponentOutcome(
                        ClassificationStatus.CLASSIFIED,
                        "",
                        DynkinType("G", 2),
                        (),
                        0,
                        matrix.entries,
                    )
                canonical = QuasiCartanMatrix.flate_entries(entries, 0, 0, object)
                return _ComponentOutcome(
                    ClassificationStatus.CLASSIFIED,
                    "",
                    DynkinType("G", 2),
                    (FlationStep(0, 0),),
                    0,
                    tuple(map(tuple, canonical.tolist())),
                )

        if size == 1:
            return _ComponentOutcome(
                ClassificationStatus.CLASSIFIED,
                "",
                DynkinType("A", 1),
                (),
                0,
                matrix.entries,
            )

        admissible, reason = cls._admissible(entries)
        if not admissible:
            return _ComponentOutcome(
                ClassificationStatus.NOT_ADMISSIBLE,
                reason,
                None,
                (),
                0,
                None,
            )

        inertia = Inertia.from_matrix(matrix)
        if inertia.negative > 0:
            return _ComponentOutcome(
                ClassificationStatus.NOT_POSITIVE_DEFINITE,
                "indefinite",
                None,
                (),
                0,
                None,
            )

        array_entries = np.array(entries, dtype=np.int64)
        symmetric = matrix.is_symmetric
        if symmetric:
            bound = cls._bound(size)
            final_entries, steps, exhausted = cls._run_symmetric(array_entries, bound)
            if exhausted:
                return cls._failure("bound exhausted", steps, bound)
            return cls._stage_three(final_entries, steps, bound)

        if size <= 4:
            bound = cls.G2_BOUND
            final_entries, steps, exhausted = cls._run_naive(array_entries, bound)
            if exhausted:
                return cls._failure("bound exhausted", steps, bound)
            return cls._stage_three(final_entries, steps, bound)

        distinguished = cls._distinguished(array_entries)
        if distinguished is None:
            return _ComponentOutcome(
                ClassificationStatus.NOT_POSITIVE_DEFINITE,
                "no distinguished vertex",
                None,
                (),
                0,
                None,
            )

        repaired, repair_steps = cls._make_solid_arched(array_entries, distinguished)
        bound = cls._bound(size - 1)
        final_entries, loop_steps, failure_reason = cls._run_distinguished(
            repaired,
            distinguished,
            bound,
        )
        steps = repair_steps + loop_steps
        if failure_reason is not None:
            return cls._failure(failure_reason, steps, bound)
        return cls._stage_three(final_entries, steps, bound)

    @classmethod
    def _admissible(cls, entries: list[list[int]]) -> tuple[bool, str]:
        """Return whether the component passes the paper's admissibility filter."""
        size = len(entries)
        for row in range(size):
            for column in range(row + 1, size):
                forward = entries[row][column]
                backward = entries[column][row]
                if abs(forward) >= 3 or abs(backward) >= 3:
                    return False, "off-diagonal magnitude is at least 3"
                if forward == backward and abs(forward) > 1:
                    return False, "parallel edges are not admissible"
                if abs(forward) > 1 and abs(backward) != 1:
                    return False, "parallel edges are not admissible"
                if abs(backward) > 1 and abs(forward) != 1:
                    return False, "parallel edges are not admissible"
                if (forward == 0) != (backward == 0):
                    return False, "matrix is not symmetrizable"
                if forward * backward < 0:
                    return False, "matrix is not symmetrizable"
        return True, ""

    @classmethod
    def _distinguished(cls, entries: npt.NDArray[np.int64]) -> int | None:
        """Return a source/sink that carries every arrow, if one exists."""
        arrows = cls._arrows(entries)
        if not arrows:
            return None
        size = entries.shape[0]
        for vertex in range(size):
            has_undirected = any(
                entries[vertex, other] == entries[other, vertex] != 0
                for other in range(size)
                if other != vertex
            )
            if has_undirected:
                continue
            if all(target == vertex for _, target, _, _ in arrows):
                return vertex
            if all(source == vertex for source, _, _, _ in arrows):
                return vertex
        return None

    @classmethod
    def _arrows(
        cls,
        entries: npt.NDArray[np.int64],
    ) -> list[tuple[int, int, int, bool]]:
        """Return arrows as (source, target, multiplicity, dotted)."""
        size = entries.shape[0]
        arrows: list[tuple[int, int, int, bool]] = []
        for row in range(size):
            for column in range(row + 1, size):
                forward = int(entries[row, column])
                backward = int(entries[column, row])
                if forward == backward:
                    continue
                if forward == 0 or backward == 0:
                    raise ValueError("matrix is not representable as a bigraph")
                if abs(forward) < abs(backward):
                    source, target = row, column
                    multiplicity = abs(backward)
                    sign = forward
                elif abs(forward) > abs(backward):
                    source, target = column, row
                    multiplicity = abs(forward)
                    sign = backward
                else:
                    raise ValueError("matrix is not representable as a bigraph")
                if sign == 0:
                    raise ValueError("matrix is not representable as a bigraph")
                arrows.append((source, target, multiplicity, sign > 0))
        return arrows

    @classmethod
    def _make_solid_arched(
        cls,
        entries: npt.NDArray[np.int64],
        distinguished: int,
    ) -> tuple[npt.NDArray[np.int64], tuple[FlationStep, ...]]:
        """Flip dotted arrows incident on the distinguished vertex."""
        current = np.array(entries, dtype=np.int64, copy=True)
        steps: list[FlationStep] = []
        for source, target, _, dotted in cls._arrows(current):
            if not dotted:
                continue
            if source == distinguished:
                other = target
            elif target == distinguished:
                other = source
            else:
                raise ValueError("distinguished vertex does not carry every arrow")
            current = QuasiCartanMatrix.flate_entries(current, other, other, np.int64)
            steps.append(FlationStep(other, other))
        if any(dotted for _, _, _, dotted in cls._arrows(current)):
            raise ValueError("could not make the bigraph solid-arched")
        return current, tuple(steps)

    @classmethod
    def _run_symmetric(
        cls,
        entries: npt.NDArray[np.int64],
        bound: int,
    ) -> tuple[npt.NDArray[np.int64], tuple[FlationStep, ...], bool]:
        """Run bounded inflations on a symmetric component."""
        state = _SymmetricState(entries)
        steps: list[FlationStep] = []
        while True:
            edge = state.next_dotted()
            if edge is None:
                return state.entries, tuple(steps), False
            if len(steps) >= bound:
                return state.entries, tuple(steps), True
            source, target = edge
            state.flate(source, target)
            steps.append(FlationStep(source, target))

    @classmethod
    def _run_distinguished(
        cls,
        entries: npt.NDArray[np.int64],
        distinguished: int,
        bound: int,
    ) -> tuple[npt.NDArray[np.int64], tuple[FlationStep, ...], str | None]:
        """Run non-symmetric Stage 2 while preserving solid-arched.

        Dotted edges are selected in ``G - x``. For each edge, both flation
        directions are considered; the first direction that keeps every arrow
        solid is used. Lemma 4.2 guarantees such a direction exists.
        """
        current = np.array(entries, dtype=np.int64, copy=True)
        others = [index for index in range(current.shape[0]) if index != distinguished]
        steps: list[FlationStep] = []
        while True:
            dotted = [
                (row, column)
                for row in others
                for column in others
                if row < column and current[row, column] > 0 and current[column, row] > 0
            ]
            if not dotted:
                return current, tuple(steps), None
            if len(steps) >= bound:
                return current, tuple(steps), "bound exhausted"

            chosen: tuple[int, int, npt.NDArray[np.int64]] | None = None
            for row, column in dotted:
                for source, target in ((row, column), (column, row)):
                    candidate = QuasiCartanMatrix.flate_entries(
                        current,
                        source,
                        target,
                        np.int64,
                    )
                    if cls._solid_arched(candidate):
                        chosen = (source, target, candidate)
                        break
                if chosen is not None:
                    break
            if chosen is None:
                return current, tuple(steps), "could not preserve solid-arched"

            source, target, current = chosen
            steps.append(FlationStep(source, target))

    @classmethod
    def _solid_arched(cls, entries: npt.NDArray[np.int64]) -> bool:
        """Return whether every arrow is solid."""
        return all(not dotted for _, _, _, dotted in cls._arrows(entries))

    @classmethod
    def _run_naive(
        cls,
        entries: npt.NDArray[np.int64],
        bound: int,
    ) -> tuple[npt.NDArray[np.int64], tuple[FlationStep, ...], bool]:
        """Run the non-symmetric n <= 4 bounded inflation loop."""
        current = np.array(entries, dtype=np.int64, copy=True)
        steps: list[FlationStep] = []
        while True:
            pair = cls._first_positive(current)
            if pair is None:
                return current, tuple(steps), False
            if len(steps) >= bound:
                return current, tuple(steps), True
            source, target = pair
            current = QuasiCartanMatrix.flate_entries(current, source, target, np.int64)
            steps.append(FlationStep(source, target))

    @staticmethod
    def _first_positive(entries: npt.NDArray[np.int64]) -> tuple[int, int] | None:
        """Return the first positive off-diagonal pair."""
        for row, column in np.argwhere(entries > 0).tolist():
            if row != column:
                return row, column
        return None

    @classmethod
    def _stage_three(
        cls,
        entries: npt.NDArray[np.int64],
        steps: tuple[FlationStep, ...],
        bound: int,
    ) -> _ComponentOutcome:
        """Classify the final solid normal form."""
        try:
            matrix = QuasiCartanMatrix(tuple(map(tuple, entries.tolist())))
        except ValueError:
            return cls._failure("normal form is not a Dynkin diagram", steps, bound)
        dynkin_type = DynkinType.from_matrix(matrix)
        if dynkin_type is None:
            return cls._failure("normal form is not a Dynkin diagram", steps, bound)
        return _ComponentOutcome(
            ClassificationStatus.CLASSIFIED,
            "",
            dynkin_type,
            steps,
            bound,
            matrix.entries,
        )

    @staticmethod
    def _failure(reason: str, steps: tuple[FlationStep, ...], bound: int) -> _ComponentOutcome:
        """Return a non-positive-definite outcome."""
        return _ComponentOutcome(
            ClassificationStatus.NOT_POSITIVE_DEFINITE,
            reason,
            None,
            steps,
            bound,
            None,
        )

    @staticmethod
    def _submatrix(matrix: QuasiCartanMatrix, indices: tuple[int, ...]) -> QuasiCartanMatrix:
        """Extract a principal submatrix."""
        return QuasiCartanMatrix(
            tuple(tuple(matrix.entries[row][column] for column in indices) for row in indices),
        )

    @staticmethod
    def _bound(size: int) -> int:
        """Return the paper's symmetric inflation bound."""
        if size < 4:
            return size * (size + 1) // 2
        if 6 <= size <= 8:
            return size * (15 * size - 168) + 504
        return size * (size - 1)
