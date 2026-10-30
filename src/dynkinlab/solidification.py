# SPDX-FileCopyrightText: 2026-present Mario Abarca <knkillname@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Bounded solidification search and its result values."""

from collections import deque
from collections.abc import Hashable, Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import permutations
from typing import Any

import networkx as nx

from dynkinlab.bigraph import Bigraph
from dynkinlab.flation import FlationSequence, FlationStep
from dynkinlab.inertia import Inertia
from dynkinlab.quasi_cartan import QuasiCartanMatrix
from dynkinlab.symmetrizer import Symmetrizer

DEFAULT_MAX_STATES = 100_000
DEFAULT_MAX_ENTRY = 1_000_000
_PERMUTATION_LIMIT = 6
_CERTIFICATE_CORANK = 2


class SolidificationStatus(StrEnum):
    """Outcome of a solidification attempt."""

    SOLIDIFIED = "solidified"
    IMPOSSIBLE = "impossible"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class NoSolidCertificate:
    """A proof-backed reason why a connected component has no solid representative.

    Parameters
    ----------
    labels : tuple[Hashable, ...]
        Labels of the offending connected component.
    inertia : Inertia
        Exact inertia of the component.
    reason : str
        Human-readable description of the obstruction.

    Raises
    ------
    ValueError
        If the labels are empty, duplicated, or unhashable, if the inertia is
        not positive semidefinite with corank at least two, or if the reason is
        empty.
    """

    labels: tuple[Hashable, ...]
    inertia: Inertia
    reason: str

    def __post_init__(self) -> None:
        """Normalize and validate the certificate."""
        try:
            labels = tuple(self.labels)
        except TypeError as error:
            raise ValueError("labels must be an iterable") from error
        if not labels:
            raise ValueError("labels must not be empty")
        for label in labels:
            try:
                hash(label)
            except TypeError as error:
                raise ValueError("labels must be hashable") from error
        if len(set(labels)) != len(labels):
            raise ValueError("labels must be unique")
        if not isinstance(self.inertia, Inertia):
            raise ValueError("inertia must be an Inertia value")
        if self.inertia.negative != 0 or self.inertia.zero < _CERTIFICATE_CORANK:
            raise ValueError("certificate requires positive semidefinite corank at least two")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string")
        object.__setattr__(self, "labels", labels)

    def verify(self, bigraph: Bigraph) -> bool:
        """Return whether this certificate applies to ``bigraph``.

        Parameters
        ----------
        bigraph : Bigraph
            Input bigraph.

        Returns
        -------
        bool
            Whether the labelled component is connected, maximal, positive
            semidefinite, of corank at least two, and has the recorded inertia.
        """
        try:
            indices = tuple(bigraph.index(label) for label in self.labels)
        except ValueError:
            return False
        entries = bigraph.matrix.entries
        if not _is_maximal_component(entries, indices):
            return False
        submatrix = _induced_entries(entries, indices)
        inertia = Inertia.from_matrix(QuasiCartanMatrix(submatrix))
        return (
            inertia == self.inertia
            and inertia.negative == 0
            and inertia.zero >= _CERTIFICATE_CORANK
        )

    def to_markdown(self) -> str:
        """Render the certificate as a Markdown summary."""
        reason = self.reason.replace("|", r"\|")
        labels = ", ".join(str(label).replace("|", r"\|") for label in self.labels)
        return "\n".join(
            (
                "**No solid representative**",
                "",
                f"Reason: {reason}",
                f"Component: {labels}",
                "",
                "| positive | negative | zero |",
                "|---:|---:|---:|",
                f"| {self.inertia.positive} | {self.inertia.negative} | {self.inertia.zero} |",
            ),
        )

    def _repr_markdown_(self) -> str:
        """Return the canonical Markdown representation for notebook display."""
        return self.to_markdown()


@dataclass(frozen=True)
class SolidificationResult:
    """Immutable outcome of a solidification attempt.

    Parameters
    ----------
    input : Bigraph
        The original bigraph.
    status : SolidificationStatus
        Whether the input solidified, is impossible, or the search was inconclusive.
    sequence : FlationSequence
        Witness sequence for a solidified result.
    final : Bigraph | None
        Final solid bigraph for a solidified result.
    certificate : NoSolidCertificate | None
        Impossibility certificate for an impossible result.
    explored_states : int
        Number of exact states accepted by the search.
    frame_classes : int
        Number of frame-isomorphism classes among those states.
    max_states : int
        Applied exact-state cap.
    max_entry : int
        Applied absolute-entry cap.
    reason : str
        Explanation for an impossible or inconclusive result.

    Raises
    ------
    ValueError
        If the status-specific fields are inconsistent or the counters are invalid.
    """

    input: Bigraph
    status: SolidificationStatus
    sequence: FlationSequence = field(default_factory=FlationSequence)
    final: Bigraph | None = None
    certificate: NoSolidCertificate | None = None
    explored_states: int = 0
    frame_classes: int = 0
    max_states: int = 0
    max_entry: int = 0
    reason: str = ""

    def __post_init__(self) -> None:
        """Validate the status-specific result shape."""
        self._validate_basics()
        if self.status is SolidificationStatus.SOLIDIFIED:
            self._validate_solidified()
        elif self.status is SolidificationStatus.IMPOSSIBLE:
            self._validate_impossible()
        else:
            self._validate_inconclusive()

    def _validate_basics(self) -> None:
        """Validate fields shared by every status."""
        if not isinstance(self.input, Bigraph):
            raise ValueError("input must be a Bigraph")
        if not isinstance(self.status, SolidificationStatus):
            raise ValueError("status must be a SolidificationStatus")
        if not isinstance(self.sequence, FlationSequence):
            raise ValueError("sequence must be a FlationSequence")
        for name, value in (
            ("explored_states", self.explored_states),
            ("frame_classes", self.frame_classes),
            ("max_states", self.max_states),
            ("max_entry", self.max_entry),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.frame_classes > self.explored_states:
            raise ValueError("frame_classes must not exceed explored_states")
        if not isinstance(self.reason, str):
            raise ValueError("reason must be a string")

    def _validate_solidified(self) -> None:
        """Validate a solidified result."""
        if self.final is None:
            raise ValueError("solidified results require a final bigraph")
        if self.certificate is not None:
            raise ValueError("solidified results cannot carry a certificate")
        if self.final.labels != self.input.labels:
            raise ValueError("final labels must match the input labels")

    def _validate_impossible(self) -> None:
        """Validate an impossible result."""
        if self.certificate is None:
            raise ValueError("impossible results require a certificate")
        if self.final is not None:
            raise ValueError("impossible results cannot carry a final bigraph")
        if len(self.sequence) != 0:
            raise ValueError("impossible results cannot carry a sequence")

    def _validate_inconclusive(self) -> None:
        """Validate an inconclusive result."""
        if self.certificate is not None:
            raise ValueError("inconclusive results cannot carry a certificate")
        if self.final is not None:
            raise ValueError("inconclusive results cannot carry a final bigraph")
        if len(self.sequence) != 0:
            raise ValueError("inconclusive results cannot carry a sequence")

    def verify(self) -> bool:
        """Return whether the result carries a valid witness or certificate.

        Returns
        -------
        bool
            True for a replayable witness to a solid bigraph or a verifiable
            impossibility certificate. Inconclusive results always return
            ``False``.
        """
        if self.status is SolidificationStatus.SOLIDIFIED:
            return (
                self.final is not None
                and self.sequence.verify(self.input, self.final)
                and _is_solid(self.final.matrix.entries)
            )
        if self.status is SolidificationStatus.IMPOSSIBLE:
            return self.certificate is not None and self.certificate.verify(self.input)
        return False

    def to_markdown(self) -> str:
        """Render the result as a Markdown summary."""
        lines = [
            "**Solidification result**",
            "",
            "| status | steps | exact states | frame classes | state cap | entry cap |",
            "|---|---:|---:|---:|---:|---:|",
            f"| {self.status.value} | {len(self.sequence)} | {self.explored_states} | "
            f"{self.frame_classes} | {self.max_states} | {self.max_entry} |",
        ]
        if self.reason:
            lines.extend(("", f"Reason: {self.reason}"))
        if self.certificate is not None:
            lines.extend(("", self.certificate.to_markdown()))
        return "\n".join(lines)

    def _repr_markdown_(self) -> str:
        """Return the canonical Markdown representation for notebook display."""
        return self.to_markdown()


@dataclass(frozen=True)
class _SearchOutcome:
    """Internal search result for one connected component."""

    steps: tuple[tuple[int, int], ...] | None
    final: tuple[tuple[int, ...], ...] | None
    explored_states: int
    frame_classes: int
    reason: str


class _FrameTracker:
    """Count frame-isomorphism classes for search states."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.count = 0
        self._keys: set[tuple[int, ...]] = set()
        self._buckets: dict[str, list[nx.Graph[Hashable]]] = {}

    def add(self, entries: tuple[tuple[int, ...], ...]) -> None:
        """Register one state's frame, incrementing the class count when new."""
        if self.size <= _PERMUTATION_LIMIT:
            key = _frame_key(entries)
            if key not in self._keys:
                self._keys.add(key)
                self.count += 1
            return
        graph = _frame_graph(entries)
        graph_hash = nx.weisfeiler_lehman_graph_hash(
            graph,
            edge_attr="weight",
            iterations=min(self.size, 4),
        )
        bucket = self._buckets.setdefault(graph_hash, [])
        for representative in bucket:
            if nx.is_isomorphic(
                graph,
                representative,
                edge_match=_same_frame_edge,
            ):
                return
        bucket.append(graph)
        self.count += 1


def solidify_bigraph(
    bigraph: Bigraph,
    *,
    max_states: int = DEFAULT_MAX_STATES,
    max_entry: int = DEFAULT_MAX_ENTRY,
) -> SolidificationResult:
    """Search for a flation sequence to a solid bigraph or certify impossibility.

    Parameters
    ----------
    bigraph : Bigraph
        Input bigraph with a symmetric underlying matrix.
    max_states : int
        Maximum number of exact states accepted by the search.
    max_entry : int
        Maximum absolute intermediate entry accepted by the search.

    Returns
    -------
    SolidificationResult
        A verified sequence, an impossibility certificate, or an explicit
        inconclusive outcome.

    Raises
    ------
    ValueError
        If the matrix is not symmetric or a cap is not a positive integer.
    """
    _validate_caps(max_states, max_entry)
    if not bigraph.matrix.is_symmetric:
        raise ValueError("solidification requires a symmetric bigraph")

    entries = bigraph.matrix.entries
    components = Symmetrizer.components(entries)
    all_steps: list[FlationStep] = []
    explored_states = 0
    frame_classes = 0
    inconclusive_reason = ""

    for component in components:
        submatrix = _induced_entries(entries, component)
        if _is_solid(submatrix):
            continue

        inertia = Inertia.from_matrix(QuasiCartanMatrix(submatrix))
        if inertia.negative == 0 and inertia.zero >= _CERTIFICATE_CORANK:
            certificate = NoSolidCertificate(
                tuple(bigraph.labels[index] for index in component),
                inertia,
                "connected positive semidefinite component has corank at least two",
            )
            return SolidificationResult(
                input=bigraph,
                status=SolidificationStatus.IMPOSSIBLE,
                certificate=certificate,
                explored_states=explored_states,
                frame_classes=frame_classes,
                max_states=max_states,
                max_entry=max_entry,
                reason=certificate.reason,
            )

        if inertia.is_positive_definite:
            outcome = _search_positive_definite(submatrix, max_states, max_entry)
        else:
            outcome = _search_bounded(submatrix, max_states, max_entry)

        explored_states += outcome.explored_states
        frame_classes += outcome.frame_classes
        if outcome.final is None or outcome.steps is None:
            if not inconclusive_reason:
                inconclusive_reason = outcome.reason
            continue

        for source, target in outcome.steps:
            all_steps.append(
                FlationStep(
                    bigraph.labels[component[source]],
                    bigraph.labels[component[target]],
                ),
            )

    if inconclusive_reason:
        return SolidificationResult(
            input=bigraph,
            status=SolidificationStatus.INCONCLUSIVE,
            explored_states=explored_states,
            frame_classes=frame_classes,
            max_states=max_states,
            max_entry=max_entry,
            reason=inconclusive_reason,
        )

    sequence = FlationSequence(all_steps)
    final = sequence.apply(bigraph)
    return SolidificationResult(
        input=bigraph,
        status=SolidificationStatus.SOLIDIFIED,
        sequence=sequence,
        final=final,
        explored_states=explored_states,
        frame_classes=frame_classes,
        max_states=max_states,
        max_entry=max_entry,
    )


def _validate_caps(max_states: int, max_entry: int) -> None:
    """Validate the two positive integer search caps."""
    for name, value in (("max_states", max_states), ("max_entry", max_entry)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")


def _search_positive_definite(
    entries: tuple[tuple[int, ...], ...],
    max_states: int,
    max_entry: int,
) -> _SearchOutcome:
    """Run the terminating dotted-edge loop on a positive definite component."""
    current = entries
    steps: list[tuple[int, int]] = []
    tracker = _FrameTracker(len(entries))
    tracker.add(current)
    explored = 1
    seen: set[Hashable] = {_state_key(current)}

    while not _is_solid(current):
        if explored >= max_states:
            return _SearchOutcome(None, None, explored, tracker.count, "state cap")
        edge = _first_dotted(current)
        if edge is None:
            return _SearchOutcome(None, None, explored, tracker.count, "orbit exhausted")
        source, target = edge
        candidate = _flate_entries(current, source, target)
        if _max_abs(candidate) > max_entry:
            return _SearchOutcome(None, None, explored, tracker.count, "entry cap")
        key = _state_key(candidate)
        if key in seen:
            return _SearchOutcome(None, None, explored, tracker.count, "cycle detected")
        seen.add(key)
        current = candidate
        steps.append((source, target))
        explored += 1
        tracker.add(current)

    return _SearchOutcome(tuple(steps), current, explored, tracker.count, "")


def _search_bounded(
    entries: tuple[tuple[int, ...], ...],
    max_states: int,
    max_entry: int,
) -> _SearchOutcome:
    """Run a deterministic breadth-first search over exact component states."""
    tracker = _FrameTracker(len(entries))
    tracker.add(entries)
    explored = 1
    seen: set[Hashable] = {_state_key(entries)}
    queue: deque[tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, int], ...]]] = deque(
        [(entries, ())],
    )
    entry_pruned = False

    while queue:
        current, path = queue.popleft()
        if _is_solid(current):
            return _SearchOutcome(path, current, explored, tracker.count, "")
        if explored >= max_states:
            return _SearchOutcome(None, None, explored, tracker.count, "state cap")

        for source, target in _moves(current):
            candidate = _flate_entries(current, source, target)
            if _max_abs(candidate) > max_entry:
                entry_pruned = True
                continue
            key = _state_key(candidate)
            if key in seen:
                continue
            seen.add(key)
            explored += 1
            tracker.add(candidate)
            next_path = (*path, (source, target))
            if _is_solid(candidate):
                return _SearchOutcome(next_path, candidate, explored, tracker.count, "")
            queue.append((candidate, next_path))
            if explored >= max_states:
                return _SearchOutcome(None, None, explored, tracker.count, "state cap")

    reason = "entry cap" if entry_pruned else "orbit exhausted"
    return _SearchOutcome(None, None, explored, tracker.count, reason)


def _moves(entries: tuple[tuple[int, ...], ...]) -> Iterable[tuple[int, int]]:
    """Yield deterministic dotted flations followed by sign inversions."""
    size = len(entries)
    for target in range(size):
        for source in range(size):
            if source != target and entries[target][source] > 0:
                yield source, target
    for vertex in range(size):
        yield vertex, vertex


def _first_dotted(entries: tuple[tuple[int, ...], ...]) -> tuple[int, int] | None:
    """Return the first dotted edge in upper-triangular order."""
    size = len(entries)
    for row in range(size):
        for column in range(row + 1, size):
            if entries[row][column] > 0 and entries[column][row] > 0:
                return row, column
    return None


def _flate_entries(
    entries: tuple[tuple[int, ...], ...],
    source: int,
    target: int,
) -> tuple[tuple[int, ...], ...]:
    """Apply one exact flation to a tuple of entry rows."""
    size = len(entries)
    rows = [list(row) for row in entries]
    if source == target:
        for column in range(size):
            rows[target][column] = -rows[target][column]
            rows[column][target] = -rows[column][target]
        return tuple(tuple(row) for row in rows)

    coefficient = rows[target][source]
    result = [row[:] for row in rows]
    for column in range(size):
        result[target][column] = rows[target][column] - coefficient * rows[source][column]
    for row in range(size):
        result[row][target] = rows[row][target] - coefficient * rows[row][source]
    result[target][target] = 2
    return tuple(tuple(row) for row in result)


def _is_solid(entries: tuple[tuple[int, ...], ...]) -> bool:
    """Return whether every off-diagonal entry is non-positive."""
    size = len(entries)
    return all(
        entries[row][column] <= 0 for row in range(size) for column in range(size) if row != column
    )


def _max_abs(entries: tuple[tuple[int, ...], ...]) -> int:
    """Return the largest absolute entry."""
    return max(abs(value) for row in entries for value in row)


def _induced_entries(
    entries: tuple[tuple[int, ...], ...],
    indices: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    """Return the principal submatrix on ``indices``."""
    return tuple(tuple(entries[row][column] for column in indices) for row in indices)


def _is_maximal_component(
    entries: tuple[tuple[int, ...], ...],
    indices: tuple[int, ...],
) -> bool:
    """Return whether ``indices`` form a maximal connected component."""
    if not _is_connected(entries, indices):
        return False
    selected = set(indices)
    size = len(entries)
    for row in indices:
        for column in range(size):
            if column in selected:
                continue
            if entries[row][column] != 0 or entries[column][row] != 0:
                return False
    return True


def _is_connected(
    entries: tuple[tuple[int, ...], ...],
    indices: tuple[int, ...],
) -> bool:
    """Return whether the induced nonzero-edge graph on ``indices`` is connected."""
    if len(indices) <= 1:
        return True
    selected = set(indices)
    pending = [indices[0]]
    reached = {indices[0]}
    while pending:
        row = pending.pop()
        for column in indices:
            if column in reached or column not in selected:
                continue
            if entries[row][column] != 0 or entries[column][row] != 0:
                reached.add(column)
                pending.append(column)
    return len(reached) == len(indices)


def _state_key(entries: tuple[tuple[int, ...], ...]) -> Hashable:
    """Return a permutation-canonical key for small states, exact otherwise."""
    if len(entries) > _PERMUTATION_LIMIT:
        return entries
    return _permutation_key(entries)


def _permutation_key(entries: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    """Return the lexicographically least entry matrix under vertex permutations."""
    size = len(entries)
    best: tuple[int, ...] | None = None
    for order in permutations(range(size)):
        key = tuple(
            entries[order[row]][order[column]] for row in range(size) for column in range(size)
        )
        if best is None or key < best:
            best = key
    if best is None:
        raise ValueError("matrix must not be empty")
    return best


def _frame_key(entries: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    """Return the lexicographically least absolute frame under vertex permutations."""
    size = len(entries)
    best: tuple[int, ...] | None = None
    for order in permutations(range(size)):
        key = tuple(
            abs(entries[order[row]][order[column]])
            for row in range(size)
            for column in range(row + 1, size)
        )
        if best is None or key < best:
            best = key
    if best is None:
        raise ValueError("matrix must not be empty")
    return best


def _frame_graph(entries: tuple[tuple[int, ...], ...]) -> "nx.Graph[Hashable]":
    """Return the weighted frame graph of an entry matrix."""
    graph: nx.Graph[Hashable] = nx.Graph()
    graph.add_nodes_from(range(len(entries)))
    for row in range(len(entries)):
        for column in range(row + 1, len(entries)):
            weight = abs(entries[row][column])
            if weight:
                graph.add_edge(row, column, weight=weight)
    return graph


def _same_frame_edge(first: dict[str, Any], second: dict[str, Any]) -> bool:
    """Compare frame edge weights during isomorphism checks."""
    return bool(first["weight"] == second["weight"])
