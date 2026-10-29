# SPDX-FileCopyrightText: 2026-present Mario Abarca <knkillname@gmail.com>
#
# SPDX-License-Identifier: MIT
"""First-class flation steps and sequences."""

from collections.abc import Hashable, Iterable
from dataclasses import dataclass, field

from dynkinlab.bigraph import Bigraph


@dataclass(frozen=True)
class FlationStep:
    """A single flation identified by source and target labels.

    Equal source and target labels represent a sign inversion.

    Parameters
    ----------
    source : Hashable
        Source vertex label.
    target : Hashable
        Target vertex label.

    Raises
    ------
    ValueError
        If either label is unhashable.
    """

    source: Hashable
    target: Hashable

    def __post_init__(self) -> None:
        """Validate the labels."""
        for name, label in (("source", self.source), ("target", self.target)):
            try:
                hash(label)
            except TypeError as error:
                raise ValueError(f"{name} must be hashable") from error

    def apply(self, bigraph: Bigraph) -> Bigraph:
        """Apply this step to a bigraph.

        Parameters
        ----------
        bigraph : Bigraph
            The input bigraph.

        Returns
        -------
        Bigraph
            The flated bigraph.
        """
        return bigraph.flation(self.source, self.target)


@dataclass(frozen=True)
class FlationSequence:
    """An immutable sequence of flation steps.

    Parameters
    ----------
    steps : Iterable[FlationStep]
        Steps in application order.

    Raises
    ------
    ValueError
        If an item is not a `FlationStep`.
    """

    steps: tuple[FlationStep, ...] = field(default_factory=tuple)

    def __init__(self, steps: Iterable[FlationStep] = ()) -> None:
        try:
            normalized = tuple(steps)
        except TypeError as error:
            raise ValueError("steps must be an iterable") from error
        for step in normalized:
            if not isinstance(step, FlationStep):
                raise ValueError("steps must contain FlationStep values")
        object.__setattr__(self, "steps", normalized)

    def __len__(self) -> int:
        """Return the number of steps."""
        return len(self.steps)

    def apply(self, bigraph: Bigraph) -> Bigraph:
        """Replay the sequence on a bigraph.

        Parameters
        ----------
        bigraph : Bigraph
            The input bigraph.

        Returns
        -------
        Bigraph
            The final bigraph.
        """
        result = bigraph
        for step in self.steps:
            result = step.apply(result)
        return result

    def inverse(self) -> "FlationSequence":
        """Return the inverse sequence.

        Flation is an involution, so reversing the steps recovers the input.
        """
        return FlationSequence(tuple(reversed(self.steps)))

    def verify(self, start: Bigraph, final: Bigraph) -> bool:
        """Return whether replaying the sequence maps ``start`` to ``final``.

        Parameters
        ----------
        start : Bigraph
            Initial bigraph.
        final : Bigraph
            Claimed final bigraph.

        Returns
        -------
        bool
            Whether the replay equals the claimed final value.
        """
        return self.apply(start) == final
