# SPDX-FileCopyrightText: 2026-present Mario Abarca <knkillname@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Display options for notebook and Markdown rendering."""

from dataclasses import dataclass
from typing import override

_FIGURE_SIZE_LENGTH = 2


@dataclass
class DisplayOptions:
    """Mutable display settings for DynkinLab value objects.

    Attributes
    ----------
    expanded_form_dimension_limit : int
        Largest form dimension for which the expanded polynomial is shown.
    matrix_index_base : int
        First index used for matrix row and column headers.
    bigraph_layout_seed : int
        Seed for deterministic bigraph layout.
    bigraph_figure_size : tuple[float, float]
        Width and height of the rendered bigraph.
    bigraph_show_labels : bool
        Whether vertex labels are drawn.
    bigraph_show_multiplicity : bool
        Whether edge and arrow multiplicities are shown.
    bigraph_arrow_style : str
        ``"arrow"`` for arrowheads or ``"line"`` for directionless lines.
    """

    expanded_form_dimension_limit: int = 3
    matrix_index_base: int = 0
    bigraph_layout_seed: int = 0
    bigraph_figure_size: tuple[float, float] = (4.0, 4.0)
    bigraph_show_labels: bool = True
    bigraph_show_multiplicity: bool = True
    bigraph_arrow_style: str = "arrow"

    @override
    def __setattr__(self, name: str, value: object) -> None:
        """Validate display settings when they are assigned."""
        if name in {
            "expanded_form_dimension_limit",
            "matrix_index_base",
            "bigraph_layout_seed",
        }:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        elif name == "bigraph_figure_size":
            if not isinstance(value, (tuple, list)) or len(value) != _FIGURE_SIZE_LENGTH:
                raise ValueError(f"{name} must be a pair of positive numbers")
            width, height = value
            if any(
                isinstance(item, bool) or not isinstance(item, (int, float))
                for item in (width, height)
            ):
                raise ValueError(f"{name} must be a pair of positive numbers")
            if width <= 0 or height <= 0:
                raise ValueError(f"{name} must be a pair of positive numbers")
            value = (float(width), float(height))
        elif name in {"bigraph_show_labels", "bigraph_show_multiplicity"}:
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean")
        elif name == "bigraph_arrow_style":
            if value not in {"arrow", "line"}:
                raise ValueError(f"{name} must be 'arrow' or 'line'")
        super().__setattr__(name, value)

    def reset(self) -> None:
        """Restore the default display settings."""
        defaults = type(self)()
        for name in (
            "expanded_form_dimension_limit",
            "matrix_index_base",
            "bigraph_layout_seed",
            "bigraph_figure_size",
            "bigraph_show_labels",
            "bigraph_show_multiplicity",
            "bigraph_arrow_style",
        ):
            setattr(self, name, getattr(defaults, name))


options = DisplayOptions()
