# SPDX-FileCopyrightText: 2026-present Mario Abarca <knkillname@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Display options for notebook and Markdown rendering."""

from dataclasses import dataclass
from typing import override


@dataclass
class DisplayOptions:
    """Mutable display settings for DynkinLab value objects.

    Attributes
    ----------
    expanded_form_dimension_limit : int
        Largest form dimension for which the expanded polynomial is shown.
    matrix_index_base : int
        First index used for matrix row and column headers.
    """

    expanded_form_dimension_limit: int = 3
    matrix_index_base: int = 0

    @override
    def __setattr__(self, name: str, value: object) -> None:
        """Validate display settings when they are assigned."""
        if name in {"expanded_form_dimension_limit", "matrix_index_base"}:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        super().__setattr__(name, value)

    def reset(self) -> None:
        """Restore the default display settings."""
        defaults = type(self)()
        self.expanded_form_dimension_limit = defaults.expanded_form_dimension_limit
        self.matrix_index_base = defaults.matrix_index_base


options = DisplayOptions()
