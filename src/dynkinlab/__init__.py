# SPDX-FileCopyrightText: 2026-present Mario Abarca <knkillname@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Public package surface for DynkinLab."""

from dynkinlab import __about__
from dynkinlab import display as display
from dynkinlab.inertia import Inertia
from dynkinlab.quadratic_form import QuadraticForm
from dynkinlab.quasi_cartan import QuasiCartanMatrix

__version__ = __about__.__version__

__all__ = ["Inertia", "QuadraticForm", "QuasiCartanMatrix", "__version__"]
