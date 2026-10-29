# SPDX-FileCopyrightText: 2026-present Mario Abarca <knkillname@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Public package surface for DynkinLab."""

from dynkinlab import display as display
from dynkinlab.bigraph import Bigraph, BigraphEdge, EdgeStyle
from dynkinlab.dynkin import (
    ClassificationStatus,
    ComponentClassification,
    DynkinClassification,
    DynkinType,
)
from dynkinlab.flation import FlationSequence, FlationStep
from dynkinlab.inertia import Inertia
from dynkinlab.quadratic_form import QuadraticForm
from dynkinlab.quasi_cartan import QuasiCartanMatrix
from dynkinlab.structure import (
    ABlockPartition,
    ATypeCertificate,
    ATypeComponentCertificate,
    Block,
    BlockCutForest,
    BlockCutTree,
)
from dynkinlab.symmetrizer import (
    NotSymmetrizableError,
    SymmetrizerCertificate,
    SymmetrizerObstruction,
)

__version__ = "0.0.1"

__all__ = [
    "ABlockPartition",
    "ATypeCertificate",
    "ATypeComponentCertificate",
    "Bigraph",
    "BigraphEdge",
    "Block",
    "BlockCutForest",
    "BlockCutTree",
    "ClassificationStatus",
    "ComponentClassification",
    "DynkinClassification",
    "DynkinType",
    "EdgeStyle",
    "FlationSequence",
    "FlationStep",
    "Inertia",
    "NotSymmetrizableError",
    "QuadraticForm",
    "QuasiCartanMatrix",
    "SymmetrizerCertificate",
    "SymmetrizerObstruction",
    "__version__",
]
