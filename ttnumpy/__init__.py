# Copyright (c) 2026 QBoard LLC. Licensed under the AGPLv3. See LICENSE.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Public tensor-train API for TTNumPy."""

from . import solvers, tensor_train, utils
from .solvers import (
    LocalSolver,
    SolverInfo,
    StopReason,
    TruncationMode,
    relative_residual,
)

from .solvers.als import als
from .solvers.amen import amen
from .solvers.mals import mals
from .tensor_train import (
    TensorTrain,
    TensorTrainType,
    canonical_tt,
    eye,
    ones,
    rand,
    tt_svd,
    zeros,
)

__all__ = [
    "tensor_train",
    "TensorTrain",
    "TensorTrainType",
    "canonical_tt",
    "tt_svd",
    "eye",
    "ones",
    "rand",
    "zeros",
    "solvers",
    "SolverInfo",
    "StopReason",
    "LocalSolver",
    "TruncationMode",
    "als",
    "amen",
    "mals",
    "relative_residual",
    "utils",
]
