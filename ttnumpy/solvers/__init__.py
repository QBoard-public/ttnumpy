# Copyright (c) 2026 QBoard LLC. Licensed under the AGPLv3. See LICENSE.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Iterative linear solvers on tensor trains.

This module exports only the vocabulary the solvers share: :class:`SolverInfo`,
the :class:`StopReason`, :class:`LocalSolver` and :class:`TruncationMode` enums,
and :func:`relative_residual`.
"""

from ttnumpy.solvers.common import (
    LocalSolver,
    SolverInfo,
    StopReason,
    TruncationMode,
    relative_residual,
)

__all__ = [
    "SolverInfo",
    "StopReason",
    "LocalSolver",
    "TruncationMode",
    "relative_residual",
]
