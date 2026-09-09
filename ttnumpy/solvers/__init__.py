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
