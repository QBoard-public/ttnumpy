# Copyright (c) 2026 QBoard LLC. Licensed under the AGPLv3. See LICENSE.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared machinery for the tensor-train linear solvers.

The vocabulary (:class:`StopReason`, :class:`LocalSolver`,
:class:`TruncationMode`, :class:`SolverInfo`), the environment builders, and
the option/monitoring/local-solve/truncation helpers used by all three
solvers (ALS, MALS and AMEn) live here.

References:
- Oseledets & Dolgov, "Solution of linear systems and matrix inversion in
  the TT-format", SIAM J. Sci. Comput. 34(5), 2012.
  https://doi.org/10.1137/110833142
- Dolgov & Savostyanov, "Alternating minimal energy methods for linear
  systems in higher dimensions", SIAM J. Sci. Comput. 36(5), 2014.
  https://doi.org/10.1137/140953289
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import scipy.linalg as lin
import scipy.sparse.linalg as spla

from ttnumpy import tensor_train
from ttnumpy.logger_config import PACKAGE_LOGGER_NAME

logger = logging.getLogger(PACKAGE_LOGGER_NAME)


class StopReason(str, Enum):
    """Why an iterative TT solve stopped.

    A ``str`` mixin so ``info.stop_reason == "converged"`` comparisons and
    JSON serialization keep working with the plain value.
    """

    #: The relative residual reached ``residual_tol``.
    CONVERGED = "converged"
    #: A sweep improved the residual by less than ``stagnation_factor``.
    STAGNATED = "stagnated"
    #: All ``max_sweeps`` sweeps were run.
    MAX_SWEEPS = "max_sweeps"

    def __str__(self) -> str:
        # Render as the bare value ("converged") on every Python version;
        # 3.11 changed mixin-enum __str__/__format__ to use the member name.
        return str.__str__(self)


class LocalSolver(str, Enum):
    """Local-system solver selection shared by the sweep solvers.

    ``AUTO`` picks MINRES or GMRES from the ``symmetric`` flag; ``DIRECT``
    forces the explicit LU solve.
    """

    AUTO = "auto"
    DIRECT = "direct"
    MINRES = "minres"
    GMRES = "gmres"

    def __str__(self) -> str:
        return str.__str__(self)


class TruncationMode(str, Enum):
    """How a solved local system is truncated to pick the next bond."""

    #: Smallest rank whose local residual stays within the per-truncation
    #: budget (``truncation_tol``).
    RESIDUAL = "residual"
    #: Drop singular values below ``error_threshold`` times the largest.
    THRESHOLD = "threshold"

    def __str__(self) -> str:
        return str.__str__(self)


@dataclass
class SolverInfo:
    """Convergence information collected during an iterative TT solve.

    Shared by the MALS, ALS and AMEn solvers.
    """

    #: Relative residual ||A @ x - b|| / ||b|| after each completed sweep.
    #: Empty if residual tracking was disabled.
    residuals: List[float] = field(default_factory=list)
    #: Number of sweeps actually performed.
    num_sweeps_run: int = 0
    #: Why the iteration ended (see :class:`StopReason`).
    stop_reason: StopReason = StopReason.MAX_SWEEPS
    #: Maximum bond dimension of the solution after each completed sweep.
    #: Populated by the rank-adaptive solvers (MALS and AMEn); left empty by
    #: ALS, whose ranks are fixed.
    max_bonds: List[int] = field(default_factory=list)
    #: Iterative local solves over the whole run.
    local_solves: int = 0
    #: Iterative local solves that missed ``local_rtol`` and were redone
    #: exactly on the explicit matrix.
    local_fallbacks: int = 0
    #: Iterative local solves that missed ``local_rtol`` and were accepted as
    #: they stood, because the dense fallback was not allowed.
    local_inexact: int = 0

    @property
    def converged(self) -> bool:
        """Whether the run stopped because residual_tol was reached."""
        return self.stop_reason == StopReason.CONVERGED


def relative_residual(
    A: tensor_train.TensorTrain,
    b: tensor_train.TensorTrain,
    x: tensor_train.TensorTrain,
    b_norm: Optional[float] = None,
) -> float:
    """Relative residual ||A @ x - b|| / ||b||, computed in TT arithmetic.

    The residual is formed as a tensor train and its norm is taken via
    orthogonalization, so nothing is densified. Taking the norm of the
    difference (rather than expanding it into scalar products) stays
    accurate for small residuals.

    Args:
        A: Operator in TTM format.
        b: Right-hand side in TT format.
        x: Candidate solution in TT format.
        b_norm: Precomputed ||b||, to avoid recomputation inside loops.

    Returns:
        float: ||A @ x - b|| / ||b||, or ||A @ x - b|| if ||b|| == 0.
    """
    residual = A.dot(x).add(b.multiply(-1), truncate=False)
    if b_norm is None:
        b_norm = b.tt_norm()
    residual_norm = residual.tt_norm()
    if b_norm == 0.0:
        return residual_norm
    return residual_norm / b_norm


def prepare_left_environment(
    ind: int,
    env: list[Optional[np.ndarray]],
    frame: tensor_train.TensorTrain,
    target: tensor_train.TensorTrain,
    A: Optional[tensor_train.TensorTrain] = None,
) -> None:
    """Extend the left environment projecting ``target`` onto ``frame``.

    All three solvers build every environment through this one recursion: the
    A and b environments of a same-train projection (``frame`` and ``target``
    both the solution) and the residual cross-environments of AMEn (distinct
    trains). ``frame`` is the train being projected onto (it is the
    conjugated one) and ``target`` the train being projected. With ``A`` the
    result is indexed (target_bond, operator_bond, frame_bond) and the
    operator's row mode contracts against ``frame``, its column mode against
    ``target``; without ``A`` it is an overlap indexed
    (target_bond, frame_bond).

    Results are written in place into ``env``.
    """
    if ind == 0:
        env[0] = np.ones([1, 1, 1]) if A is not None else np.ones([1, 1])
        return
    frame_core = frame[ind - 1][:, :, 0, :].conj()
    target_core = target[ind - 1][:, :, 0, :]
    # explicit chain rather than one einsum: at low operator rank einsum's
    # path optimizer picks a single naive contraction over the pairwise one,
    # which costs four orders of magnitude at large mode size
    if A is not None:
        # "abc,cek,bedj,adi->ijk"
        step = np.tensordot(env[ind - 1], frame_core, axes=([2], [0]))
        step = np.tensordot(step, A[ind - 1], axes=([1, 2], [0, 1]))
        env[ind] = np.tensordot(step, target_core, axes=([0, 2], [0, 1])).transpose(
            (2, 1, 0)
        )
    else:
        # "ab,bcj,aci->ij"
        step = np.tensordot(env[ind - 1], frame_core, axes=([1], [0]))
        env[ind] = np.tensordot(step, target_core, axes=([0, 1], [0, 1])).T


def prepare_right_environment(
    ind: int,
    env: list[Optional[np.ndarray]],
    frame: tensor_train.TensorTrain,
    target: tensor_train.TensorTrain,
    A: Optional[tensor_train.TensorTrain] = None,
) -> None:
    """Extend the right environment projecting ``target`` onto ``frame``.

    Mirror of :func:`prepare_left_environment`; see it for the index
    convention. Results are written in place into ``env``.
    """
    if ind == len(frame) - 1:
        env[-1] = np.ones([1, 1, 1]) if A is not None else np.ones([1, 1])
        return
    frame_core = frame[ind + 1][:, :, 0, :].conj()
    target_core = target[ind + 1][:, :, 0, :]
    # see prepare_left_environment on why this is a tensordot chain
    if A is not None:
        # "idc,jdeb,kea,abc->kji"
        step = np.tensordot(target_core, env[ind + 1], axes=([2], [0]))
        step = np.tensordot(step, A[ind + 1], axes=([1, 2], [2, 3]))
        env[ind] = np.tensordot(step, frame_core, axes=([1, 3], [2, 1]))
    else:
        # "jcb,ica,ab->ij"
        step = np.tensordot(target_core, env[ind + 1], axes=([2], [0]))
        env[ind] = np.tensordot(step, frame_core, axes=([1, 2], [1, 2]))


# Fraction of residual_tol the iterative local solves target, so that a
# loosely solved system does not drive the truncation. Shared over a sweep the
# two budgets are the same order, so the margin holds only while a solve lands
# well inside its target -- which the exact direct path always does.
_LOCAL_TOL_FACTOR = 0.1

# Tolerance the iterative local solves target when nothing else fixes one.
_DEFAULT_LOCAL_RTOL = 1e-8

# Restarts a Krylov solve gets to reach local_rtol before it is given up on.
_MAX_LOCAL_RESTARTS = 4

# Tightening factor for the Krylov tolerance on each restart.
_RESTART_RTOL_SHRINK = 1e-2

# Stagnation factor used when stagnation_factor="auto" resolves to an
# enabled check (i.e. when residuals are tracked anyway).
_DEFAULT_STAGNATION_FACTOR = 0.9

# Share of a sweep's iterative local solves that makes a fallback or
# accepted-inexact rate worth reporting above DEBUG.
_LOCAL_RATE_REPORT_SHARE = 0.5


@dataclass(frozen=True)
class _SolverSettings:
    """Validated and derived options shared by ALS, AMEn and MALS.

    Built by :meth:`resolve`, which runs the argument checks common to all
    three solvers and derives the fields both drivers otherwise computed
    twice, so the sweep loops read a single settings object.
    """

    max_sweeps: int
    # TODO: residual_tol, truncation_tol and local_rtol are all relative
    # residual targets, spelled ``_tol`` in two and ``_rtol`` in one QBRD-5867
    residual_tol: Optional[float]
    #: Resolved stagnation factor: a float in (0, 1], or None if disabled.
    stagnation_factor: Optional[float]
    #: Krylov method for large local systems, or None to always use the
    #: explicit LU solve.
    iterative_solver: Optional[LocalSolver]
    direct_solve_size: int
    #: Whether the relative residual is measured after every sweep.
    track_residual: bool
    #: Tolerance the iterative local solves are held to; None exactly when
    #: there is no iterative path (``local_solver="direct"``).
    local_rtol: Optional[float]
    #: Whether a local solve that misses ``local_rtol`` may be redone on the
    #: explicit matrix, as the caller asked.
    allow_dense_fallback: bool = False
    #: Truncations performed per sweep, used to split the global residual
    #: budget between them. 1 for a solver that does not truncate.
    truncation_steps: int = 1

    @property
    def truncation_mode(self) -> TruncationMode:
        """Residual-budgeted when ``residual_tol`` is set, else the cutoff."""
        return (
            TruncationMode.RESIDUAL
            if self.residual_tol is not None
            else TruncationMode.THRESHOLD
        )

    @property
    def truncation_tol(self) -> Optional[float]:
        """Residual budget for a single truncation, or None without a target.

        ``residual_tol`` shared over the ``truncation_steps`` truncations of a
        sweep, whose errors add in quadrature. The stopping test uses the full
        ``residual_tol``; see the user guide for why only one of the two is
        divided.
        """
        if self.residual_tol is None:
            return None
        return self.residual_tol / np.sqrt(self.truncation_steps)

    def use_iterative(self, size: int) -> bool:
        """Whether a local system of ``size`` unknowns goes to the Krylov path."""
        return self.iterative_solver is not None and size > self.direct_solve_size

    @classmethod
    def resolve(
        cls,
        *,
        max_sweeps: int,
        residual_tol: Optional[float],
        stagnation_factor: Union[float, str, None],
        local_solver: Union[str, LocalSolver],
        symmetric: bool,
        direct_solve_size: int,
        return_info: bool,
        local_rtol: Optional[float] = None,
        allow_dense_fallback: bool = False,
        truncation_steps: int = 1,
    ) -> "_SolverSettings":
        """Validate the shared options and derive the dependent ones."""
        if residual_tol is not None and residual_tol <= 0.0:
            raise ValueError(f"residual_tol must be positive, got {residual_tol}")
        if truncation_steps < 1:
            raise ValueError(
                f"truncation_steps must be positive, got {truncation_steps}"
            )

        if isinstance(stagnation_factor, str):
            if stagnation_factor != "auto":
                raise ValueError(
                    "stagnation_factor must be a float in (0, 1], None or "
                    f"'auto', got {stagnation_factor!r}"
                )
            # the stagnation check needs a residual per sweep, so enable
            # it by default only when residuals are computed anyway
            stagnation_factor = (
                _DEFAULT_STAGNATION_FACTOR
                if residual_tol is not None or return_info
                else None
            )
        if stagnation_factor is not None and not 0.0 < stagnation_factor <= 1.0:
            raise ValueError(
                f"stagnation_factor must be in (0, 1], got {stagnation_factor}"
            )

        # residual tracking is enabled when the caller asked for it, or when
        # it is needed for the stopping checks (residual_tol or stagnation_factor)
        track_residual = (
            return_info
            or residual_tol is not None
            or (stagnation_factor is not None and max_sweeps > 1)
        )

        if local_rtol is not None and local_rtol <= 0.0:
            raise ValueError(f"local_rtol must be positive, got {local_rtol}")

        try:
            local_solver = LocalSolver(local_solver)
        except ValueError as exc:
            allowed = ", ".join(repr(member.value) for member in LocalSolver)
            raise ValueError(
                f"local_solver must be one of {allowed}, got {local_solver!r}"
            ) from exc

        if local_solver is LocalSolver.AUTO:
            iterative_solver = LocalSolver.MINRES if symmetric else LocalSolver.GMRES
        elif local_solver is LocalSolver.DIRECT:
            iterative_solver = None
        else:
            iterative_solver = local_solver

        # local residual tolerance needed only in the iterative path;
        # if None, calculated from residual_tol or set to a fixed default
        if iterative_solver is None:
            resolved_local_rtol = None
        elif local_rtol is not None:
            resolved_local_rtol = local_rtol
        elif residual_tol is not None:
            resolved_local_rtol = _LOCAL_TOL_FACTOR * residual_tol
        else:
            resolved_local_rtol = _DEFAULT_LOCAL_RTOL

        if (
            resolved_local_rtol is not None
            and residual_tol is not None
            and resolved_local_rtol > 0.5 * residual_tol
        ):
            logger.warning(
                "local_rtol %.2e exceeds half of residual_tol %.2e: the local "
                "solves may be too loose to drive the truncation budget",
                resolved_local_rtol,
                residual_tol,
            )
        return cls(
            max_sweeps=max_sweeps,
            residual_tol=residual_tol,
            stagnation_factor=stagnation_factor,
            iterative_solver=iterative_solver,
            direct_solve_size=direct_solve_size,
            track_residual=track_residual,
            local_rtol=resolved_local_rtol,
            allow_dense_fallback=allow_dense_fallback,
            truncation_steps=truncation_steps,
        )


@dataclass
class _SweepStats:
    """Tallies of the decisions taken inside one sweep.

    Filled by the local-solve and truncation helpers as a sweep runs, then
    reported and cleared by :meth:`_SweepMonitor.record`. Counting per sweep
    rather than logging per site keeps a solve to a handful of lines.
    """

    #: Cap the rank decisions are counted against; None disables their line.
    max_rank: Optional[int] = None
    #: Rank decisions, and how many ``max_rank`` bounded rather than accuracy.
    rank_decisions: int = 0
    rank_capped: int = 0
    #: Iterative local solves, and how many missed tolerance and were redone
    #: exactly on the explicit matrix.
    local_solves: int = 0
    local_fallbacks: int = 0
    #: Iterative local solves that missed tolerance and were accepted as they
    #: stood, because the caller ruled out the dense fallback.
    local_inexact: int = 0

    def record_rank(self, capped: bool) -> None:
        """Count one rank decision."""
        self.rank_decisions += 1
        self.rank_capped += int(capped)

    def record_local_solve(
        self, fell_back: bool = False, inexact: bool = False
    ) -> None:
        """Count one iterative local solve."""
        self.local_solves += 1
        self.local_fallbacks += int(fell_back)
        self.local_inexact += int(inexact)

    def clear(self) -> None:
        """Reset the counters for the next sweep."""
        self.rank_decisions = 0
        self.rank_capped = 0
        self.local_solves = 0
        self.local_fallbacks = 0
        self.local_inexact = 0


class _SweepMonitor:
    """Per-sweep residual tracking, stopping decision and logging.

    Shared tail of the one-site and two-site drivers: each sweep calls
    :meth:`record` once the cores are updated, stopping when it returns True,
    and :meth:`finish` emits the closing summary. The :class:`SolverInfo` it
    fills is available as :attr:`info`.
    """

    def __init__(
        self,
        settings: _SolverSettings,
        name: str,
        A: tensor_train.TensorTrain,
        b: tensor_train.TensorTrain,
        x: Optional[tensor_train.TensorTrain] = None,
        error_threshold: Optional[float] = None,
        max_rank: Optional[int] = None,
    ) -> None:
        self.settings = settings
        self.name = name
        self.A = A
        self.b = b
        self.b_norm = b.tt_norm() if settings.track_residual else None
        self.info = SolverInfo()
        self._sweep_started = time.perf_counter()
        self._log_start(x, error_threshold, max_rank)

    def _log_start(
        self,
        x: Optional[tensor_train.TensorTrain],
        error_threshold: Optional[float],
        max_rank: Optional[int],
    ) -> None:
        """Record the options this solve is running under."""
        settings = self.settings
        logger.debug(
            "%s starting: %s cores, modes %s, initial max bond %s | "
            "max_sweeps=%d truncation=%s residual_tol=%s error_threshold=%s "
            "max_rank=%s stagnation_factor=%s | local_solver=%s local_rtol=%s "
            "direct_solve_size=%d dense_fallback=%s",
            self.name,
            len(self.b),
            self.b.physical_dims() if hasattr(self.b, "physical_dims") else "?",
            max(x.bonds) if x is not None else "?",
            settings.max_sweeps,
            settings.truncation_mode,
            settings.residual_tol,
            error_threshold,
            max_rank,
            settings.stagnation_factor,
            settings.iterative_solver or "direct",
            settings.local_rtol,
            settings.direct_solve_size,
            settings.allow_dense_fallback,
        )

    def record(
        self,
        sweep: int,
        x: tensor_train.TensorTrain,
        max_bond: Optional[int] = None,
        stats: Optional[_SweepStats] = None,
    ) -> bool:
        """Update :attr:`info` for the completed sweep; True means stop.

        ``max_bond`` is appended to ``info.max_bonds`` when given (the
        rank-adaptive solvers' per-sweep trace). ``stats`` is reported, added
        to the run totals on :attr:`info` and cleared when given. The relative
        residual is measured, logged and tested against ``residual_tol`` and
        the stagnation factor only when residual tracking is enabled.

        The sweep's wall time is logged alongside, which localises cost to a
        sweep; the benchmark runner only times whole solves.
        """
        elapsed = time.perf_counter() - self._sweep_started
        info = self.info
        info.num_sweeps_run = sweep + 1
        if max_bond is not None:
            info.max_bonds.append(max_bond)
        self._log_sweep_stats(sweep, stats)
        if stats is not None:
            info.local_solves += stats.local_solves
            info.local_fallbacks += stats.local_fallbacks
            info.local_inexact += stats.local_inexact
            stats.clear()
        try:
            return self._record_residual(sweep, x, elapsed)
        finally:
            # measuring the residual is part of a sweep's cost, so the clock
            # for the next one starts only once this sweep is fully accounted
            self._sweep_started = time.perf_counter()

    def _record_residual(
        self, sweep: int, x: tensor_train.TensorTrain, elapsed: float
    ) -> bool:
        """Measure and test the sweep's residual; True means stop."""
        info = self.info
        if not self.settings.track_residual:
            logger.debug(
                "%s sweep %d/%d: %.3fs",
                self.name,
                sweep + 1,
                self.settings.max_sweeps,
                elapsed,
            )
            return False

        rel_residual = relative_residual(self.A, self.b, x, b_norm=self.b_norm)
        info.residuals.append(rel_residual)
        logger.debug(
            "%s sweep %d/%d: relative residual %.3e, max bond %d, %.3fs",
            self.name,
            sweep + 1,
            self.settings.max_sweeps,
            rel_residual,
            max(x.bonds),
            elapsed,
        )

        if (
            self.settings.residual_tol is not None
            and rel_residual <= self.settings.residual_tol
        ):
            info.stop_reason = StopReason.CONVERGED
            return True
        if (
            self.settings.stagnation_factor is not None
            and len(info.residuals) >= 2
            and info.residuals[-1]
            > self.settings.stagnation_factor * info.residuals[-2]
        ):
            info.stop_reason = StopReason.STAGNATED
            previous = info.residuals[-2]
            logger.warning(
                "%s stagnated at sweep %d: relative residual %.3e vs %.3e "
                "previous (ratio %.2f > %.2f)",
                self.name,
                sweep + 1,
                info.residuals[-1],
                previous,
                info.residuals[-1] / previous if previous else float("inf"),
                self.settings.stagnation_factor,
            )
            return True
        return False

    def _log_sweep_stats(self, sweep: int, stats: Optional[_SweepStats]) -> None:
        """Report the sweep's tallies, if there are any.

        Rank capping is only meaningful with a ``max_rank`` to cap against,
        and ALS truncates nothing, so either line is skipped when its counter
        is empty. The caller owns the tallies afterwards; see :meth:`record`.
        """
        if stats is None:
            return
        if stats.rank_decisions and stats.max_rank is not None:
            logger.debug(
                "%s sweep %d: %d/%d rank decisions capped by max_rank=%d",
                self.name,
                sweep + 1,
                stats.rank_capped,
                stats.rank_decisions,
                stats.max_rank,
            )
        if stats.local_solves and self.settings.allow_dense_fallback:
            if self._is_high_rate(stats.local_fallbacks, stats.local_solves):
                logger.warning(
                    "%s sweep %d: %d/%d iterative local solves rebuilt the "
                    "explicit matrix; at this rate local_solver='direct' is "
                    "cheaper, or raise local_rtol",
                    self.name,
                    sweep + 1,
                    stats.local_fallbacks,
                    stats.local_solves,
                )
            else:
                logger.debug(
                    "%s sweep %d: %d/%d iterative local solves fell back to "
                    "the explicit matrix",
                    self.name,
                    sweep + 1,
                    stats.local_fallbacks,
                    stats.local_solves,
                )
        if stats.local_inexact:
            logger.log(
                (
                    logging.WARNING
                    if self._is_high_rate(stats.local_inexact, stats.local_solves)
                    else logging.DEBUG
                ),
                "%s sweep %d: %d/%d iterative local solves missed local_rtol "
                "%.2e and were accepted inexact (allow_dense_fallback=False); "
                "the run may stagnate above residual_tol",
                self.name,
                sweep + 1,
                stats.local_inexact,
                stats.local_solves,
                self.settings.local_rtol,
            )

    @staticmethod
    def _is_high_rate(count: int, total: int) -> bool:
        """Whether ``count`` covers enough of the sweep to report loudly."""
        return bool(total) and count / total > _LOCAL_RATE_REPORT_SHARE

    def finish(self) -> None:
        """Log the closing summary line, if any residual was recorded.

        A solve that runs out of sweeps while a ``residual_tol`` was asked for
        returned something less accurate than requested, which the summary
        line alone does not make obvious.
        """
        if not self.info.residuals:
            return
        logger.info(
            "%s stopped after %d sweep(s) (%s), relative residual %.3e",
            self.name,
            self.info.num_sweeps_run,
            self.info.stop_reason,
            self.info.residuals[-1],
        )
        if (
            self.settings.residual_tol is not None
            and self.info.stop_reason is StopReason.MAX_SWEEPS
        ):
            logger.warning(
                "%s did not reach residual_tol %.2e after %d sweep(s) "
                "(final relative residual %.3e)",
                self.name,
                self.settings.residual_tol,
                self.info.num_sweeps_run,
                self.info.residuals[-1],
            )


def _validate_solver_inputs(
    A: tensor_train.TensorTrain,
    b: tensor_train.TensorTrain,
    x0: tensor_train.TensorTrain,
) -> None:
    """Check structural compatibility of a TT linear-solve triple ``A x = b``.

    Shared by all three solvers. Verifies that ``A`` is a TTM, ``b`` and
    ``x0`` are TensorTrains, the three trains share the same number of cores,
    ``A`` is square, and its row/column modes match ``b`` and ``x0``
    respectively. Raises ``WrongTTType`` or ``DimensionMismatch`` on the first
    violation.
    """
    if x0.is_ttm():
        raise tensor_train.WrongTTType(
            "Initial guess x0 must be a TensorTrain, not a TTM."
        )
    if b.is_ttm():
        raise tensor_train.WrongTTType(
            "Right-hand side b must be a TensorTrain, not a TTM."
        )
    if not A.is_ttm():
        raise tensor_train.WrongTTType("Matrix A must be a TTM, not a TensorTrain.")
    if not len(A) == len(b) == len(x0):
        raise tensor_train.DimensionMismatch(
            "A, b and x0 must have the same number of cores, "
            f"got {len(A)}, {len(b)} and {len(x0)}"
        )
    if A.row_modes != A.col_modes:
        raise tensor_train.DimensionMismatch(
            f"Solver requires a square operator, got row modes {A.row_modes} "
            f"and column modes {A.col_modes}"
        )
    if A.row_modes != b.row_modes:
        raise tensor_train.DimensionMismatch(
            f"Row modes of A must match physical dimensions of b: "
            f"{A.row_modes} != {b.row_modes}"
        )
    if A.col_modes != x0.row_modes:
        raise tensor_train.DimensionMismatch(
            f"Column modes of A must match physical dimensions of x0: "
            f"{A.col_modes} != {x0.row_modes}"
        )


def _solve_local_system(
    operator: Union[np.ndarray, spla.LinearOperator],
    b_loc: np.ndarray,
    *,
    overwrite_a: bool,
    iterative_solver: Optional[LocalSolver],
    local_rtol: Optional[float],
    warm_start: Optional[np.ndarray] = None,
    explicit_fallback: Optional[Callable[[], np.ndarray]] = None,
    stats: Optional[_SweepStats] = None,
) -> Tuple[np.ndarray, Union[np.ndarray, spla.LinearOperator]]:
    """Solve one local system, returning the solution and the operator used.

    With ``iterative_solver`` None the explicit ``operator`` matrix is solved
    by LU. Otherwise a warm-started Krylov method held to ``local_rtol`` is
    used, which :meth:`_SolverSettings.resolve` guarantees is set whenever a
    method is; neither has a default here, so a caller has to state both. The
    returned operator is whichever one produced the solution — the two-site
    caller needs it to measure the exact residual for residual-based
    truncation.

    ``overwrite_a`` is forwarded to the direct solves: pass False to keep the
    operator intact when residual-based truncation will reuse it, True when
    the solution alone is wanted.

    ``explicit_fallback`` decides what happens when the Krylov solve still
    misses its tolerance: given a callable producing the explicit matrix, the
    solve is redone on it exactly; left None, the solve stays matrix-free and
    the last iterate is returned as it stands. The outer sweep is
    self-correcting, so an inexact local solve usually costs iterations
    rather than the answer.

    ``stats``, when given, counts the iterative solves and how they ended for
    the sweep's log lines. The direct path is not counted: it never attempted
    an iterative solve, so it cannot have missed anything.
    """
    b_vec = b_loc.ravel()
    if iterative_solver is None:
        u = lin.solve(
            operator,
            b_vec,
            overwrite_a=overwrite_a,
            overwrite_b=False,
            check_finite=False,
        )
        return np.asarray(u).ravel(), operator

    target = local_rtol * np.linalg.norm(b_vec)
    u = warm_start
    krylov_rtol = local_rtol
    # scipy's stopping tests can report convergence far from the target on
    # ill-conditioned systems, so check the true residual and restart tighter
    # until it actually holds.
    for _ in range(_MAX_LOCAL_RESTARTS):
        if iterative_solver is LocalSolver.MINRES:
            u, _ = spla.minres(operator, b_vec, x0=u, rtol=krylov_rtol)
        else:
            u, _ = spla.gmres(operator, b_vec, x0=u, rtol=krylov_rtol, atol=0.0)
        if np.linalg.norm(operator @ u - b_vec) <= target:
            if stats is not None:
                stats.record_local_solve(fell_back=False)
            return np.asarray(u).ravel(), operator
        krylov_rtol = max(krylov_rtol * _RESTART_RTOL_SHRINK, 1e-15)

    # out of restarts: redo the solve exactly, unless the caller ruled the
    # dense operator out by passing no fallback
    if explicit_fallback is None:
        if stats is not None:
            stats.record_local_solve(inexact=True)
        return np.asarray(u).ravel(), operator
    if stats is not None:
        stats.record_local_solve(fell_back=True)
    operator = explicit_fallback()
    u = lin.solve(
        operator,
        b_vec,
        overwrite_a=overwrite_a,
        overwrite_b=False,
        check_finite=False,
    )
    return np.asarray(u).ravel(), operator


def _truncation_rank_by_residual(
    A_loc: np.ndarray,
    b_loc: np.ndarray,
    U: np.ndarray,
    S: np.ndarray,
    V: np.ndarray,
    exact_residual: float,
    truncation_tol: float,
    residual_damp: float,
    max_rank: Optional[int],
) -> Tuple[int, bool]:
    """Smallest truncation rank whose local residual stays within budget.

    Residual-based truncation (trick 2, section 4.2 of Oseledets & Dolgov
    2012): keep the smallest rank r such that

        ||A_loc @ w_r - b_loc|| <= max(residual_damp * exact_residual,
                                       0.5 * truncation_tol * ||b_loc||),

    where w_r is the rank-r SVD truncation of the local solution. The
    absolute floor keeps compression possible when the local solve is
    near exact, while staying safely below the global target.
    ``truncation_tol`` is one truncation's share of ``residual_tol``, not the
    global target; see ``truncation_tol``.

    Returns the rank and whether ``max_rank`` was the binding constraint:
    the budget was still unmet at the last rank tried, and more singular
    values were available than ``max_rank`` allowed.
    """
    if S[0] == 0.0:
        return 1, False
    limit = len(S)
    if max_rank is not None:
        limit = min(limit, max_rank)
    # The floor is what keeps an accurate solve compressible. The damp term
    # only takes over once the local solve's residual exceeds
    # 0.5 * truncation_tol / residual_damp, so on the direct path, whose
    # residual is at machine precision, residual_damp has no effect at all.
    threshold = max(
        residual_damp * exact_residual,
        0.5 * truncation_tol * np.linalg.norm(b_loc),
    )
    # terms[i] = vec(s_i * u_i v_i^T), the i-th rank-1 piece of the solution
    terms = np.einsum(
        "ik,kj->kij", U[:, :limit] * S[:limit], V[:limit, :], optimize="optimal"
    )
    contributions = A_loc @ terms.reshape(limit, -1).T
    partial = -b_loc.copy()
    for rank in range(limit):
        partial += contributions[:, rank]
        if np.linalg.norm(partial) <= threshold:
            return rank + 1, False
    # the budget was never met: max_rank bound the search only if it, rather
    # than the number of singular values, is what stopped it
    return limit, max_rank is not None and max_rank < len(S)


# TODO: ``error_threshold`` compares each singular value with sigma_1 of its
# own core, a different criterion from the Frobenius ``error`` of
# tensor_train.truncate under a different name QBRD-5867
def _svd_truncation_rank(
    S: np.ndarray, error_threshold: float, max_rank: Optional[int]
) -> Tuple[int, bool]:
    """Rank keeping singular values above ``error_threshold * sigma_1``.

    Returns the rank and whether ``max_rank`` cut it below what the accuracy
    rule alone would have kept.
    """
    if S.size == 0 or S[0] == 0.0:
        return 1, False
    rank = max(int(np.count_nonzero(S / S[0] > error_threshold)), 1)
    if max_rank is not None and rank > max_rank:
        return max_rank, True
    return rank, False


def _choose_truncation_rank(
    mode: TruncationMode,
    operator: Union[np.ndarray, spla.LinearOperator],
    b_loc: np.ndarray,
    U: np.ndarray,
    S: np.ndarray,
    V: np.ndarray,
    exact_residual: float,
    error_threshold: float,
    max_rank: Optional[int],
    truncation_tol: Optional[float],
    residual_damp: float,
    stats: Optional[_SweepStats] = None,
) -> int:
    """Truncation rank for a solved local system, shared by all solvers.

    ``mode`` selects the rule: :data:`TruncationMode.RESIDUAL` keeps the
    smallest rank whose local residual stays within budget;
    :data:`TruncationMode.THRESHOLD` drops singular values below
    ``error_threshold * sigma_1`` (:func:`_svd_truncation_rank`). This
    is the one accuracy-vs-rank rule MALS applies to its supercore and AMEn to
    each solved core.

    ``stats``, when given, tallies the decision for the sweep's log line.
    """
    if mode is TruncationMode.RESIDUAL:
        rank, capped = _truncation_rank_by_residual(
            operator,
            b_loc,
            U,
            S,
            V,
            exact_residual,
            truncation_tol,
            residual_damp,
            max_rank,
        )
    else:
        rank, capped = _svd_truncation_rank(S, error_threshold, max_rank)
    if stats is not None:
        stats.record_rank(capped)
    return rank
