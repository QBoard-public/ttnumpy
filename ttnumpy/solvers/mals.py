# Copyright (c) 2026 QBoard LLC. Licensed under the AGPLv3. See LICENSE.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Modified Alternating Least Squares (MALS) linear solver on tensor trains.

Cost, for ``n`` sites of mode size ``d``, solution bond dimension ``r``,
operator bond dimension ``R`` and right-hand-side bond dimension ``r_b``. A
sweep visits every bond twice, and the local system on two neighbouring sites
has ``N = r^2 d^2`` unknowns.

- Local solve on the explicit matrix, used up to ``direct_solve_size``
  unknowns: ``O(r^4 d^4 R + r^2 d^4 R^2)`` to build the matrix, ``O(N^3)`` to
  factorize it, ``O(N^2)`` memory.
- Matrix-free local solve: ``O(r^3 d^2 R + r^2 d^3 R^2)`` per Krylov
  iteration.
- Splitting the solved supercore: an SVD in ``O(r^3 d^3)``. With
  ``residual_tol`` set, choosing the rank applies the local operator to up to
  ``r d`` rank-one terms.
- Environment update after each step: ``O(r^3 d R + r^2 d^2 R^2)`` for the
  operator, and ``O(r r_b d (r d + r_b))`` for the right-hand side with its
  projection onto the two sites. The environments of all sites hold
  ``O(n (r^2 R + r r_b))`` numbers.
- Residual after a sweep, when residuals are tracked:
  ``O(n d (d r^2 R^2 + (r R + r_b)^3))`` to form and orthonormalize the train
  ``A x - b``.

Overall, a sweep takes ``O(n (r^6 d^6 + r^4 d^4 R + r^2 d^4 R^2))`` with
explicit local matrices, or ``O(n m (r^3 d^2 R + r^2 d^3 R^2) + n r^3 d^3)``
with ``m`` applications of the local operator per step (the Krylov
iterations, plus up to ``r d`` for residual-based truncation), plus
``O(n r r_b d (r d + r_b))`` for the right-hand side and, when residuals are
tracked, ``O(n d (d r^2 R^2 + (r R + r_b)^3))``. Memory is
``O(n (d r^2 + d^2 R^2 + d r_b^2 + r^2 R + r r_b))`` for the trains and
environments, plus ``O(r^4 d^4)`` for an explicit local matrix,
``O(r^3 d^3)`` for the rank-one terms of residual-based truncation and
``O(n d (r R + r_b)^2)`` for the train ``A x - b``.

Reference:
- Oseledets & Dolgov, SIAM J. Sci. Comput. 34(5), 2012,
  https://doi.org/10.1137/110833142 (sections 3.3-3.4 and 4).
"""

from typing import Optional, Tuple, Union

import numpy as np

from ttnumpy import tensor_train
from ttnumpy.solvers.common import (
    LocalSolver,
    SolverInfo,
    _SolverSettings,
    _SweepMonitor,
    _SweepStats,
    _validate_solver_inputs,
    prepare_left_environment,
    prepare_right_environment,
)
from ttnumpy.solvers.two_site import _solve_local_two_site


def mals(
    A: tensor_train.TensorTrain,
    b: tensor_train.TensorTrain,
    x0: Optional[tensor_train.TensorTrain] = None,
    inplace: bool = False,
    max_rank: Optional[int] = None,
    max_sweeps: int = 2,
    residual_tol: Optional[float] = None,
    residual_damp: float = 2.0,
    stagnation_factor: Union[float, str, None] = "auto",
    error_threshold: float = 1e-7,
    local_solver: Union[str, LocalSolver] = "auto",
    direct_solve_size: int = 1000,
    local_rtol: Optional[float] = None,
    allow_dense_fallback: bool = False,
    symmetric: bool = False,
    return_info: bool = False,
) -> Union[tensor_train.TensorTrain, Tuple[tensor_train.TensorTrain, SolverInfo]]:
    """Modified Alternating Least Squares (MALS) method for solving Ax = b.

    Sweeps over pairs of neighbouring cores, solving a small projected
    system for each pair and adapting the ranks in the decimation step.
    After each full sweep the relative residual ||A @ x - b|| / ||b|| is
    measured in TT arithmetic; the iteration stops when it reaches
    ``residual_tol``, stagnates, or ``max_sweeps`` is exhausted.

    Truncation works in one of two modes:

    * ``residual_tol`` set (recommended): residual-based truncation. Each
      decimation keeps the smallest rank whose local residual stays within
      budget, so ``residual_tol`` is the single accuracy parameter and
      ``error_threshold`` is ignored.
    * ``residual_tol`` None: cutoff discarding singular values below
      ``error_threshold * sigma_1``. This controls the error of the
      *solution*, but leaves the *residual* bounded only by
      cond(A) * error_threshold.

    Local systems are solved directly (LU on the explicit matrix) while
    small; above ``direct_solve_size`` unknowns they are solved matrix-free
    with a warm-started Krylov method. A Krylov solve that misses ``local_rtol``
    is accepted as it stands, which keeps the memory bounded and leaves the
    outer sweep to correct it; ``allow_dense_fallback`` trades that for
    exactness.

    Args:
        A: Operator in TTM format.
        b: Right-hand side in TT format.
        x0: Initial guess for the solution in TT format.
        inplace: If True, modify x0 in place instead of copying it.
        max_rank: Maximum allowed bond dimension.
        max_sweeps: Maximum number of sweeps.
        residual_tol: Target relative residual; enables residual-based
            truncation and early stopping. None disables both. The stopping
            test uses it whole; the truncations of a sweep share it, since
            their errors add in quadrature.
        residual_damp: Allowed growth factor of the local residual due to
            truncation (reasonable values 2 to 5). Must be >= 1. It scales
            the local solve's own residual, which the direct path drives to
            machine precision, so it changes nothing there -- only a loosely
            solved iterative system is affected.
        stagnation_factor: Stop when a sweep improves the residual by less
            than this factor; must be in (0, 1]. "auto" (default) enables
            the check with factor 0.9 when residuals are tracked anyway
            (residual_tol set or return_info=True) and disables it
            otherwise, so pure error_threshold runs do not pay a per-sweep
            residual evaluation. An explicit float always enables the check
            (and with it the residual tracking); None disables it.
        error_threshold: Relative SVD truncation threshold; only used when
            residual_tol is None.
        local_solver: Krylov method for large local systems. "auto" picks
            MINRES when ``symmetric`` is True and GMRES otherwise;
            "minres"/"gmres" force one; "direct" always uses the explicit
            LU solve.
        direct_solve_size: Local systems with at most this many rows are
            solved directly.
        local_rtol: Relative tolerance the iterative local solves are held
            to. None derives it from ``residual_tol`` when that is set, and
            otherwise uses a fixed default.
        allow_dense_fallback: If True, a local solve that misses
            ``local_rtol`` is redone exactly on the explicit matrix.
            False by default and the inexact iterate is kept instead.
        symmetric: Declare A symmetric so "auto" can use MINRES.
        return_info: If True, also return a SolverInfo with per-sweep
            residuals, per-sweep maximum bond dimensions (``max_bonds``), the
            stop reason, and how the iterative local solves ended
            (``local_solves``, ``local_fallbacks``, ``local_inexact``).

    Returns:
        TensorTrain: Solution x, or a tuple (x, SolverInfo) if return_info
        is True.

    Reference:
        Oseledets & Dolgov, SIAM J. Sci. Comput. 34(5), 2012,
        https://doi.org/10.1137/110833142 (sections 3.3-3.4 and 4).
    """
    if max_rank is not None and max_rank <= 0:
        raise ValueError(f"max_rank must be a positive integer, got {max_rank}")
    if residual_damp < 1.0:
        raise ValueError(f"residual_damp must be >= 1, got {residual_damp}")

    settings = _SolverSettings.resolve(
        max_sweeps=max_sweeps,
        residual_tol=residual_tol,
        stagnation_factor=stagnation_factor,
        local_solver=local_solver,
        symmetric=symmetric,
        direct_solve_size=direct_solve_size,
        return_info=return_info,
        local_rtol=local_rtol,
        allow_dense_fallback=allow_dense_fallback,
        truncation_steps=max(2 * A.n_cores - 3, 1),
    )

    if x0 is None:
        x0 = tensor_train.ones(b.physical_dims(), ranks=b.bonds)

    _validate_solver_inputs(A, b, x0)

    x = x0 if inplace else x0.copy()

    if x.orth_center != 0:
        x.orthonormalize(canonical="right", inplace=True)

    right_env_A = [None] * len(x)
    right_env_b = [None] * len(x)
    left_env_A = [None] * len(x)
    left_env_b = [None] * len(x)

    for i in range(len(x) - 1, 0, -1):
        prepare_right_environment(i, right_env_A, x, x, A)
        prepare_right_environment(i, right_env_b, x, b)

    monitor = _SweepMonitor(
        settings,
        "MALS",
        A,
        b,
        x,
        error_threshold=error_threshold,
        max_rank=max_rank,
    )
    sweep_stats = _SweepStats(max_rank=max_rank)

    # (site index, block) left behind by the latest decimation; feeds
    # consistent warm starts while neighbouring cores are stale mid-sweep
    center: Optional[Tuple[int, np.ndarray]] = None

    for sweep in range(settings.max_sweeps):

        # left-to-right sweep
        for k in range(len(x) - 1):
            prepare_left_environment(k, left_env_A, x, x, A)
            prepare_left_environment(k, left_env_b, x, b)

            if k != len(x) - 2:
                U, S, V, next_bond = _solve_local_two_site(
                    k,
                    x,
                    A,
                    b,
                    left_env_A,
                    right_env_A,
                    left_env_b,
                    right_env_b,
                    settings,
                    error_threshold=error_threshold,
                    max_rank=max_rank,
                    residual_damp=residual_damp,
                    center=center,
                    stats=sweep_stats,
                )
                center = (
                    k + 1,
                    (S[:next_bond, None] * V[:next_bond, :]).reshape(
                        next_bond, x.cores[k + 1].shape[1], x.bonds[k + 2]
                    ),
                )
                x.cores[k] = U[:, :next_bond].reshape(
                    x.bonds[k], x.cores[k].shape[1], 1, next_bond
                )
                x.bonds[k + 1] = next_bond
                x.orth_center = k + 1

        # right-to-left sweep
        for k in range(len(x) - 2, -1, -1):
            prepare_right_environment(k + 1, right_env_A, x, x, A)
            prepare_right_environment(k + 1, right_env_b, x, b)

            U, S, V, next_bond = _solve_local_two_site(
                k,
                x,
                A,
                b,
                left_env_A,
                right_env_A,
                left_env_b,
                right_env_b,
                settings,
                error_threshold=error_threshold,
                max_rank=max_rank,
                residual_damp=residual_damp,
                center=center,
                stats=sweep_stats,
            )
            center = (
                k,
                (U[:, :next_bond] * S[:next_bond]).reshape(
                    x.bonds[k], x.cores[k].shape[1], next_bond
                ),
            )
            x.cores[k + 1] = V[:next_bond, :].reshape(
                next_bond, x.cores[k + 1].shape[1], 1, x.bonds[k + 2]
            )
            x.bonds[k + 1] = next_bond
            if k == 0:
                x.cores[k] = (U[:, :next_bond] @ np.diag(S[:next_bond])).reshape(
                    x.bonds[k],
                    x.cores[k].shape[1],
                    1,
                    next_bond,
                )
            x.orth_center = k

        if monitor.record(sweep, x, max_bond=max(x.bonds), stats=sweep_stats):
            break

    x.max_rank = max(x.bonds)
    monitor.finish()

    if return_info:
        return x, monitor.info
    return x
