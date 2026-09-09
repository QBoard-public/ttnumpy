"""Alternating Least Squares (ALS) linear solver on tensor trains.

Reference:
- Dolgov & Savostyanov, SIAM J. Sci. Comput. 36(5), 2014,
  https://doi.org/10.1137/140953289.
"""

from typing import Optional, Tuple, Union

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
from ttnumpy.solvers.one_site import (
    _shift_center_left,
    _shift_center_right,
    _solve_local_one_site,
)


def als(
    A: tensor_train.TensorTrain,
    b: tensor_train.TensorTrain,
    x0: Optional[tensor_train.TensorTrain] = None,
    inplace: bool = False,
    max_sweeps: int = 4,
    residual_tol: Optional[float] = None,
    stagnation_factor: Union[float, str, None] = "auto",
    local_solver: Union[str, LocalSolver] = "auto",
    direct_solve_size: int = 1000,
    local_rtol: Optional[float] = None,
    allow_dense_fallback: bool = False,
    symmetric: bool = False,
    return_info: bool = False,
) -> Union[tensor_train.TensorTrain, Tuple[tensor_train.TensorTrain, SolverInfo]]:
    """Alternating Least Squares (ALS) method for solving Ax = b.

    Sweeps over single cores, solving a projected system for each and moving
    the orthogonality centre along. The bond dimensions never grow, so ``x0``
    must already carry enough rank to represent the solution; :func:`amen`
    adds the residual enrichment that lifts this restriction.

    Local systems are solved directly (LU on the explicit matrix) while
    small; above ``direct_solve_size`` unknowns they are solved matrix-free
    with a warm-started Krylov method held to ``local_rtol``.

    Args:
        A: Operator in TTM format.
        b: Right-hand side in TT format.
        x0: Initial guess in TT format; its ranks are kept throughout.
            Defaults to an all-ones train with the ranks of ``b``.
        inplace: If True, modify x0 in place instead of copying it.
        max_sweeps: Maximum number of sweeps.
        residual_tol: Target relative residual for early stopping. None
            disables it (and, with it, the ``local_rtol`` derivation below).
        stagnation_factor: Stop when a sweep improves the residual by less
            than this factor; must be in (0, 1]. "auto" (default) enables the
            check with factor 0.9 when residuals are tracked anyway
            (residual_tol set or return_info=True) and disables it otherwise.
            An explicit float always enables the check; None disables it.
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
            ``local_rtol`` is redone exactly on the explicit matrix, at the
            cost of building it. False by default, which
            keeps the inexact iterate and the memory bound.
        symmetric: Declare A symmetric so "auto" can use MINRES.
        return_info: If True, also return a SolverInfo with per-sweep
            residuals, the stop reason, and how the iterative local solves
            ended (``local_solves``, ``local_fallbacks``, ``local_inexact``).

    Returns:
        TensorTrain: Solution x, or a tuple (x, SolverInfo) if return_info
        is True.
    """
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
    )

    if x0 is None:
        x0 = tensor_train.ones(b.physical_dims(), ranks=b.bonds)

    _validate_solver_inputs(A, b, x0)

    x = x0 if inplace else x0.copy()

    if x.orth_center != 0:
        x.orthonormalize(canonical="right", inplace=True)

    n_cores = len(x)
    right_env_A = [None] * n_cores
    right_env_b = [None] * n_cores
    left_env_A = [None] * n_cores
    left_env_b = [None] * n_cores

    # a one-site system at site k consumes the right environment at k,
    # so the environment at index 0 is needed too
    for i in range(n_cores - 1, -1, -1):
        prepare_right_environment(i, right_env_A, x, x, A)
        prepare_right_environment(i, right_env_b, x, b)

    monitor = _SweepMonitor(settings, "ALS", A, b, x)
    sweep_stats = _SweepStats()

    for sweep in range(settings.max_sweeps):

        # left-to-right sweep, leaving the centre on the last core
        for k in range(n_cores - 1):
            prepare_left_environment(k, left_env_A, x, x, A)
            prepare_left_environment(k, left_env_b, x, b)
            u, _, _ = _solve_local_one_site(
                k,
                x,
                A,
                b,
                left_env_A,
                right_env_A,
                left_env_b,
                right_env_b,
                settings,
                stats=sweep_stats,
            )
            core = u.reshape(x.bonds[k], x.cores[k].shape[1], x.bonds[k + 1])
            _shift_center_right(x, k, core)

        # the backward sweep starts on the last core, which needs its left
        # environment built from the just-orthogonalised neighbour
        prepare_left_environment(n_cores - 1, left_env_A, x, x, A)
        prepare_left_environment(n_cores - 1, left_env_b, x, b)

        # right-to-left sweep
        for k in range(n_cores - 1, -1, -1):
            prepare_right_environment(k, right_env_A, x, x, A)
            prepare_right_environment(k, right_env_b, x, b)
            u, _, _ = _solve_local_one_site(
                k,
                x,
                A,
                b,
                left_env_A,
                right_env_A,
                left_env_b,
                right_env_b,
                settings,
                stats=sweep_stats,
            )
            if k == 0:
                x.cores[0] = u.reshape(x.bonds[0], x.cores[0].shape[1], 1, x.bonds[1])
                x.orth_center = 0
            else:
                core = u.reshape(x.bonds[k], x.cores[k].shape[1], x.bonds[k + 1])
                _shift_center_left(x, k, core)

        if monitor.record(sweep, x, stats=sweep_stats):
            break

    x.max_rank = max(x.bonds)
    monitor.finish()

    if return_info:
        return x, monitor.info
    return x
