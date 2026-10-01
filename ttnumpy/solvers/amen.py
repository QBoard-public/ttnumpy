# Copyright (c) 2026 QBoard LLC. Licensed under the AGPLv3. See LICENSE.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Alternating Minimal Energy (AMEn) linear solver on tensor trains.

Extends the one-site ALS sweep with residual enrichment: a separate low-rank
train ``z ~ b - A x`` is maintained by projection and used to widen each
solved core, so the bond dimensions adapt instead of staying fixed. The z
machinery and the truncate-then-enrich rank control live here alongside the
solver.

Cost, for ``n`` sites of mode size ``d``, solution bond dimension ``r``
(enrichment included), operator bond dimension ``R``, right-hand-side bond
dimension ``r_b`` and residual bond dimension ``k`` (``kickrank``). A sweep
visits every site twice, and the local system at a site has ``N = r^2 d``
unknowns.

- Local solve, as in ALS: on the explicit matrix, used up to
  ``direct_solve_size`` unknowns, ``O(r^4 d^2 R + r^2 d^2 R^2)`` to build it,
  ``O(N^3)`` to factorize it and ``O(N^2)`` memory; matrix-free,
  ``O(r^3 d R + r^2 d^2 R^2)`` per Krylov iteration.
- Residual core and enrichment block: ``O(k (r^2 d R + r d^2 R^2 + k r d R))``.
- Truncation of the solved core: an SVD in ``O(r^3 d)``. With
  ``residual_tol`` set, choosing the rank applies the local operator to up to
  ``r`` rank-one terms.
- Environment updates after each step: ``O(r^3 d R + r^2 d^2 R^2)`` for the
  operator, ``O(r r_b d (r + r_b))`` for the right-hand side and
  ``O(k (r^2 d R + r d^2 R^2 + k r d R + r_b d (k + r_b)))`` for the
  residual. The environments of all sites hold
  ``O(n (r^2 R + r r_b + k r R + k r_b))`` numbers.
- Residual after a sweep, when residuals are tracked:
  ``O(n d (d r^2 R^2 + (r R + r_b)^3))`` to form and orthonormalize the train
  ``A x - b``.

Overall, a sweep takes ``O(n (r^6 d^3 + r^4 d^2 R + r^2 d^2 R^2))`` with
explicit local matrices, or ``O(n m (r^3 d R + r^2 d^2 R^2))`` with ``m``
applications of the local operator per step (the Krylov iterations, plus up
to ``r`` for residual-based truncation), plus
``O(n k (r^2 d R + r d^2 R^2 + k r d R))`` for the residual train,
``O(n (r + k) r_b d (r + k + r_b))`` for the right-hand side and, when
residuals are tracked, ``O(n d (d r^2 R^2 + (r R + r_b)^3))``. Memory is
``O(n (d r^2 + d k^2 + d^2 R^2 + d r_b^2 + r^2 R + r r_b + k r R + k r_b))``
for the trains and environments, plus ``O(r^4 d^2)`` for an explicit local
matrix and ``O(n d (r R + r_b)^2)`` for the train ``A x - b``.

Reference:
- Dolgov & Savostyanov, SIAM J. Sci. Comput. 36(5), 2014,
  https://doi.org/10.1137/140953289.
"""

from typing import Optional, Tuple, Union

import numpy as np
import scipy.linalg as lin
import scipy.sparse.linalg as spla

from ttnumpy import tensor_train
from ttnumpy.solvers.common import (
    LocalSolver,
    SolverInfo,
    TruncationMode,
    _choose_truncation_rank,
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
    local_rhs_one_site,
)


def init_residual_train(
    b: tensor_train.TensorTrain,
    kickrank: int,
    seed: Optional[int] = None,
) -> tensor_train.TensorTrain:
    """Random rank-``kickrank`` train, right-orthonormalised, to hold z.

    A random start is used rather than the initial residual ``b - A x``: the
    first fitting sweep corrects it, and it doubles as a random enrichment
    direction.

    Args:
        b: Right-hand side, used for the mode sizes.
        kickrank: Bond dimension of the residual train; bonds near the ends
            are capped by what the mode sizes can support.
        seed: Seed for the random cores, for reproducible runs.

    Returns:
        TensorTrain: right-orthonormal train with the mode sizes of ``b``.

    Raises:
        ValueError: If ``kickrank`` is not positive.
    """
    if kickrank <= 0:
        raise ValueError(f"kickrank must be a positive integer, got {kickrank}")
    z = tensor_train.rand(b.physical_dims(), ranks=kickrank, seed=seed)
    return z.orthonormalize(canonical="right")


def residual_core(
    ind: int,
    left_env_zb: list,
    right_env_zb: list,
    left_env_zAx: list,
    right_env_zAx: list,
    A: tensor_train.TensorTrain,
    b: tensor_train.TensorTrain,
    x: tensor_train.TensorTrain,
    x_core: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Local core of the residual train z ~ b - A x at site ``ind``.

    Requires the surrounding cores of ``z`` to be orthonormal frames, in
    which case the least-squares optimum is the projection of ``b - A x``
    onto that frame and no local system has to be solved. The two halves of
    the target are projected separately and subtracted.

    ``x_core`` overrides the stored core of ``x`` at ``ind``, indexed
    (left_bond, mode, right_bond); the AMEn sweep passes the freshly solved
    core so that the residual reflects the current local solution.

    Returns the core indexed (z_left_bond, mode, z_right_bond).
    """
    # b projected onto z's frame, reusing the one-site rhs contraction
    b_part = local_rhs_one_site(ind, left_env_zb, right_env_zb, b).reshape(
        left_env_zb[ind].shape[1],
        b.cores[ind].shape[1],
        right_env_zb[ind].shape[1],
    )
    if x_core is None:
        x_core = x.cores[ind][:, :, 0, :]
    # A x projected onto z's frame
    ax_part = np.einsum(
        "iaj,aklb,ilm,mbn->jkn",
        left_env_zAx[ind],
        A.cores[ind],
        x_core,
        right_env_zAx[ind],
        optimize="optimal",
    )
    return b_part - ax_part


def fit_residual_train(
    A: tensor_train.TensorTrain,
    b: tensor_train.TensorTrain,
    x: tensor_train.TensorTrain,
    z: tensor_train.TensorTrain,
    sweeps: int = 1,
) -> tensor_train.TensorTrain:
    """Fit the rank-limited train ``z`` to the residual ``b - A x`` by ALS.

    Sweeps forward and back over the cores of ``z``, replacing each with the
    projection of ``b - A x`` onto the surrounding frames. ``z`` must be
    right-orthonormal on entry (as produced by :func:`init_residual_train`)
    and is modified in place, ending right-orthonormal with its centre on
    core 0.

    Args:
        A: Operator in TTM format.
        b: Right-hand side in TT format.
        x: Current solution estimate in TT format.
        z: Residual train to refit, modified in place.
        sweeps: Number of forward/backward sweeps.

    Returns:
        TensorTrain: the refitted ``z`` (the same object that was passed in).
    """
    n_cores = len(z)
    right_env_zb = [None] * n_cores
    right_env_zAx = [None] * n_cores
    left_env_zb = [None] * n_cores
    left_env_zAx = [None] * n_cores

    # one-site steps consume the environment at their own index, so index 0
    # is needed too
    for i in range(n_cores - 1, -1, -1):
        prepare_right_environment(i, right_env_zb, z, b)
        prepare_right_environment(i, right_env_zAx, z, x, A)

    for _ in range(sweeps):
        for k in range(n_cores - 1):
            prepare_left_environment(k, left_env_zb, z, b)
            prepare_left_environment(k, left_env_zAx, z, x, A)
            core = residual_core(
                k, left_env_zb, right_env_zb, left_env_zAx, right_env_zAx, A, b, x
            )
            _shift_center_right(z, k, core)

        prepare_left_environment(n_cores - 1, left_env_zb, z, b)
        prepare_left_environment(n_cores - 1, left_env_zAx, z, x, A)

        for k in range(n_cores - 1, -1, -1):
            prepare_right_environment(k, right_env_zb, z, b)
            prepare_right_environment(k, right_env_zAx, z, x, A)
            core = residual_core(
                k, left_env_zb, right_env_zb, left_env_zAx, right_env_zAx, A, b, x
            )
            if k > 0:
                _shift_center_left(z, k, core)
            else:
                z.cores[0] = core.reshape(z.bonds[0], core.shape[1], 1, z.bonds[1])
                z.orth_center = 0

    z.max_rank = max(z.bonds)
    return z


def _truncate_core_right(
    core: np.ndarray,
    mode: TruncationMode,
    error_threshold: float,
    max_rank: Optional[int],
    operator: Optional[Union[np.ndarray, spla.LinearOperator]] = None,
    b_loc: Optional[np.ndarray] = None,
    truncation_tol: Optional[float] = None,
    residual_damp: float = 2.0,
    stats: Optional[_SweepStats] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """SVD-truncate the right bond of a solved core before enrichment.

    Splits ``core`` (left_bond, mode, right_bond) into a left-orthogonal
    factor (left_bond, mode, rank) and a carry (rank, right_bond) whose
    product is the truncated core. ``mode`` selects the truncation rule as in
    :func:`_choose_truncation_rank`; :data:`TruncationMode.RESIDUAL` needs the
    local ``operator`` and ``b_loc`` and truncates exactly as MALS does its
    supercore against ``truncation_tol``, one truncation's share of
    ``residual_tol``.
    """
    r_left, mode_size, r_right = core.shape
    exact_residual = (
        float(np.linalg.norm(operator @ core.ravel() - b_loc))
        if mode is TruncationMode.RESIDUAL
        else 0.0
    )
    U, S, Vh = lin.svd(
        core.reshape(r_left * mode_size, r_right),
        full_matrices=False,
        overwrite_a=True,
        check_finite=False,
        lapack_driver="gesvd",
    )
    rank = _choose_truncation_rank(
        mode,
        operator,
        b_loc,
        U,
        S,
        Vh,
        exact_residual,
        error_threshold,
        max_rank,
        truncation_tol,
        residual_damp,
        stats=stats,
    )
    return U[:, :rank].reshape(r_left, mode_size, rank), S[:rank, None] * Vh[:rank]


def _truncate_core_left(
    core: np.ndarray,
    mode: TruncationMode,
    error_threshold: float,
    max_rank: Optional[int],
    operator: Optional[Union[np.ndarray, spla.LinearOperator]] = None,
    b_loc: Optional[np.ndarray] = None,
    truncation_tol: Optional[float] = None,
    residual_damp: float = 2.0,
    stats: Optional[_SweepStats] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mirror of :func:`_truncate_core_right` for the backward sweep.

    Returns a right-orthogonal factor (rank, mode, right_bond) and a carry
    (left_bond, rank) whose product is the truncated core. ``mode`` selects
    the truncation rule exactly as in :func:`_truncate_core_right`.
    """
    r_left, mode_size, r_right = core.shape
    exact_residual = (
        float(np.linalg.norm(operator @ core.ravel() - b_loc))
        if mode is TruncationMode.RESIDUAL
        else 0.0
    )
    U, S, Vh = lin.svd(
        core.reshape(r_left, mode_size * r_right),
        full_matrices=False,
        overwrite_a=True,
        check_finite=False,
        lapack_driver="gesvd",
    )
    rank = _choose_truncation_rank(
        mode,
        operator,
        b_loc,
        U,
        S,
        Vh,
        exact_residual,
        error_threshold,
        max_rank,
        truncation_tol,
        residual_damp,
        stats=stats,
    )
    return Vh[:rank].reshape(rank, mode_size, r_right), U[:, :rank] * S[:rank]


def _enrich_right(
    x: tensor_train.TensorTrain,
    k: int,
    core: np.ndarray,
    carry: np.ndarray,
    block: np.ndarray,
    max_rank: Optional[int] = None,
    stats: Optional[_SweepStats] = None,
) -> None:
    """Widen core ``k``'s right bond with ``block``, preserving the tensor.

    ``core`` (left_bond, mode, rank) times ``carry`` (rank, old_right_bond)
    is the current value of core ``k``. ``block`` (left_bond, mode, rho) is
    appended along the right bond and the neighbour absorbs ``carry`` padded
    with ``rho`` zero rows, so the appended directions contribute nothing to
    the represented tensor. The centre moves to ``k + 1``.

    With ``max_rank`` set the widened bond is hard-capped: after the shift the
    solved core occupies the leading ``rank`` bonds and the enrichment the
    rest, whose neighbour rows are zero, so the surplus enrichment directions
    beyond ``max_rank`` (never the solution, since ``rank <= max_rank``) are
    dropped without changing the represented tensor.
    """
    enriched = np.concatenate([core, block], axis=2)
    zeros = np.zeros((block.shape[2], carry.shape[1]), dtype=carry.dtype)
    x.cores[k + 1] = np.tensordot(
        np.concatenate([carry, zeros], axis=0), x.cores[k + 1], axes=(1, 0)
    )
    _shift_center_right(x, k, enriched)
    capped = max_rank is not None and x.bonds[k + 1] > max_rank
    if capped:
        x.cores[k] = x.cores[k][:, :, :, :max_rank]
        x.cores[k + 1] = x.cores[k + 1][:max_rank]
        x.bonds[k + 1] = max_rank
    if stats is not None:
        stats.record_rank(capped)


def _enrich_left(
    x: tensor_train.TensorTrain,
    k: int,
    core: np.ndarray,
    carry: np.ndarray,
    block: np.ndarray,
    max_rank: Optional[int] = None,
    stats: Optional[_SweepStats] = None,
) -> None:
    """Mirror of :func:`_enrich_right` for the backward sweep.

    ``carry`` (old_left_bond, rank) times ``core`` (rank, mode, right_bond)
    is the current value of core ``k``; ``block`` (rho, mode, right_bond)
    widens the left bond and the left neighbour absorbs the zero-padded
    carry. The centre moves to ``k - 1``. ``max_rank`` hard-caps the widened
    bond exactly as in :func:`_enrich_right`.
    """
    enriched = np.concatenate([core, block], axis=0)
    zeros = np.zeros((carry.shape[0], block.shape[0]), dtype=carry.dtype)
    x.cores[k - 1] = np.tensordot(
        x.cores[k - 1], np.concatenate([carry, zeros], axis=1), axes=(3, 0)
    )
    _shift_center_left(x, k, enriched)
    capped = max_rank is not None and x.bonds[k] > max_rank
    if capped:
        x.cores[k] = x.cores[k][:max_rank]
        x.cores[k - 1] = x.cores[k - 1][:, :, :, :max_rank]
        x.bonds[k] = max_rank
    if stats is not None:
        stats.record_rank(capped)


def amen(
    A: tensor_train.TensorTrain,
    b: tensor_train.TensorTrain,
    x0: Optional[tensor_train.TensorTrain] = None,
    inplace: bool = False,
    kickrank: int = 4,
    error_threshold: float = 1e-7,
    max_rank: Optional[int] = None,
    max_sweeps: int = 1,
    residual_tol: Optional[float] = None,
    residual_damp: float = 2.0,
    stagnation_factor: Union[float, str, None] = "auto",
    local_solver: Union[str, LocalSolver] = "auto",
    direct_solve_size: int = 1000,
    local_rtol: Optional[float] = None,
    allow_dense_fallback: bool = False,
    symmetric: bool = False,
    return_info: bool = False,
    seed: Optional[int] = None,
) -> Union[tensor_train.TensorTrain, Tuple[tensor_train.TensorTrain, SolverInfo]]:
    """Alternating Minimal Energy (AMEn) method for solving Ax = b.

    One-site sweep as in :func:`als`, except that each solved core is
    SVD-truncated and then widened with a low-rank approximation of the
    residual ``b - A x``, so the bond dimensions adapt instead of staying
    fixed. The residual approximation is a separate rank-``kickrank`` train
    maintained by projection and refitted core by core as the sweep
    proceeds. The enrichment leaves the represented solution unchanged; it
    enlarges the subspace the next local solve optimises over.

    Args:
        A: Operator in TTM format.
        b: Right-hand side in TT format.
        x0: Initial guess in TT format. Defaults to an all-ones train with
            the ranks of ``b``; unlike :func:`als` a poor guess is fine,
            since enrichment grows the ranks as needed.
        inplace: If True, modify x0 in place instead of copying it.
        kickrank: Bond dimension of the residual train, and the maximum
            rank added to each bond per enrichment. Typical values 2-4.
        error_threshold: Relative SVD cutoff for the truncation of each solved
            core; only used when ``residual_tol`` is None. Singular values
            below ``error_threshold * sigma_1`` are dropped before the
            enrichment is appended.
        residual_damp: Allowed growth factor of the local residual due to
            truncation when ``residual_tol`` is set (reasonable values 2 to
            5); see :func:`mals` for when it has any effect. Must be >= 1.
            Ignored in the ``error_threshold`` cutoff mode.
        max_rank: Hard cap on every bond dimension, matching :func:`mals`.
            The solved core is truncated to at most ``max_rank`` and the
            enrichment then fills only the remaining slots below it, so a bond
            that reaches ``max_rank`` is enriched by less than ``kickrank``
            (by nothing once the solution alone fills the cap). None leaves
            the ranks unbounded.
        max_sweeps: Maximum number of sweeps. Ranks grow by at most
            2 * kickrank per sweep (less near ``max_rank``), so starting from
            a low-rank guess the solver needs several sweeps to build up the
            rank it needs.
        residual_tol: Target relative residual for early stopping, and (as in
            :func:`mals`) the switch to residual-based core truncation, which
            makes it the single accuracy knob and ignores ``error_threshold``.
            None disables both. The stopping test uses it whole; the
            truncations of a sweep share it, since their errors add in
            quadrature.
        stagnation_factor: Stop when a sweep improves the residual by less
            than this factor; must be in (0, 1]. "auto" (default) enables the
            check with factor 0.9 when residuals are tracked anyway
            (residual_tol set or return_info=True) and disables it
            otherwise. An explicit float always enables the check; None
            disables it.
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
            cost of building it; see :func:`mals`. False by default, which
            keeps the inexact iterate and the memory bound.
        symmetric: Declare A symmetric so "auto" can use MINRES.
        return_info: If True, also return a SolverInfo with per-sweep
            residuals, per-sweep maximum bond dimensions (``max_bonds``), the
            stop reason, and how the iterative local solves ended
            (``local_solves``, ``local_fallbacks``, ``local_inexact``).
        seed: Seed for the random initial residual train, for reproducible
            runs.

    Returns:
        TensorTrain: Solution x, or a tuple (x, SolverInfo) if return_info
        is True.

    Raises:
        ValueError: If ``kickrank`` or ``max_rank`` is not positive,
            ``residual_damp`` is below 1, or an argument shared with
            :func:`als` is invalid.
        DimensionMismatch: If A, b and x0 are not shape-compatible.

    Example:
        Solve a QTT-structured system from the default guess, letting the
        ranks grow until the residual target is met::

            x, info = amen(
                A, b, residual_tol=1e-8, max_sweeps=20,
                symmetric=True, return_info=True,
            )
            print(info.stop_reason, info.residuals[-1], info.max_bonds)

    Reference:
        Dolgov & Savostyanov, SIAM J. Sci. Comput. 36(5), 2014,
        https://doi.org/10.1137/140953289.
    """
    if kickrank <= 0:
        raise ValueError(f"kickrank must be a positive integer, got {kickrank}")
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
        truncation_steps=max(2 * (A.n_cores - 1), 1),
    )

    if x0 is None:
        x0 = tensor_train.ones(b.physical_dims(), ranks=b.bonds)

    _validate_solver_inputs(A, b, x0)

    x = x0 if inplace else x0.copy()

    if x.orth_center != 0:
        x.orthonormalize(canonical="right", inplace=True)

    n_cores = len(x)
    # residual-based core truncation (AMEn's analogue of MALS trick 2) needs
    # the local operator to survive the solve; plain sigma-cutoff does not
    residual_truncation = settings.truncation_mode is TruncationMode.RESIDUAL
    right_env_A = [None] * n_cores
    right_env_b = [None] * n_cores
    left_env_A = [None] * n_cores
    left_env_b = [None] * n_cores

    # projecting b and A x onto z's frame refits z; projecting z onto x's
    # frame produces the enrichment block
    z = init_residual_train(b, kickrank, seed=seed)
    fit_residual_train(A, b, x, z, sweeps=1)
    left_env_zb = [None] * n_cores
    right_env_zb = [None] * n_cores
    left_env_zAx = [None] * n_cores
    right_env_zAx = [None] * n_cores
    left_env_xz = [None] * n_cores
    right_env_xz = [None] * n_cores

    # a one-site system at site k consumes the right environment at k, so
    # unlike MALS the environment at index 0 is needed too
    for i in range(n_cores - 1, -1, -1):
        prepare_right_environment(i, right_env_A, x, x, A)
        prepare_right_environment(i, right_env_b, x, b)
        prepare_right_environment(i, right_env_zb, z, b)
        prepare_right_environment(i, right_env_zAx, z, x, A)
        prepare_right_environment(i, right_env_xz, x, z)

    monitor = _SweepMonitor(
        settings,
        "AMEn",
        A,
        b,
        x,
        error_threshold=error_threshold,
        max_rank=max_rank,
    )
    sweep_stats = _SweepStats(max_rank=max_rank)

    for sweep in range(settings.max_sweeps):

        # left-to-right sweep, leaving the centre on the last core
        for k in range(n_cores - 1):
            prepare_left_environment(k, left_env_A, x, x, A)
            prepare_left_environment(k, left_env_b, x, b)
            prepare_left_environment(k, left_env_zb, z, b)
            prepare_left_environment(k, left_env_zAx, z, x, A)
            prepare_left_environment(k, left_env_xz, x, z)
            u, operator, b_loc = _solve_local_one_site(
                k,
                x,
                A,
                b,
                left_env_A,
                right_env_A,
                left_env_b,
                right_env_b,
                settings,
                preserve_operator=residual_truncation,
                stats=sweep_stats,
            )
            core3d = u.reshape(x.bonds[k], x.cores[k].shape[1], x.bonds[k + 1])
            # z from the un-truncated solution, before the SVD may overwrite it
            z_new = residual_core(
                k,
                left_env_zb,
                right_env_zb,
                left_env_zAx,
                right_env_zAx,
                A,
                b,
                x,
                x_core=core3d,
            )
            core, carry = _truncate_core_right(
                core3d,
                settings.truncation_mode,
                error_threshold,
                max_rank,
                operator,
                b_loc,
                settings.truncation_tol,
                residual_damp,
                stats=sweep_stats,
            )
            # z's core rewritten in x's left frame
            block = np.einsum("ba,bin->ain", left_env_xz[k], z_new)
            _enrich_right(x, k, core, carry, block, max_rank, stats=sweep_stats)
            _shift_center_right(z, k, z_new)

        # the backward sweep starts on the last core, which needs its left
        # environment built from the just-orthogonalised neighbour
        prepare_left_environment(n_cores - 1, left_env_A, x, x, A)
        prepare_left_environment(n_cores - 1, left_env_b, x, b)
        prepare_left_environment(n_cores - 1, left_env_zb, z, b)
        prepare_left_environment(n_cores - 1, left_env_zAx, z, x, A)
        prepare_left_environment(n_cores - 1, left_env_xz, x, z)

        # right-to-left sweep
        for k in range(n_cores - 1, -1, -1):
            prepare_right_environment(k, right_env_A, x, x, A)
            prepare_right_environment(k, right_env_b, x, b)
            prepare_right_environment(k, right_env_zb, z, b)
            prepare_right_environment(k, right_env_zAx, z, x, A)
            prepare_right_environment(k, right_env_xz, x, z)
            u, operator, b_loc = _solve_local_one_site(
                k,
                x,
                A,
                b,
                left_env_A,
                right_env_A,
                left_env_b,
                right_env_b,
                settings,
                preserve_operator=residual_truncation,
                stats=sweep_stats,
            )
            core3d = u.reshape(x.bonds[k], x.cores[k].shape[1], x.bonds[k + 1])
            if k == 0:
                x.cores[0] = core3d.reshape(
                    x.bonds[0], x.cores[0].shape[1], 1, x.bonds[1]
                )
                x.orth_center = 0
                # leave z fitted and right-canonical for the next sweep
                z_new = residual_core(
                    0,
                    left_env_zb,
                    right_env_zb,
                    left_env_zAx,
                    right_env_zAx,
                    A,
                    b,
                    x,
                    x_core=core3d,
                )
                z.cores[0] = z_new.reshape(z.bonds[0], z_new.shape[1], 1, z.bonds[1])
                z.orth_center = 0
            else:
                # z from the un-truncated solution, before the SVD may
                # overwrite it
                z_new = residual_core(
                    k,
                    left_env_zb,
                    right_env_zb,
                    left_env_zAx,
                    right_env_zAx,
                    A,
                    b,
                    x,
                    x_core=core3d,
                )
                core, carry = _truncate_core_left(
                    core3d,
                    settings.truncation_mode,
                    error_threshold,
                    max_rank,
                    operator,
                    b_loc,
                    settings.truncation_tol,
                    residual_damp,
                    stats=sweep_stats,
                )
                block = np.tensordot(z_new, right_env_xz[k], axes=(2, 0))
                _enrich_left(x, k, core, carry, block, max_rank, stats=sweep_stats)
                _shift_center_left(z, k, z_new)

        if monitor.record(sweep, x, max_bond=max(x.bonds), stats=sweep_stats):
            break

    x.max_rank = max(x.bonds)
    monitor.finish()

    if return_info:
        return x, monitor.info
    return x
