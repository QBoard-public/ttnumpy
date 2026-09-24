# Copyright (c) 2026 QBoard LLC. Licensed under the AGPLv3. See LICENSE.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Two-site local algebra for the MALS sweep.

The local right-hand side and operator (explicit or matrix-free) of a
two-core projected system, the warm start read off the current supercore,
and the solve that assembles them, splits the solved supercore by SVD and
picks the truncation rank that adapts the shared bond.
"""

from typing import Optional, Tuple

import numpy as np
import scipy.linalg as lin
import scipy.sparse.linalg as spla

from ttnumpy import tensor_train
from ttnumpy.solvers.common import (
    TruncationMode,
    _choose_truncation_rank,
    _SolverSettings,
    _solve_local_system,
    _SweepStats,
)


def local_rhs_two_site(
    ind: int,
    left_env_b: list,
    right_env_b: list,
    b: tensor_train.TensorTrain,
) -> np.ndarray:
    """Right-hand side of the two-site local system at step ``ind``.

    Returns the flattened vector indexed (row_bond, row_mode_k,
    row_mode_{k+1}, right_bond), matching the row space of the local operator.
    """
    b_loc = np.einsum(
        "ai,ajb,bkc,cl->ijkl",
        left_env_b[ind],
        b.cores[ind][:, :, 0, :],
        b.cores[ind + 1][:, :, 0, :],
        right_env_b[ind + 1],
        optimize="optimal",
    )
    return b_loc.reshape(-1)


def local_explicit_matrix_two_site(
    ind: int,
    left_env_A: list,
    right_env_A: list,
    A: tensor_train.TensorTrain,
) -> np.ndarray:
    """Two-site local operator at step ``ind`` as an explicit matrix.

    Forms the full (rows x cols) matrix by contracting the left environment,
    both operator cores and the right environment. Rows are ordered
    (row_bond, row_mode_k, row_mode_{k+1}, right_bond) and columns
    (col_bond, col_mode_k, col_mode_{k+1}, right_bond). Contrast
    :func:`local_operator_two_site`, which never forms this matrix.
    """
    A_loc = np.einsum(
        "iaj,aklb,bmnc,ocp->ijklmnop",
        left_env_A[ind],
        A.cores[ind],
        A.cores[ind + 1],
        right_env_A[ind + 1],
        optimize="optimal",
    )
    r_left = left_env_A[ind].shape[0]
    r_right = right_env_A[ind + 1].shape[0]
    rows = r_left * A.row_modes[ind] * A.row_modes[ind + 1] * r_right
    cols = r_left * A.col_modes[ind] * A.col_modes[ind + 1] * r_right
    # reorder legs so all row indices precede all column indices
    return A_loc.transpose([1, 2, 4, 7, 0, 3, 5, 6]).reshape(rows, cols)


def local_operator_two_site(
    left_env: np.ndarray,
    A_k: np.ndarray,
    A_kp1: np.ndarray,
    right_env: np.ndarray,
    size: int,
) -> spla.LinearOperator:
    """Matrix-free two-site local operator B = Psi . A_k . A_{k+1} . Phi.

    Applying B is a short chain of small contractions instead of a matvec
    against the explicit matrix (section 3.4 of Oseledets & Dolgov 2012),
    which makes iterative local solvers affordable at large ranks.

    Returns:
        LinearOperator of shape (size, size) with matvec and matmat.
    """
    r_left = left_env.shape[0]
    r_right = right_env.shape[0]
    in_shape = (r_left, A_k.shape[2], A_kp1.shape[2], r_right)

    def matmat(block: np.ndarray) -> np.ndarray:
        # y indexed (i, l, n, o, z) with z the batch axis
        y = np.asarray(block).reshape(in_shape + (-1,))
        # contract the right environment over o: (i, l, n, z, c, p)
        t = np.tensordot(y, right_env, axes=(3, 0))
        # contract A_kp1 over (n, c): (i, l, z, p, b, m)
        t = np.tensordot(t, A_kp1, axes=((2, 4), (2, 3)))
        # contract A_k over (l, b): (i, z, p, m, a, k)
        t = np.tensordot(t, A_k, axes=((1, 4), (2, 3)))
        # contract the left environment over (i, a): (z, p, m, k, j)
        t = np.tensordot(t, left_env, axes=((0, 4), (0, 1)))
        # reorder to rows (j, k, m, p) with the batch axis last
        return t.transpose(4, 3, 2, 1, 0).reshape(size, -1)

    return spla.LinearOperator(
        (size, size),
        matvec=lambda vec: matmat(vec).ravel(),
        matmat=matmat,
        dtype=np.result_type(left_env, A_k, A_kp1, right_env),
    )


def local_warm_start(
    x: tensor_train.TensorTrain,
    ind: int,
    center: Optional[Tuple[int, np.ndarray]] = None,
) -> Optional[np.ndarray]:
    """Current two-site supercore, used as the initial guess for local solves.

    ``center`` is the non-orthogonal (site index, block) pair left behind by
    the previous decimation; it replaces the stored core at that site, whose
    bond dimensions are stale mid-sweep. Returns None when the shared bond of
    the two blocks is inconsistent (no valid supercore to warm-start from).
    """
    left = x.cores[ind][:, :, 0, :]
    right = x.cores[ind + 1][:, :, 0, :]
    if center is not None:
        center_index, center_block = center
        if center_index == ind:
            left = center_block
        elif center_index == ind + 1:
            right = center_block
    if left.shape[-1] != right.shape[0]:
        return None
    return np.tensordot(left, right, axes=(2, 0)).reshape(-1)


def _solve_local_two_site(
    k: int,
    x: tensor_train.TensorTrain,
    A: tensor_train.TensorTrain,
    b: tensor_train.TensorTrain,
    left_env_A: list,
    right_env_A: list,
    left_env_b: list,
    right_env_b: list,
    settings: _SolverSettings,
    *,
    error_threshold: float,
    max_rank: Optional[int],
    residual_damp: float,
    center: Optional[Tuple[int, np.ndarray]] = None,
    stats: Optional[_SweepStats] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Assemble and solve the two-site local system spanning sites k, k+1.

    Chooses between the direct solve on the explicit matrix and the iterative
    solve based on the local system size, then SVD-splits the supercore and
    returns the factors together with the selected truncation rank.
    ``settings.truncation_mode`` picks the rule: residual mode spends
    ``settings.truncation_tol`` per truncation, threshold mode discards
    singular values below ``error_threshold * sigma_1``.
    """
    svd_rows = x.bonds[k] * x.row_modes[k]
    svd_cols = x.row_modes[k + 1] * x.bonds[k + 2]
    size = svd_rows * svd_cols
    b_loc = local_rhs_two_site(k, left_env_b, right_env_b, b)

    residual_based = settings.truncation_mode is TruncationMode.RESIDUAL
    use_iterative = settings.use_iterative(size)
    if use_iterative:
        operator = local_operator_two_site(
            left_env_A[k],
            A.cores[k],
            A.cores[k + 1],
            right_env_A[k + 1],
            size,
        )
        warm_start = local_warm_start(x, k, center)
        if warm_start is not None and warm_start.size != size:
            warm_start = None

        # only a callable makes the dense operator reachable; without one
        # the solve stays matrix-free however badly it converges
        explicit_fallback = None
        if settings.allow_dense_fallback:

            def explicit_fallback():
                return local_explicit_matrix_two_site(k, left_env_A, right_env_A, A)

    else:
        operator = local_explicit_matrix_two_site(k, left_env_A, right_env_A, A)
        warm_start = None
        explicit_fallback = None

    # residual-based truncation reuses the operator, so it must survive the
    # solve; the exact residual is read off before the SVD overwrites u
    u, operator = _solve_local_system(
        operator,
        b_loc,
        overwrite_a=not residual_based,
        warm_start=warm_start,
        iterative_solver=settings.iterative_solver if use_iterative else None,
        local_rtol=settings.local_rtol,
        explicit_fallback=explicit_fallback,
        stats=stats,
    )
    exact_residual = (
        float(np.linalg.norm(operator @ u - b_loc)) if residual_based else 0.0
    )
    U, S, V = lin.svd(
        u.reshape(svd_rows, svd_cols),
        full_matrices=False,
        overwrite_a=True,
        check_finite=False,
        lapack_driver="gesvd",
    )
    next_bond = _choose_truncation_rank(
        settings.truncation_mode,
        operator,
        b_loc,
        U,
        S,
        V,
        exact_residual,
        error_threshold,
        max_rank,
        settings.truncation_tol,
        residual_damp,
        stats=stats,
    )
    return U, S, V, next_bond
