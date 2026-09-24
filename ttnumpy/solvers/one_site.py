# Copyright (c) 2026 QBoard LLC. Licensed under the AGPLv3. See LICENSE.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""One-site local algebra shared by the ALS and AMEn sweeps.

The local right-hand side and operator (explicit or matrix-free) of a
single-core projected system, the solve that assembles and dispatches them,
and the QR centre-shift helpers that store a solved core orthogonal while
moving the orthogonality centre to the neighbour.
"""

from typing import Optional, Tuple, Union

import numpy as np
import scipy.linalg as lin
import scipy.sparse.linalg as spla

from ttnumpy import tensor_train
from ttnumpy.solvers.common import (
    _SolverSettings,
    _solve_local_system,
    _SweepStats,
)


def local_rhs_one_site(
    ind: int,
    left_env_b: list,
    right_env_b: list,
    b: tensor_train.TensorTrain,
) -> np.ndarray:
    """Right-hand side of the one-site local system at step ``ind``.

    Returns the flattened vector indexed (row_bond, row_mode_k, right_bond),
    matching the row space of the local operator. Both environments are taken
    at ``ind``; ``right_env_b[ind]`` covers sites ``ind + 1`` onwards.
    """
    b_loc = np.einsum(
        "ai,ajc,cl->ijl",
        left_env_b[ind],
        b.cores[ind][:, :, 0, :],
        right_env_b[ind],
        optimize="optimal",
    )
    return b_loc.reshape(-1)


def local_explicit_matrix_one_site(
    ind: int,
    left_env_A: list,
    right_env_A: list,
    A: tensor_train.TensorTrain,
) -> np.ndarray:
    """One-site local operator at step ``ind`` as an explicit matrix.

    Forms the full (rows x cols) matrix by contracting the left environment,
    the operator core and the right environment. Rows are ordered
    (row_bond, row_mode_k, right_bond) and columns (col_bond, col_mode_k,
    right_bond). Both environments are taken at ``ind``. Contrast
    :func:`local_operator_one_site`, which never forms this matrix.
    """
    A_loc = np.einsum(
        "iaj,aklb,mbn->ijklmn",
        left_env_A[ind],
        A.cores[ind],
        right_env_A[ind],
        optimize="optimal",
    )
    r_left = left_env_A[ind].shape[0]
    r_right = right_env_A[ind].shape[0]
    rows = r_left * A.row_modes[ind] * r_right
    cols = r_left * A.col_modes[ind] * r_right
    # reorder legs so all row indices precede all column indices
    return A_loc.transpose([1, 2, 5, 0, 3, 4]).reshape(rows, cols)


def local_operator_one_site(
    left_env: np.ndarray,
    A_k: np.ndarray,
    right_env: np.ndarray,
    size: int,
) -> spla.LinearOperator:
    """Matrix-free one-site local operator B = Psi . A_k . Phi.

    One-site analogue of ``local_operator_two_site``, applying B as a chain of
    small contractions instead of a matvec against the explicit matrix.

    Returns:
        LinearOperator of shape (size, size) with matvec and matmat.
    """
    r_left = left_env.shape[0]
    r_right = right_env.shape[0]
    in_shape = (r_left, A_k.shape[2], r_right)

    # Pairwise tensordot calls rather than one einsum: as in the two-site
    # operator, einsum's path optimizer degrades badly here.
    def matmat(block: np.ndarray) -> np.ndarray:
        # y indexed (i, l, m, z) with z the batch axis
        y = np.asarray(block).reshape(in_shape + (-1,))
        # contract the right environment over m: (i, l, z, b, n)
        t = np.tensordot(y, right_env, axes=(2, 0))
        # contract the operator core over (l, b): (i, z, n, a, k)
        t = np.tensordot(t, A_k, axes=((1, 3), (2, 3)))
        # contract the left environment over (i, a): (z, n, k, j)
        t = np.tensordot(t, left_env, axes=((0, 3), (0, 1)))
        # reorder to rows (j, k, n) with the batch axis last
        return t.transpose(3, 2, 1, 0).reshape(size, -1)

    return spla.LinearOperator(
        (size, size),
        matvec=lambda vec: matmat(vec).ravel(),
        matmat=matmat,
        dtype=np.result_type(left_env, A_k, right_env),
    )


def _solve_local_one_site(
    k: int,
    x: tensor_train.TensorTrain,
    A: tensor_train.TensorTrain,
    b: tensor_train.TensorTrain,
    left_env_A: list,
    right_env_A: list,
    left_env_b: list,
    right_env_b: list,
    settings: _SolverSettings,
    preserve_operator: bool = False,
    stats: Optional[_SweepStats] = None,
) -> Tuple[np.ndarray, Union[np.ndarray, spla.LinearOperator], np.ndarray]:
    """Assemble and solve the one-site local system at site ``k``.

    Chooses between the direct solve on the explicit matrix and the iterative
    solve based on the local system size, then returns the solution (a flat
    vector indexed (left_bond, mode_k, right_bond)) together with the
    operator that produced it and the local right-hand side. The operator and
    ``b_loc`` let AMEn measure the residual for residual-based truncation;
    pass ``preserve_operator`` there so a direct solve does not overwrite it.
    """
    size = x.bonds[k] * x.row_modes[k] * x.bonds[k + 1]
    b_loc = local_rhs_one_site(k, left_env_b, right_env_b, b)

    use_iterative = settings.use_iterative(size)
    if use_iterative:
        operator = local_operator_one_site(
            left_env_A[k],
            A.cores[k],
            right_env_A[k],
            size,
        )
        warm_start = x.cores[k][:, :, 0, :].reshape(-1)
        if warm_start.size != size:
            warm_start = None

        # only a callable makes the dense operator reachable; without one
        # the solve stays matrix-free however badly it converges
        explicit_fallback = None
        if settings.allow_dense_fallback:

            def explicit_fallback():
                return local_explicit_matrix_one_site(k, left_env_A, right_env_A, A)

    else:
        operator = local_explicit_matrix_one_site(k, left_env_A, right_env_A, A)
        warm_start = None
        explicit_fallback = None

    u, operator = _solve_local_system(
        operator,
        b_loc,
        overwrite_a=not preserve_operator,
        warm_start=warm_start,
        iterative_solver=settings.iterative_solver if use_iterative else None,
        local_rtol=settings.local_rtol,
        explicit_fallback=explicit_fallback,
        stats=stats,
    )
    return u, operator, b_loc


def _shift_center_right(x: tensor_train.TensorTrain, k: int, core: np.ndarray) -> None:
    """Store ``core`` at site ``k`` left-orthogonal, centre moving to ``k + 1``.

    The QR factor is absorbed by the right neighbour, so the represented
    tensor is unchanged. ``core`` is indexed (left_bond, mode, right_bond).
    """
    mode = core.shape[1]
    Q, R = lin.qr(core.reshape(-1, core.shape[-1]), mode="economic")
    new_bond = Q.shape[1]
    x.cores[k] = Q.reshape(core.shape[0], mode, 1, new_bond)
    x.cores[k + 1] = np.tensordot(R, x.cores[k + 1], axes=(1, 0))
    x.bonds[k + 1] = new_bond
    x.orth_center = k + 1


def _shift_center_left(x: tensor_train.TensorTrain, k: int, core: np.ndarray) -> None:
    """Store ``core`` at site ``k`` right-orthogonal, centre moving to ``k - 1``.

    Mirror of :func:`_shift_center_right`: the triangular factor is absorbed
    by the left neighbour, leaving the represented tensor unchanged.
    """
    mode = core.shape[1]
    right_bond = core.shape[-1]
    Q, R = lin.qr(core.reshape(core.shape[0], -1).T, mode="economic")
    new_bond = Q.shape[1]
    x.cores[k] = Q.T.reshape(new_bond, mode, 1, right_bond)
    x.cores[k - 1] = np.tensordot(x.cores[k - 1], R.T, axes=(3, 0))
    x.bonds[k] = new_bond
    x.orth_center = k - 1
