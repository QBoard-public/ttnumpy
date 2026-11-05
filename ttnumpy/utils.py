# Copyright (c) 2026 QBoard LLC. Licensed under the AGPLv3. See LICENSE.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Module for utility functions on tensor trains."""

import logging
from typing import List, Optional, Tuple

import numpy as np
from scipy.linalg import LinAlgError, qr, rq, svd

from ttnumpy.logger_config import PACKAGE_LOGGER_NAME

logger = logging.getLogger(PACKAGE_LOGGER_NAME)


def tensor_norm(tensor: np.ndarray) -> float:
    """Frobenius tensor norm."""
    return np.linalg.norm(tensor)


def truncated_bond(
    singular_values: np.ndarray, delta: float, max_rank: Optional[int] = None
) -> int:
    """Smallest bond dimension whose discarded tail stays within ``delta``.

    Returns the smallest ``r`` satisfying

    .. math::

        \\sqrt{\\sum_{i > r} \\sigma_i^2} \\le \\delta,

    floored at 1 and capped by ``max_rank``. This is the criterion the
    TT-SVD error bound assumes: the Frobenius norm of what is thrown away
    is what enters the bound, so bounding it per split bounds the global
    error by the sum in quadrature over the splits.

    Singular values at or below ``len(singular_values) * eps *
    singular_values[0]`` count as zero, where ``eps`` is the machine epsilon
    of their dtype (float64 for integers), so ``delta=0`` drops only the
    rounding noise of that precision.

    Args:
        singular_values: Singular values in descending order.
        delta: Largest acceptable Frobenius norm of the discarded tail.
        max_rank: Maximum rank after truncation.

    Raises:
        ValueError: If max_rank is not a positive integer.

    Returns:
        Bond dimension after truncation.
    """
    if max_rank is not None and max_rank <= 0:
        raise ValueError("max_rank must be a positive integer.")
    values = np.asarray(singular_values)
    dtype = values.dtype if np.issubdtype(values.dtype, np.inexact) else np.float64
    values = values.astype(float)
    # LAPACK reports a zero singular value anywhere between eps * sigma_max and
    # exactly zero, with eps of the precision it worked in, so with delta=0 an
    # untouched tail makes the bond follow the build rather than the matrix
    if values.size:
        cutoff = np.finfo(dtype).eps * values.size * values[0]
        values = np.where(values <= cutoff, 0.0, values)
    squares = values**2
    tail = np.concatenate([np.cumsum(squares[::-1])[::-1], [0.0]])
    bond = max(1, int(np.searchsorted(-tail, -(delta * delta), side="left")))
    if max_rank is not None:
        bond = min(bond, max_rank)
    return bond


def left_orthonormalize(
    cores: List[np.ndarray], start: int, end: int
) -> List[np.ndarray]:
    """Left orthonormalize tensor train cores.

    Args:
        cores: Tensor train cores.
        start: Start position for orthonormalization.
        end: End position for orthonormalization.

    Returns:
        Orthonormalized tensor train cores.
    """
    for i in range(start, end):
        shape = cores[i].shape
        Q, R = qr(
            cores[i].reshape(
                (
                    shape[0] * shape[1] * shape[2],
                    shape[3],
                )
            ),
            mode="economic",
        )
        cores[i] = Q.reshape((shape[0], shape[1], shape[2], -1))
        cores[i + 1] = np.tensordot(R, cores[i + 1], axes=(1, 0))
    return cores


def right_orthonormalize(
    cores: List[np.ndarray], start: int, end: int
) -> List[np.ndarray]:
    """Right orthonormalize tensor train cores.

    Args:
        cores: Tensor train cores.
        start: Start position for orthonormalization.
        end: End position for orthonormalization.

    Returns:
        Orthonormalized tensor train cores.
    """
    for i in range(start, end, -1):
        shape = cores[i].shape
        R, Q = rq(
            cores[i].reshape(
                (
                    shape[0],
                    shape[1] * shape[2] * shape[3],
                )
            ),
            mode="economic",
        )
        cores[i] = Q.reshape((-1, shape[1], shape[2], shape[3]))
        cores[i - 1] = np.tensordot(cores[i - 1], R, axes=(3, 0))
    return cores


def _svd_with_fallback(
    matrix: np.ndarray, core: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Thin SVD of an unfolding of core ``core``, with ``gesvd`` if ``gesdd`` fails.

    Raises:
        ValueError: If ``gesdd`` does not converge on a matrix with NaN or
            infinite entries.
    """
    # without overwrite_a the matrix is still intact for the gesvd attempt
    try:
        return svd(matrix, full_matrices=False, check_finite=False)
    except LinAlgError as error:
        # NaN or inf entries can end in the same error; gesvd does not
        # decompose them either
        if not np.isfinite(matrix).all():
            raise ValueError(f"Core {core} has NaN or infinite entries") from error
        logger.warning(
            "SVD of core %d (%d x %d unfolding) did not converge with LAPACK "
            "gesdd; repeating it with gesvd",
            core,
            *matrix.shape,
        )
        return svd(
            matrix, full_matrices=False, check_finite=False, lapack_driver="gesvd"
        )


def left_svd(
    cores: List[np.ndarray],
    error: float,
    max_rank: Optional[int] = None,
) -> List[np.ndarray]:
    """Left svd truncation of tensor train cores.

    Each unfolding is decomposed with LAPACK ``gesdd``. If ``gesdd`` does not
    converge, the unfolding is decomposed again with ``gesvd`` and a warning is
    logged.

    Args:
        cores: Tensor train cores.
        error: Truncation threshold.
        max_rank: Maximum rank after truncation.

    Raises:
        ValueError: If ``gesdd`` does not converge on an unfolding with NaN or
            infinite entries.

    Returns:
        Truncated tensor train cores.
    """
    norm = tensor_norm(
        cores[0].reshape((cores[0].shape[1] * cores[0].shape[2], cores[0].shape[3]))
    )
    delta = error / np.sqrt(len(cores) - 1) * norm  # delta for svd truncation
    bond = 1
    for i in range(len(cores) - 1):
        shape = cores[i].shape
        physical_indexes = (shape[1], shape[2])
        matrix = cores[i].reshape(
            (
                bond * shape[1] * shape[2],
                shape[3],
            )
        )
        U, S, V = _svd_with_fallback(matrix, i)
        next_bond = truncated_bond(S, delta, max_rank)
        U, C = (
            U[:, :next_bond].reshape((bond,) + physical_indexes + (next_bond,)),
            np.diag(S[:next_bond]) @ V[:next_bond, :],
        )
        cores[i] = U
        next_bond = C.shape[0]
        cores[i + 1] = np.tensordot(C, cores[i + 1], axes=(1, 0))
        bond = next_bond
    return cores


def right_svd(
    cores: List[np.ndarray],
    error: float,
    max_rank: Optional[int] = None,
) -> List[np.ndarray]:
    """Right svd truncation of tensor train cores.

    Each unfolding is decomposed with LAPACK ``gesdd``. If ``gesdd`` does not
    converge, the unfolding is decomposed again with ``gesvd`` and a warning is
    logged.

    Args:
        cores: Tensor train cores.
        error: Truncation threshold.
        max_rank: Maximum rank after truncation.

    Raises:
        ValueError: If ``gesdd`` does not converge on an unfolding with NaN or
            infinite entries.

    Returns:
        Truncated tensor train cores.
    """
    norm = tensor_norm(
        cores[-1].reshape((cores[-1].shape[0], cores[-1].shape[1] * cores[-1].shape[2]))
    )
    delta = error / np.sqrt(len(cores) - 1) * norm  # delta for svd truncation
    bond = 1
    for i in range(len(cores) - 1, 0, -1):
        shape = cores[i].shape
        physical_indexes = (shape[1], shape[2])
        matrix = cores[i].reshape(
            (
                shape[0],
                shape[1] * shape[2] * bond,
            )
        )
        U, S, V = _svd_with_fallback(matrix, i)
        next_bond = truncated_bond(S, delta, max_rank)
        C, V = (
            U[:, :next_bond] @ np.diag(S[:next_bond]),
            V[:next_bond, :].reshape((next_bond,) + physical_indexes + (bond,)),
        )
        cores[i] = V
        cores[i - 1] = np.tensordot(cores[i - 1], C, axes=(3, 0))
        bond = next_bond
    return cores
