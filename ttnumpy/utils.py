# Copyright (c) 2026 QBoard LLC. Licensed under the AGPLv3. See LICENSE.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Module for utility functions on tensor trains."""

import logging
from typing import List, Optional, Tuple

import numpy as np
from scipy.linalg import LinAlgError, qr, svd

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
        values[values <= cutoff] = 0.0
    tail = np.cumsum((values**2)[::-1])[::-1]
    bond = max(1, int(np.count_nonzero(tail > delta * delta)))
    if max_rank is not None:
        bond = min(bond, max_rank)
    return bond


def _multiply_left(matrix: np.ndarray, core: np.ndarray) -> np.ndarray:
    """Contract ``matrix`` (m, r) with the first axis of ``core`` (r, ...)."""
    return (matrix @ core.reshape(core.shape[0], -1)).reshape(
        (matrix.shape[0],) + core.shape[1:]
    )


def _multiply_right(core: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Contract the last axis of ``core`` (..., r) with ``matrix`` (r, m)."""
    return (core.reshape(-1, core.shape[-1]) @ matrix).reshape(
        core.shape[:-1] + (matrix.shape[1],)
    )


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
        cores[i + 1] = _multiply_left(R, cores[i + 1])
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
        Q, R = qr(
            cores[i].reshape((shape[0], shape[1] * shape[2] * shape[3])).T,
            mode="economic",
        )
        cores[i] = Q.T.reshape((-1, shape[1], shape[2], shape[3]))
        cores[i - 1] = _multiply_right(cores[i - 1], R.T)
    return cores


def left_randomized_orthonormalize(
    cores: List[np.ndarray], rank: int, rng: np.random.Generator
) -> List[np.ndarray]:
    """Left orthonormalize tensor train cores on a random sketch of the train.

    Randomize-then-Orthogonalize of Al Daas et al., SIAM J. Sci. Comput. 45
    (2023), A74-A95. The train is contracted from the right with a random
    tensor train of bonds at most ``rank``. Each core but the last is
    then replaced by the orthonormal factor of a QR factorization of its
    unfolding times that contraction. The last core takes the remainder.

    The result is the orthogonal projection of the train onto the span of the
    new left-orthonormal cores, with bonds at most ``rank``. When no unfolding
    of the train has rank above ``rank``, it equals the train with a rounding
    error that grows with the number of cores; otherwise its distance to the
    train has no bound.

    For ``d`` cores with bonds ``r`` and ``n`` entries per physical index, the
    cost is ``O(d n r^2 l)`` with ``l = rank``. :func:`left_orthonormalize`
    costs ``O(d n r^3)``.

    Args:
        cores: Tensor train cores.
        rank: Bound for the bonds of the result.
        rng: Generator the random tensor train is drawn from.

    Raises:
        ValueError: If ``rank`` is not positive.

    Returns:
        Left-orthonormal tensor train cores.
    """
    if rank < 1:
        raise ValueError(f"rank must be positive, but got {rank}")
    count = len(cores)
    real = np.finfo(np.result_type(*cores, np.float32)).dtype

    # sketches[i]: the cores from i on, contracted with the random train
    sketches = [None] * count + [np.ones((1, 1), dtype=real)]
    for i in range(count - 1, 0, -1):
        contracted = _multiply_right(cores[i], sketches[i + 1])
        contracted = contracted.reshape(contracted.shape[0], -1)
        bond = min(rank, *contracted.shape)
        random_core = rng.standard_normal((contracted.shape[1], bond), dtype=real)
        sketches[i] = contracted @ random_core
        scale = np.abs(sketches[i]).max()
        if scale > 0:
            sketches[i] /= scale

    for i in range(count - 1):
        shape = cores[i].shape
        matrix = cores[i].reshape((shape[0] * shape[1] * shape[2], shape[3]))
        Q, _ = qr(matrix @ sketches[i + 1], mode="economic")
        cores[i] = Q.reshape((shape[0], shape[1], shape[2], -1))
        cores[i + 1] = _multiply_left(Q.conj().T @ matrix, cores[i + 1])
    return cores


def right_randomized_orthonormalize(
    cores: List[np.ndarray], rank: int, rng: np.random.Generator
) -> List[np.ndarray]:
    """Right orthonormalize tensor train cores on a random sketch of the train.

    Mirror image of :func:`left_randomized_orthonormalize`: the train is
    contracted from the left, the cores are replaced from the last to the
    second and the first core takes the remainder.

    Args:
        cores: Tensor train cores.
        rank: Bound for the bonds of the result.
        rng: Generator the random tensor train is drawn from.

    Raises:
        ValueError: If ``rank`` is not positive.

    Returns:
        Right-orthonormal tensor train cores.
    """
    mirrored = [core.transpose(3, 1, 2, 0) for core in reversed(cores)]
    mirrored = left_randomized_orthonormalize(mirrored, rank, rng)
    cores[:] = [
        np.ascontiguousarray(core.transpose(3, 1, 2, 0)) for core in reversed(mirrored)
    ]
    return cores


def _hadamard_times_sketch(
    lhs: np.ndarray, rhs: np.ndarray, sketch: np.ndarray
) -> np.ndarray:
    """Contract a core of an element-wise product with ``sketch`` on its right bond.

    The cores of the two factors have shapes (r, m, n, r') and (s, m, n, s'),
    ``sketch`` has (r', s', l). The result has shape (r * s, m * n * l). The
    core of the product, of shape (r * s, m, n, r' * s'), is not formed.
    """
    bond, modes, next_bond = lhs.shape[0], lhs.shape[1] * lhs.shape[2], lhs.shape[3]
    half = lhs.reshape(-1, next_bond) @ sketch.reshape(next_bond, -1)
    half = half.reshape(bond, modes, rhs.shape[3], -1).transpose(1, 0, 3, 2)
    half = half.reshape(modes, -1, rhs.shape[3])
    full = half @ rhs.reshape(rhs.shape[0], modes, -1).transpose(1, 2, 0)
    full = full.reshape(modes, bond, -1, rhs.shape[0]).transpose(1, 3, 0, 2)
    return full.reshape(bond * rhs.shape[0], -1)


def _carry_times_hadamard(
    carry: np.ndarray, lhs: np.ndarray, rhs: np.ndarray
) -> np.ndarray:
    """Contract ``carry`` with a core of an element-wise product on its left bond.

    ``carry`` has shape (l, r, s), the cores of the two factors have
    (r, m, n, r') and (s, m, n, s'). The result has shape
    (l * m * n, r' * s'). The core of the product, of shape
    (r * s, m, n, r' * s'), is not formed.
    """
    rank, bond, modes = carry.shape[0], lhs.shape[0], lhs.shape[1] * lhs.shape[2]
    half = carry.transpose(0, 2, 1).reshape(-1, bond) @ lhs.reshape(bond, -1)
    half = half.reshape(rank, rhs.shape[0], modes, -1).transpose(2, 0, 3, 1)
    half = half.reshape(modes, -1, rhs.shape[0])
    full = half @ rhs.reshape(rhs.shape[0], modes, -1).transpose(1, 0, 2)
    full = full.reshape(modes, rank, lhs.shape[3], -1).transpose(1, 0, 2, 3)
    return full.reshape(rank * modes, -1)


def randomized_hadamard(
    lhs_cores: List[np.ndarray],
    rhs_cores: List[np.ndarray],
    rank: int,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """Left-orthonormal cores of the element-wise product of two tensor trains.

    HaTT-1 of Sun, Huang, Xiao and Yang, arXiv:2410.04385. It is
    :func:`left_randomized_orthonormalize` of the train whose cores are the
    products of the cores of the two trains, with bonds the products of their
    bonds. Those cores are not formed: each contraction with them is a
    contraction with one core of each train.

    The result is the orthogonal projection of the product onto the span of
    the returned left-orthonormal cores, with bonds at most ``rank``. When no
    unfolding of the product has rank above ``rank``, it equals the product
    with a rounding error that grows with the number of cores; otherwise its
    distance to the product has no bound.

    For ``d`` cores with bonds ``r`` and ``s`` in the two trains and ``n``
    entries per physical index, the cost is ``O(d n r s l (r + s + l))`` with
    ``l = rank``. A core of the product holds ``O(n r^2 s^2)`` numbers.

    Args:
        lhs_cores: Cores of the first tensor train.
        rhs_cores: Cores of the second tensor train, with the modes of the
            first.
        rank: Bound for the bonds of the result.
        rng: Generator the random tensor train is drawn from.

    Raises:
        ValueError: If ``rank`` is not positive.

    Returns:
        Left-orthonormal tensor train cores.
    """
    if rank < 1:
        raise ValueError(f"rank must be positive, but got {rank}")
    count = len(lhs_cores)
    real = np.finfo(np.result_type(*lhs_cores, *rhs_cores, np.float32)).dtype

    # sketches[i]: the product cores from i on, contracted with the random train
    sketches = [None] * count + [np.ones((1, 1, 1), dtype=real)]
    for i in range(count - 1, 0, -1):
        contracted = _hadamard_times_sketch(lhs_cores[i], rhs_cores[i], sketches[i + 1])
        bond = min(rank, *contracted.shape)
        random_core = rng.standard_normal((contracted.shape[1], bond), dtype=real)
        sketch = contracted @ random_core
        scale = np.abs(sketch).max()
        if scale > 0:
            sketch /= scale
        sketches[i] = sketch.reshape(lhs_cores[i].shape[0], rhs_cores[i].shape[0], bond)

    cores = []
    # carry: the new cores before i, conjugated, contracted with the product
    # cores before i
    carry = np.ones((1, 1, 1), dtype=real)
    for i in range(count):
        shape = (carry.shape[0],) + lhs_cores[i].shape[1:3] + (-1,)
        matrix = _carry_times_hadamard(carry, lhs_cores[i], rhs_cores[i])
        if i == count - 1:
            cores.append(matrix.reshape(shape))
            break
        Q, _ = qr(
            matrix @ sketches[i + 1].reshape(matrix.shape[1], -1), mode="economic"
        )
        cores.append(Q.reshape(shape))
        carry = (Q.conj().T @ matrix).reshape(
            Q.shape[1], lhs_cores[i].shape[3], rhs_cores[i].shape[3]
        )
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
            S[:next_bond, None] * V[:next_bond, :],
        )
        cores[i] = U
        next_bond = C.shape[0]
        cores[i + 1] = _multiply_left(C, cores[i + 1])
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
            U[:, :next_bond] * S[:next_bond],
            V[:next_bond, :].reshape((next_bond,) + physical_indexes + (bond,)),
        )
        cores[i] = V
        cores[i - 1] = _multiply_right(cores[i - 1], C)
        bond = next_bond
    return cores
