# Copyright (c) 2026 QBoard LLC. Licensed under the AGPLv3. See LICENSE.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Partial rank-revealing LU decomposition with complete pivoting.

The decomposition writes a matrix as ``L D U`` truncated at rank ``k``.
Truncated there it equals the cross interpolation
``A(:, J) A(I, J)^-1 A(I, :)`` over the selected rows ``I`` and columns
``J``, without forming or inverting ``A(I, J)``.

Reference:
    Y. Núñez Fernández et al., Learning tensor networks with tensor cross
    interpolation: new algorithms and libraries, SciPost Phys. 18, 104
    (2025), Sec. 3.3.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True, eq=False)
class PrrLuResult:
    """Truncated LU factorization of a matrix, with the pivots it selected.

    ``lower @ diag(diagonal) @ upper`` is the rank-``k`` approximation of the
    factorized matrix and reproduces it exactly on the pivot rows and columns.
    """

    #: Unit-lower-trapezoidal factor of shape (rows, rank). Rows are indexed
    #: as in the factorized matrix, not in pivot order.
    lower: np.ndarray
    #: Pivot values, in the order the elimination selected them.
    diagonal: np.ndarray
    #: Unit-upper-trapezoidal factor of shape (rank, columns). Columns are
    #: indexed as in the factorized matrix, not in pivot order.
    upper: np.ndarray
    #: Indices of the pivot rows, in selection order.
    row_pivots: np.ndarray
    #: Indices of the pivot columns, in selection order.
    col_pivots: np.ndarray
    #: Largest modulus remaining in the Schur complement.
    pivot_error: float

    @property
    def rank(self) -> int:
        """Number of pivots selected."""
        return int(self.diagonal.size)

    def reconstruct(self) -> np.ndarray:
        """Rank-``k`` approximation of the factorized matrix."""
        return (self.lower * self.diagonal) @ self.upper


def _validate(matrix: np.ndarray, tolerance: float, max_rank: Optional[int]) -> None:
    """Check the arguments of :func:`prrlu`.

    Raises:
        ValueError: If matrix is not two-dimensional, tolerance is negative
            or NaN, or max_rank is not a positive integer.
    """
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be two-dimensional, got {matrix.ndim} axes.")
    if tolerance < 0.0 or np.isnan(tolerance):
        raise ValueError(f"tolerance must be non-negative; got {tolerance}.")
    if max_rank is not None and max_rank <= 0:
        raise ValueError(f"max_rank must be a positive integer; got {max_rank}.")


def _eliminate(
    work: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    max_steps: int,
    tolerance: float,
) -> int:
    """Run Gaussian elimination with complete pivoting, in place.

    Each pivot is swapped to the leading position as it is selected.

    Args:
        work: Array to eliminate, modified in place.
        rows: Original index of each row, permuted in place.
        cols: Original index of each column, permuted in place.
        max_steps: Largest number of pivots to take.
        tolerance: Absolute threshold at or below which a pivot is refused.

    Returns:
        int: Number of pivots taken.
    """
    rank = 0
    for step in range(max_steps):
        remaining = np.abs(work[step:, step:])
        row, col = divmod(int(np.argmax(remaining)), remaining.shape[1])
        if remaining[row, col] <= tolerance:
            break
        row += step
        col += step

        work[[step, row], :] = work[[row, step], :]
        rows[[step, row]] = rows[[row, step]]
        work[:, [step, col]] = work[:, [col, step]]
        cols[[step, col]] = cols[[col, step]]

        work[step + 1 :, step] /= work[step, step]
        work[step + 1 :, step + 1 :] -= np.outer(
            work[step + 1 :, step], work[step, step + 1 :]
        )
        rank = step + 1
    return rank


def _assemble_factors(
    work: np.ndarray, rows: np.ndarray, cols: np.ndarray, rank: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split an eliminated array into ``L``, ``D`` and ``U``.

    Args:
        work: Array left by the elimination, in pivot order.
        rows: Original index of each row of ``work``.
        cols: Original index of each column of ``work``.
        rank: Number of pivots taken.

    Returns:
        tuple: The lower, diagonal and upper factors, with rows and columns
        back in the order of the factorized matrix.
    """
    diagonal = np.diag(work)[:rank].copy()
    lower = np.tril(work[:, :rank])
    np.fill_diagonal(lower, 1.0)
    upper = np.triu(work[:rank, :]) / diagonal[:, np.newaxis] if rank else work[:0, :]

    ordered_lower = np.zeros_like(lower)
    ordered_lower[rows] = lower
    ordered_upper = np.zeros_like(upper)
    ordered_upper[:, cols] = upper
    return ordered_lower, diagonal, ordered_upper


def prrlu(
    matrix: np.ndarray,
    tolerance: float = 0.0,
    max_rank: Optional[int] = None,
) -> PrrLuResult:
    """Factorize a matrix by Gaussian elimination with complete pivoting.

    Elimination stops once the largest modulus left in the Schur complement
    falls to ``tolerance``, or once ``max_rank`` pivots have been taken.

    A tolerance below the rounding level of ``matrix``,
    ``min(matrix.shape) * eps * max|matrix|`` with ``eps`` the machine epsilon
    of its dtype, is raised to that level, so ``0.0`` refuses only pivots at
    rounding noise.

    Args:
        matrix: Two-dimensional array to factorize.
        tolerance: Absolute threshold on the largest modulus of the Schur
            complement, not normalized by the entries of ``matrix``.
        max_rank: Upper bound on the number of pivots.

    Raises:
        ValueError: If matrix is not two-dimensional, tolerance is negative
            or NaN, or max_rank is not a positive integer.

    Returns:
        PrrLuResult: The factors, the pivot indices and the pivot error.
    """
    array = np.asarray(matrix)
    _validate(array, tolerance, max_rank)

    dtype = array.dtype if np.issubdtype(array.dtype, np.inexact) else np.float64
    work = array.astype(dtype, copy=True)
    rows = np.arange(work.shape[0])
    cols = np.arange(work.shape[1])

    max_steps = min(work.shape)
    if max_rank is not None:
        max_steps = min(max_steps, max_rank)

    tolerance = max(
        tolerance, min(work.shape) * np.finfo(dtype).eps * np.abs(work).max(initial=0.0)
    )
    rank = _eliminate(work, rows, cols, max_steps, tolerance)

    schur = np.abs(work[rank:, rank:])
    pivot_error = float(schur.max()) if schur.size else 0.0
    lower, diagonal, upper = _assemble_factors(work, rows, cols, rank)

    return PrrLuResult(
        lower=lower,
        diagonal=diagonal,
        upper=upper,
        row_pivots=rows[:rank].copy(),
        col_pivots=cols[:rank].copy(),
        pivot_error=pivot_error,
    )
