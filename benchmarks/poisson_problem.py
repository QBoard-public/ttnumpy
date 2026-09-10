"""The Poisson benchmark problem: QTT Laplacian, assembly and TT solve.

Builds -Delta u = rhs on a square with homogeneous Dirichlet boundaries in
both the forms the benchmarks need: a sparse matrix with a dense right-hand
side for the classical backends, and a TT operator with a TT right-hand side
for MALS and AMEn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Union

import numpy as np
from scipy import sparse

import ttnumpy as tt
from ttnumpy import TensorTrain

ZERO = np.zeros((2, 2))
I = np.eye(2)
SIGMA_01 = np.eye(2, k=1)
SIGMA_10 = np.eye(2, k=-1)
SIGMA_00 = np.array([[1, 0], [0, 0]])
SIGMA_11 = np.array([[0, 0], [0, 1]])


def folding_group(mode_size: int, cores_per_axis: int) -> int:
    """Number of QTT cores fused into one core of ``mode_size``.

    Args:
        mode_size: Requested mode size; must be a power of two.
        cores_per_axis: QTT cores addressing a single grid axis.

    Returns:
        ``log2(mode_size)``.

    Raises:
        ValueError: If ``mode_size`` is not a power of two of at least 2, or if
            it does not divide the cores of one axis, which would fuse cores of
            two different axes into a single mode.
    """
    if mode_size < 2 or mode_size & (mode_size - 1):
        raise ValueError(f"mode_size must be a power of two >= 2; got {mode_size}")
    group = mode_size.bit_length() - 1
    if cores_per_axis % group:
        raise ValueError(
            f"log2(mode_size) = {group} must divide the {cores_per_axis} cores of "
            f"one axis, otherwise a fused core would mix two grid axes"
        )
    return group


def fold_cores(train: TensorTrain, group: int) -> TensorTrain:
    """Fuse every ``group`` adjacent cores into one core.

    Only the tensorization changes; the represented tensor does not. Mode
    sizes multiply, so ``group`` cores of mode size 2 become one core of mode
    size ``2**group``. Accepts both TT and TTM trains.

    Args:
        train: Train to fold.
        group: Cores per fused core; 1 returns a copy.

    Returns:
        The same tensor with ``n_cores // group`` cores.

    Raises:
        ValueError: If ``group`` is not positive or does not divide the core count.
    """
    if group < 1:
        raise ValueError(f"group must be positive; got {group}")
    if train.n_cores % group:
        raise ValueError(
            f"group={group} does not divide the {train.n_cores} cores of the train"
        )
    if group == 1:
        return train.copy()

    cores = []
    for start in range(0, train.n_cores, group):
        core = train.cores[start]
        for offset in range(1, group):
            core = np.tensordot(core, train.cores[start + offset], axes=([3], [0]))
            left, rows, cols, next_rows, next_cols, right = core.shape
            # gather the row modes before the column modes, then fuse each pair
            core = core.transpose((0, 1, 3, 2, 4, 5)).reshape(
                left, rows * next_rows, cols * next_cols, right
            )
        cores.append(core)
    return TensorTrain(cores)


def create_laplace_2d(N: int, h: float, mode_size: int = 2, dtype=np.float64):
    """Build the 2D finite-difference Laplacian as a QTT matrix.

    Args:
        N: QTT depth per axis; the grid has ``2**N`` points per axis.
        h: Grid spacing.
        mode_size: Mode size of the returned cores; a power of two whose
            base-2 logarithm divides ``N``.
        dtype: dtype of the returned cores.

    Returns:
        The Laplacian operator in TTM format.

    Raises:
        ValueError: If ``mode_size`` is not an admissible power of two.
    """
    B_0 = (
        np.block([I, SIGMA_01, SIGMA_10, I, ZERO, ZERO])
        .reshape(1, 2, 6, 2)
        .swapaxes(2, 3)
        / h**2
    )
    B_y = (
        np.block(
            [
                [I, SIGMA_01, SIGMA_10, ZERO, ZERO, ZERO],
                [ZERO, SIGMA_10, ZERO, ZERO, ZERO, ZERO],
                [ZERO, ZERO, SIGMA_01, ZERO, ZERO, ZERO],
                [ZERO, ZERO, ZERO, I, ZERO, ZERO],
                [ZERO, ZERO, ZERO, ZERO, I, ZERO],
                [ZERO, ZERO, ZERO, ZERO, ZERO, I],
            ]
        )
        .reshape(6, 2, 6, 2)
        .swapaxes(2, 3)
    )

    B_x = (
        np.block(
            [
                [I, ZERO, ZERO, ZERO, ZERO, ZERO],
                [ZERO, I, ZERO, ZERO, ZERO, ZERO],
                [ZERO, ZERO, I, ZERO, ZERO, ZERO],
                [ZERO, ZERO, ZERO, I, SIGMA_01, SIGMA_10],
                [ZERO, ZERO, ZERO, ZERO, SIGMA_10, ZERO],
                [ZERO, ZERO, ZERO, ZERO, ZERO, SIGMA_01],
            ]
        )
        .reshape(6, 2, 6, 2)
        .swapaxes(2, 3)
    )

    B_L = np.block(
        [[-2 * I], [I], [I], [-2 * I + SIGMA_01 + SIGMA_10], [SIGMA_10], [SIGMA_01]]
    ).reshape(6, 2, 2, 1)
    laplacian = TensorTrain([B_0] + [B_y] * (N - 1) + [B_x] * (N - 1) + [B_L])
    return fold_cores(laplacian, folding_group(mode_size, N))


# A right-hand side: a constant, samples on the interior grid (shape
# (K, K) indexed [y, x] or flat (K**2,)), or a vectorized callable f(x, y).
RHSLike = Union[float, np.ndarray, Callable[[np.ndarray, np.ndarray], np.ndarray]]


@dataclass(frozen=True)
class PoissonProblem:
    """Discrete Poisson problem on the unit square.

    ``K = 2**N`` is the number of interior points per axis and ``h`` the grid
    spacing. ``rhs`` holds the right-hand side sampled on the interior grid and
    flattened in C order (``[y, x]``, y slow); ``rhs_qtt`` is its TT form.
    ``laplacian`` is always the sparse matrix, while ``laplacian_tt`` is the
    TTM only when the problem was built with ``tt_format=True`` and the sparse
    matrix again otherwise. ``exact`` is the sparse direct reference solution,
    or ``None`` under ``compute_reference=False``. ``mode_size`` is the
    tensorization of the TT operator and right-hand side; the grid, the
    operator and the solution do not depend on it.
    """

    N: int
    K: int
    h: float
    laplacian_tt: tt.TensorTrain | sparse.csr_matrix
    rhs_qtt: tt.TensorTrain
    rhs: np.ndarray
    exact: np.ndarray | None
    laplacian: sparse.csr_matrix
    mode_size: int = 2


def _build_rhs_vector(rhs: RHSLike, K: int, h: float) -> np.ndarray:
    """Sample ``rhs`` on the interior grid and flatten it in C order."""
    if callable(rhs):
        coordinates = (np.arange(K) + 1) * h
        # default "xy" indexing: axis 0 is y (slow), axis 1 is x (fast),
        # matching the core ordering of create_laplace_2d
        x_grid, y_grid = np.meshgrid(coordinates, coordinates)
        values = np.asarray(rhs(x_grid, y_grid), dtype=float)
        if values.shape != (K, K):
            raise ValueError(
                f"rhs callable must return an array of shape ({K}, {K}); "
                f"got shape {values.shape}"
            )
        return values.reshape(-1)

    values = np.asarray(rhs, dtype=float)
    if values.ndim == 0:
        return np.full(K**2, float(values))
    if values.shape == (K, K):
        return values.reshape(-1)
    if values.shape == (K**2,):
        return values.copy()
    raise ValueError(
        f"rhs must be a scalar, a callable f(x, y), or an array of shape "
        f"({K}, {K}) or ({K**2},); got shape {values.shape}"
    )


def build_poisson_problem(
    L: float = 1.0,
    N: int = 3,
    rhs: RHSLike = 1.0,
    rhs_tt_error: float = 1e-12,
    tt_format: bool = False,
    compute_reference: bool = True,
    mode_size: int = 2,
) -> PoissonProblem:
    r"""Build :math:`-\Delta u = \mathrm{rhs}` on :math:`(0, L)^2`, Dirichlet.

    The grid has ``K = 2**N`` interior points per axis with spacing
    ``h = L / (K + 1)``, so the interior nodes are ``x_i = y_i = (i + 1) * h``.

    Args:
        L: Side length of the square domain.
        N: QTT depth per axis.
        rhs: A constant, an array of samples on the interior grid (shape
            ``(K, K)`` indexed ``[y, x]`` or flat ``(K**2,)``), or a
            vectorized callable ``f(x, y)`` evaluated on the interior nodes.
        rhs_tt_error: Relative error of the ``tt_svd`` compression of the
            sampled right-hand side.
        tt_format: Also build the Laplacian as a TTM, which
            :func:`solve_poisson_tt` requires.
        compute_reference: Solve the sparse system directly to fill
            ``exact``. Turn it off on grids where ``spsolve`` dominates.
        mode_size: Tensorization of the operator and the right-hand side,
            leaving the grid alone: ``2`` is QTT, ``2**k`` fuses ``k`` QTT
            cores per core (:func:`fold_cores`), and ``2**N`` leaves one core
            per grid axis.

    Returns:
        The assembled problem.

    Raises:
        ValueError: If ``rhs`` has an unusable shape, or if the base-2
            logarithm of ``mode_size`` does not divide ``N``.
    """
    K = 2**N
    h = L / (K + 1)
    rhs_vector = _build_rhs_vector(rhs, K, h)

    one_dim_laplacian = (
        sparse.diags(
            [np.ones(K - 1), -2.0 * np.ones(K), np.ones(K - 1)],
            offsets=[-1, 0, 1],
            shape=(K, K),
            format="csr",
        )
        / h**2
    )
    identity = sparse.eye(K, format="csr")
    laplacian = sparse.kron(identity, one_dim_laplacian, format="csr") + sparse.kron(
        one_dim_laplacian, identity, format="csr"
    )

    if tt_format:
        laplacian_tt = create_laplace_2d(N=N, h=h, mode_size=mode_size)
    else:
        laplacian_tt = laplacian

    if compute_reference:
        exact_vector = sparse.linalg.spsolve(laplacian, -rhs_vector)
    else:
        exact_vector = None

    rhs_qtt = tt.tt_svd(
        rhs_vector.reshape([2] * (2 * N)),
        error=rhs_tt_error,
    )
    if mode_size != 2:
        rhs_qtt = fold_cores(rhs_qtt, folding_group(mode_size, N))

    return PoissonProblem(
        N=N,
        K=K,
        h=h,
        laplacian_tt=laplacian_tt,
        rhs_qtt=rhs_qtt,
        rhs=rhs_vector,
        exact=exact_vector,
        laplacian=laplacian,
        mode_size=mode_size,
    )


def solve_poisson_tt(
    problem: PoissonProblem,
    *,
    solver: str = "mals",
    error_threshold: float = 1e-4,
    max_rank: int | None = 50,
    max_sweeps: int = 1,
    residual_tol: float | None = None,
    kickrank: int = 4,
    seed: int | None = None,
    return_info: bool = False,
) -> tt.TensorTrain | tuple[tt.TensorTrain, tt.SolverInfo]:
    """Solve a prebuilt Poisson problem with one of the TT solvers.

    Args:
        problem: Problem to solve; must carry ``tt_format=True``.
        solver: ``"mals"``, ``"amen"`` or ``"als"``.
        error_threshold: Relative singular-value cutoff for MALS and AMEn.
        max_rank: Bond dimension cap for MALS and AMEn; ``None`` for no cap.
        max_sweeps: Sweep budget.
        residual_tol: Target relative residual; ``None`` selects
            threshold-mode truncation.
        kickrank: Enrichment rank; used by ``"amen"`` only.
        seed: Seed for AMEn's random residual train, for reproducible runs.
        return_info: Return ``(solution, SolverInfo)`` instead of the
            solution alone.

    Returns:
        The solution, or ``(solution, SolverInfo)`` when ``return_info``.

    Raises:
        TypeError: If the problem is not in TT format.
        ValueError: If ``solver`` is not a known solver name.
    """
    if not isinstance(problem.laplacian_tt, tt.TensorTrain):
        raise TypeError(
            "Problem must be created with tt_format=True to solve with TT solvers"
        )
    if solver not in ("mals", "amen", "als"):
        raise ValueError(f"solver must be one of 'mals', 'amen', 'als'; got {solver!r}")

    # the initial guess follows the tensorization of the right-hand side
    initial = tt.ones(problem.rhs_qtt.physical_dims(), ranks=3).compress_svd(
        canonical="right"
    )
    # the discrete Laplacian is symmetric, so "auto" picks MINRES
    common = {
        "x0": initial,
        "max_sweeps": max_sweeps,
        "residual_tol": residual_tol,
        "symmetric": True,
        "return_info": return_info,
    }
    if solver == "als":
        return tt.als(problem.laplacian_tt, -problem.rhs_qtt, **common)
    if solver == "amen":
        return tt.amen(
            problem.laplacian_tt,
            -problem.rhs_qtt,
            error_threshold=error_threshold,
            max_rank=max_rank,
            kickrank=kickrank,
            seed=seed,
            **common,
        )
    return tt.mals(
        problem.laplacian_tt,
        -problem.rhs_qtt,
        error_threshold=error_threshold,
        max_rank=max_rank,
        **common,
    )
