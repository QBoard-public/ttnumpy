"""Tests for the Poisson benchmark problem.

Covers the QTT Laplacian, the folding that implements ``mode_size``, and the
assembly of the operator/right-hand side pair the benchmark backends share.
"""

import numpy as np
import pytest

from benchmarks.poisson_problem import (
    build_poisson_problem,
    create_laplace_2d,
    fold_cores,
    folding_group,
    solve_poisson_tt,
)


def _solution_vector(solution) -> np.ndarray:
    return np.asarray(solution.get_full_tensor()).reshape(-1)


def test_folding_group_validates_mode_size():
    assert folding_group(2, 4) == 1
    assert folding_group(4, 4) == 2
    assert folding_group(16, 4) == 4
    with pytest.raises(ValueError, match="power of two"):
        folding_group(3, 4)
    with pytest.raises(ValueError, match="power of two"):
        folding_group(1, 4)
    # log2(8) = 3 does not divide 4 cores per axis, so a fused core would mix axes
    with pytest.raises(ValueError, match="must divide"):
        folding_group(8, 4)


def test_fold_cores_rejects_bad_group():
    train = create_laplace_2d(N=2, h=0.1)
    with pytest.raises(ValueError, match="positive"):
        fold_cores(train, 0)
    with pytest.raises(ValueError, match="does not divide"):
        fold_cores(train, 3)


@pytest.mark.parametrize("mode_size", [2, 4, 16])
def test_laplacian_folding_is_exact(mode_size: int):
    """Folding changes the tensorization, never the operator it represents."""
    N = 4
    K = 2**N
    h = 1.0 / (K + 1)
    reference = create_laplace_2d(N=N, h=h).get_full_tensor().reshape(K * K, K * K)

    folded = create_laplace_2d(N=N, h=h, mode_size=mode_size)

    assert folded.n_cores == 2 * N // (mode_size.bit_length() - 1)
    assert folded.row_modes == [mode_size] * folded.n_cores
    np.testing.assert_allclose(
        np.asarray(folded.get_full_tensor()).reshape(K * K, K * K),
        reference,
        atol=1e-12,
    )


@pytest.mark.parametrize("mode_size", [4, 16])
def test_problem_folding_keeps_operator_and_rhs(mode_size: int):
    """The tensorization is free to change; the discrete problem is not."""
    N = 4
    qtt = build_poisson_problem(N=N, tt_format=True, mode_size=2)
    folded = build_poisson_problem(N=N, tt_format=True, mode_size=mode_size)

    assert folded.mode_size == mode_size
    assert folded.rhs_qtt.row_modes == [mode_size] * folded.rhs_qtt.n_cores
    # operator and right-hand side must fold to the same number of cores, or
    # the solvers reject the triple as mismatched
    assert folded.laplacian_tt.n_cores == folded.rhs_qtt.n_cores
    np.testing.assert_array_equal(folded.rhs, qtt.rhs)
    np.testing.assert_allclose(
        _solution_vector(folded.rhs_qtt),
        _solution_vector(qtt.rhs_qtt),
        atol=1e-14,
    )


def test_solve_rejects_problem_without_tt_operator():
    """The benchmark backends share one problem, only some of them TT."""
    problem = build_poisson_problem(N=3, compute_reference=False)

    with pytest.raises(TypeError, match="tt_format=True"):
        solve_poisson_tt(problem)


@pytest.mark.parametrize("solver", ["mals", "amen"])
@pytest.mark.parametrize("mode_size", [2, 4, 16])
def test_solvers_agree_across_mode_sizes(solver: str, mode_size: int):
    """Every solver and tensorization must solve the same discrete system."""
    N = 4
    problem = build_poisson_problem(N=N, tt_format=True, mode_size=mode_size)

    solution, info = solve_poisson_tt(
        problem,
        solver=solver,
        residual_tol=1e-8,
        max_rank=None,
        max_sweeps=20,
        seed=0,
        return_info=True,
    )

    assert info.converged
    vector = _solution_vector(solution)
    assert vector.shape == (4**N,)
    relative = np.linalg.norm(vector - problem.exact) / np.linalg.norm(problem.exact)
    assert relative < 1e-8


def test_constant_rhs_fills_the_whole_grid():
    """The default right-hand side is a constant over the interior nodes."""
    problem = build_poisson_problem(N=3, rhs=2.5, compute_reference=False)

    assert problem.rhs.shape == (problem.K**2,)
    np.testing.assert_allclose(problem.rhs, 2.5)


def test_callable_rhs_is_sampled_on_the_interior_nodes():
    """f(x, y) must be evaluated at (i+1)h, with y slow and x fast.

    The index order matters: it has to match the core ordering of
    create_laplace_2d, or the operator and the right-hand side describe
    different problems.
    """
    N = 3
    K = 2**N
    h = 1.0 / (K + 1)

    problem = build_poisson_problem(
        N=N, rhs=lambda x, y: x + 10.0 * y, compute_reference=False
    )

    nodes = (np.arange(K) + 1) * h
    expected = np.array([[x + 10.0 * y for x in nodes] for y in nodes]).reshape(-1)
    np.testing.assert_allclose(problem.rhs, expected, rtol=1e-12)


def test_callable_rhs_must_return_the_grid_shape():
    with pytest.raises(ValueError, match="shape"):
        build_poisson_problem(N=3, rhs=lambda x, y: np.ones(3), compute_reference=False)


@pytest.mark.parametrize("flat", [False, True], ids=["grid", "flat"])
def test_array_rhs_is_accepted_in_either_layout(flat: bool):
    """Samples may arrive as (K, K) indexed [y, x] or already flattened."""
    N = 3
    K = 2**N
    grid = np.arange(float(K * K)).reshape(K, K)

    problem = build_poisson_problem(
        N=N, rhs=grid.reshape(-1) if flat else grid, compute_reference=False
    )

    np.testing.assert_array_equal(problem.rhs, grid.reshape(-1))


def test_array_rhs_of_the_wrong_shape_is_rejected():
    with pytest.raises(ValueError, match="rhs must be"):
        build_poisson_problem(N=3, rhs=np.ones(7), compute_reference=False)


def test_array_rhs_does_not_alias_the_caller_s_array():
    """A flat array is copied, so mutating it afterwards cannot change the problem."""
    N = 3
    samples = np.ones(4**N)

    problem = build_poisson_problem(N=N, rhs=samples, compute_reference=False)
    samples[0] = 999.0

    assert problem.rhs[0] == 1.0


def test_rhs_reaches_the_solution_through_the_tt_form():
    """A non-constant rhs must survive compression into the TT right-hand side."""
    N = 3
    problem = build_poisson_problem(
        N=N, rhs=lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y), tt_format=True
    )

    solution = solve_poisson_tt(
        problem, solver="mals", residual_tol=1e-10, max_rank=None, max_sweeps=20
    )

    vector = _solution_vector(solution)
    relative = np.linalg.norm(vector - problem.exact) / np.linalg.norm(problem.exact)
    assert relative < 1e-8


def test_als_branch_runs_but_cannot_grow_past_its_guess():
    """``solver="als"`` is a supported choice with a documented ceiling.

    ALS is fixed-rank and the dispatcher hands it a rank-3 guess, so on 2D
    Poisson it drives the residual down by several orders of magnitude and
    then stagnates rather than converging. Pinning that keeps the branch
    exercised and records why the benchmark suite runs MALS and AMEn instead.
    """
    N = 3
    problem = build_poisson_problem(N=N, tt_format=True)

    solution, info = solve_poisson_tt(
        problem, solver="als", residual_tol=1e-8, max_sweeps=20, return_info=True
    )

    assert not info.converged
    assert str(info.stop_reason) == "stagnated"
    assert max(solution.bonds) <= 3

    vector = _solution_vector(solution)
    assert vector.shape == (4**N,)
    relative = np.linalg.norm(vector - problem.exact) / np.linalg.norm(problem.exact)
    assert relative < 1e-2


def test_solve_rejects_an_unknown_solver_name():
    problem = build_poisson_problem(N=3, tt_format=True, compute_reference=False)

    with pytest.raises(ValueError, match="must be one of"):
        solve_poisson_tt(problem, solver="gmres")
