"""Tests for the MALS two-site linear solver."""

import unittest

import numpy as np
import scipy as sp

from ttnumpy import tensor_train
from ttnumpy.solvers.mals import mals
from solver_test_utils import qtt_laplacian_1d

TEST_DELTA = 1e-7
DECIMAL = 10


class TestMals(unittest.TestCase):
    """Tests for the MALS solver and its two-site local algebra."""

    def test_tt_mals(self):
        """Test MALS algorithm for solving linear systems in TT format."""

        # order 8 keeps the exact (error=0) TT decompositions of the explicit
        # test matrices small enough for CI memory limits
        order = 8

        # ------------- TOEPLITZ MATRIX -------------#
        # generate Toeplitz matrix
        toeplitz_mat = sp.linalg.toeplitz(
            np.arange(1, 2**order + 1), np.arange(1, 2**order + 1)
        )
        toeplitz_tt = tensor_train.tt_svd(
            toeplitz_mat.reshape([2] * 2 * order),
            physical_indexes=[(2, 2)] * order,
            ttm=True,
            error=0,
        )

        rhs_mat = np.ones(toeplitz_mat.shape[0])
        rhs_tt = tensor_train.ones(toeplitz_tt.row_modes)

        # not random (all ones) guess for MALS algorithm
        initial_tt = tensor_train.ones(toeplitz_tt.row_modes, ranks=5).compress_svd(
            canonical="right"
        )

        solution_mat = np.linalg.solve(toeplitz_mat, rhs_mat)
        solution_tt_solve = mals(
            toeplitz_tt,
            rhs_tt,
            x0=initial_tt,
            error_threshold=TEST_DELTA,
            max_rank=10,
            max_sweeps=1,
        )

        rel_err_mat_solve = np.linalg.norm(
            solution_mat - solution_tt_solve.get_full_tensor().reshape(-1)
        ) / np.linalg.norm(solution_mat)
        rel_err_tt_solve = np.linalg.norm(
            (toeplitz_tt.dot(solution_tt_solve, truncate=True) - rhs_tt)
            .get_full_tensor()
            .reshape(-1)
        ) / np.linalg.norm(rhs_tt.get_full_tensor().reshape(-1))

        self.assertLess(rel_err_mat_solve, TEST_DELTA)
        self.assertLess(rel_err_tt_solve, TEST_DELTA)

        # random guess for MALS algorithm
        initial_tt_random = tensor_train.rand(toeplitz_tt.row_modes, ranks=5, seed=153)
        solution_tt_solve_random = mals(
            toeplitz_tt,
            rhs_tt,
            x0=initial_tt_random,
            error_threshold=TEST_DELTA,
            max_rank=10,
            max_sweeps=1,
        )

        rel_err_mat_solve_random = np.linalg.norm(
            solution_mat - solution_tt_solve_random.get_full_tensor().reshape(-1)
        ) / np.linalg.norm(solution_mat)
        rel_err_tt_solve_random = np.linalg.norm(
            (toeplitz_tt.dot(solution_tt_solve_random, truncate=True) - rhs_tt)
            .get_full_tensor()
            .reshape(-1)
        ) / np.linalg.norm(rhs_tt.get_full_tensor().reshape(-1))

        self.assertLess(rel_err_mat_solve_random, TEST_DELTA)
        self.assertLess(rel_err_tt_solve_random, TEST_DELTA)

        # ------------- RANDOM MATRIX -------------#
        operator_mat_not_pos = np.random.default_rng(31).random(
            (toeplitz_mat.shape[0], toeplitz_mat.shape[1])
        )
        operator_tt_not_pos = tensor_train.tt_svd(
            operator_mat_not_pos.reshape([2] * 2 * order),
            physical_indexes=[(2, 2)] * order,
            ttm=True,
            error=0,
        )

        solution_mat_not_pos = np.linalg.solve(operator_mat_not_pos, rhs_mat)
        rhs_mat = operator_mat_not_pos @ solution_mat_not_pos
        rhs_tt = tensor_train.tt_svd(rhs_mat.reshape([2] * order), error=0)

        solution_tt_not_pos = mals(
            operator_tt_not_pos, rhs_tt, error_threshold=TEST_DELTA, max_sweeps=2
        )

        rel_err_mat_solve_not_pos = np.linalg.norm(
            solution_mat_not_pos - solution_tt_not_pos.get_full_tensor().reshape(-1)
        ) / np.linalg.norm(solution_mat_not_pos)
        rel_err_tt_solve_not_pos = np.linalg.norm(
            (operator_tt_not_pos.dot(solution_tt_not_pos, truncate=True) - rhs_tt)
            .get_full_tensor()
            .reshape(-1)
        ) / np.linalg.norm(rhs_tt.get_full_tensor().reshape(-1))

        self.assertLess(rel_err_mat_solve_not_pos, TEST_DELTA)
        self.assertLess(rel_err_tt_solve_not_pos, TEST_DELTA)

        # ------------- HILBERT MATRIX -------------#
        hilbert_mat = sp.linalg.hilbert(2**order)
        hilbert_tt = tensor_train.tt_svd(
            hilbert_mat.reshape([4] * 2 * (order // 2)),
            physical_indexes=[(4, 4)] * (order // 2),
            ttm=True,
            error=0,
        )

        rhs_tt = tensor_train.ones(hilbert_tt.row_modes, ranks=10)
        initial_tt = tensor_train.ones(hilbert_tt.row_modes, ranks=10).compress_svd(
            canonical="right"
        )

        solution_tt_hilbert = mals(
            hilbert_tt, rhs_tt, x0=initial_tt, error_threshold=TEST_DELTA, max_sweeps=2
        )

        rel_err_tt_solve_hilbert = np.linalg.norm(
            (hilbert_tt.dot(solution_tt_hilbert, truncate=True) - rhs_tt)
            .get_full_tensor()
            .reshape(-1)
        ) / np.linalg.norm(rhs_tt.get_full_tensor().reshape(-1))

        self.assertLess(rel_err_tt_solve_hilbert, TEST_DELTA * 100)

        # --------------DIAGONAL MATRIX -------------#
        diagonal_mat = np.diag(np.arange(1, 2**order + 1))
        diagonal_tt = tensor_train.tt_svd(
            diagonal_mat.reshape([2] * 2 * order),
            physical_indexes=[(2, 2)] * order,
            ttm=True,
            error=0,
        )

        rhs_tt = tensor_train.ones(diagonal_tt.row_modes, ranks=10)
        initial_tt = tensor_train.ones(diagonal_tt.row_modes, ranks=10).compress_svd(
            canonical="right"
        )

        solution_tt_diagonal = mals(
            diagonal_tt,
            rhs_tt,
            x0=initial_tt,
            error_threshold=TEST_DELTA / 10,
            max_sweeps=1,
        )

        rel_err_tt_solve_diagonal = np.linalg.norm(
            (diagonal_tt.dot(solution_tt_diagonal, truncate=True) - rhs_tt)
            .get_full_tensor()
            .reshape(-1)
        ) / np.linalg.norm(rhs_tt.get_full_tensor().reshape(-1))

        self.assertLess(rel_err_tt_solve_diagonal, TEST_DELTA)

    def test_mals_residual_tracking_and_stopping(self):
        """MALS must track per-sweep residuals and stop on tolerance or stagnation."""

        order = 8
        toeplitz_mat = sp.linalg.toeplitz(
            np.arange(1, 2**order + 1), np.arange(1, 2**order + 1)
        )
        toeplitz_tt = tensor_train.tt_svd(
            toeplitz_mat.reshape([2] * 2 * order),
            physical_indexes=[(2, 2)] * order,
            ttm=True,
            error=0,
        )
        rhs_tt = tensor_train.ones(toeplitz_tt.row_modes)
        initial_tt = tensor_train.ones(toeplitz_tt.row_modes, ranks=5).compress_svd(
            canonical="right"
        )

        # stops as soon as residual_tol is reached
        solution, info = mals(
            toeplitz_tt,
            rhs_tt,
            x0=initial_tt,
            error_threshold=TEST_DELTA,
            max_rank=10,
            max_sweeps=10,
            residual_tol=1e-6,
            return_info=True,
        )
        self.assertEqual(info.stop_reason, "converged")
        self.assertTrue(info.converged)
        self.assertLess(info.num_sweeps_run, 10)
        self.assertEqual(len(info.residuals), info.num_sweeps_run)
        self.assertLessEqual(info.residuals[-1], 1e-6)

        # reported residual must correspond to the returned solution
        rhs_vec = np.ones(2**order)
        explicit_residual = np.linalg.norm(
            toeplitz_mat @ solution.get_full_tensor().reshape(-1) - rhs_vec
        ) / np.linalg.norm(rhs_vec)
        self.assertAlmostEqual(info.residuals[-1], explicit_residual, places=DECIMAL)

        # rank capped at 1: progress stalls and the stagnation check must fire
        _, stalled_info = mals(
            toeplitz_tt,
            rhs_tt,
            x0=initial_tt,
            error_threshold=TEST_DELTA,
            max_rank=1,
            max_sweeps=8,
            residual_tol=1e-12,
            stagnation_factor=0.9,
            return_info=True,
        )
        self.assertEqual(stalled_info.stop_reason, "stagnated")
        self.assertLess(stalled_info.num_sweeps_run, 8)

        # with both checks disabled, all sweeps run and residuals are still tracked
        _, info_max = mals(
            toeplitz_tt,
            rhs_tt,
            x0=initial_tt,
            error_threshold=TEST_DELTA,
            max_rank=1,
            max_sweeps=3,
            residual_tol=None,
            stagnation_factor=None,
            return_info=True,
        )
        self.assertEqual(info_max.stop_reason, "max_sweeps")
        self.assertEqual(info_max.num_sweeps_run, 3)
        self.assertEqual(len(info_max.residuals), 3)

        # invalid stopping parameters are rejected
        with self.assertRaises(ValueError):
            mals(toeplitz_tt, rhs_tt, residual_tol=0.0)
        with self.assertRaises(ValueError):
            mals(toeplitz_tt, rhs_tt, stagnation_factor=1.5)

    def test_mals_syncs_max_rank_and_tracks_max_bonds(self):
        """MALS must sync x.max_rank and record a per-sweep max-bond trace,
        like the one-site solvers (C2)."""
        laplacian_tt, rhs_tt, _ = qtt_laplacian_1d(6)
        guess = tensor_train.ones(laplacian_tt.row_modes, ranks=1)
        self.assertEqual(guess.max_rank, 1)

        solution, info = mals(
            laplacian_tt,
            rhs_tt,
            x0=guess,
            residual_tol=1e-8,
            max_sweeps=5,
            symmetric=True,
            return_info=True,
        )

        # the decimation grew the ranks past the rank-1 guess
        self.assertGreater(max(solution.bonds), 1)
        # the max_rank member reflects the adapted bonds, not the stale guess
        self.assertEqual(solution.max_rank, max(solution.bonds))
        # per-sweep max-bond trace matches the sweeps actually run
        self.assertEqual(len(info.max_bonds), info.num_sweeps_run)
        self.assertEqual(info.max_bonds[-1], max(solution.bonds))

    def test_mals_residual_based_truncation(self):
        """With residual_tol set, decimation keeps the smallest rank whose
        local residual stays within budget (trick 2 of Oseledets & Dolgov)."""

        laplacian_tt, rhs_tt, initial_tt = qtt_laplacian_1d(8)

        solution, info = mals(
            laplacian_tt,
            rhs_tt,
            x0=initial_tt,
            max_rank=30,
            max_sweeps=15,
            residual_tol=1e-6,
            return_info=True,
        )

        self.assertTrue(info.converged)
        self.assertLessEqual(info.residuals[-1], 1e-6)
        # the rank search must not spuriously keep full-rank supercores
        self.assertLessEqual(max(solution.bonds), 5)

        with self.assertRaises(ValueError):
            mals(laplacian_tt, rhs_tt, residual_damp=0.5)

    def test_mals_iterative_local_solver(self):
        """Forced matrix-free Krylov local solves must reach the same residual
        target as the explicit direct path (trick 1 of Oseledets & Dolgov)."""

        laplacian_tt, rhs_tt, initial_tt = qtt_laplacian_1d(8)
        shared = dict(
            x0=initial_tt,
            max_rank=30,
            max_sweeps=15,
            residual_tol=1e-6,
            return_info=True,
        )

        _, direct_info = mals(laplacian_tt, rhs_tt, local_solver="direct", **shared)
        self.assertTrue(direct_info.converged)

        # minres (declared symmetric), every local solve iterative
        _, minres_info = mals(
            laplacian_tt,
            rhs_tt,
            local_solver="minres",
            direct_solve_size=0,
            **shared,
        )
        self.assertTrue(minres_info.converged)
        self.assertLessEqual(minres_info.residuals[-1], 1e-6)

        # gmres (no symmetry assumption), every local solve iterative
        _, gmres_info = mals(
            laplacian_tt,
            rhs_tt,
            local_solver="gmres",
            direct_solve_size=0,
            **shared,
        )
        self.assertTrue(gmres_info.converged)
        self.assertLessEqual(gmres_info.residuals[-1], 1e-6)

        with self.assertRaises(ValueError):
            mals(laplacian_tt, rhs_tt, local_solver="cg")
