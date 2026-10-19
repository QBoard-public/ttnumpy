# Copyright (c) 2026 QBoard LLC. Licensed under the AGPLv3. See LICENSE.
# SPDX-License-Identifier: AGPL-3.0-or-later

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

    # order 8 bounds the exact (error=0) TT decompositions: the random operator
    # below has no TT structure and reaches bond 2**(ORDER/2)
    ORDER = 8

    def _solution_error(self, solution, reference):
        """Relative error of the solved vector against a dense reference."""
        vector = solution.get_full_tensor().reshape(-1)
        return np.linalg.norm(reference - vector) / np.linalg.norm(reference)

    def _residual(self, operator_tt, solution, rhs_tt):
        """Relative residual, computed in TT arithmetic rather than borrowed."""
        rhs_vector = rhs_tt.get_full_tensor().reshape(-1)
        residual = (
            (operator_tt.dot(solution, truncate=True) - rhs_tt)
            .get_full_tensor()
            .reshape(-1)
        )
        return np.linalg.norm(residual) / np.linalg.norm(rhs_vector)

    def _toeplitz(self):
        """Symmetric Toeplitz A[i, j] = 1 + |i - j|, condition number ~4.6e4."""
        size = 2**self.ORDER
        matrix = sp.linalg.toeplitz(np.arange(1, size + 1), np.arange(1, size + 1))
        train = tensor_train.tt_svd(
            matrix.reshape([2] * 2 * self.ORDER),
            physical_indexes=[(2, 2)] * self.ORDER,
            ttm=True,
            error=0,
        )
        return matrix, train

    def test_mals_solves_a_toeplitz_system_from_either_guess(self):
        """Solve a Toeplitz system from different initial guesses."""
        toeplitz_mat, toeplitz_tt = self._toeplitz()
        rhs_tt = tensor_train.ones(toeplitz_tt.row_modes)
        reference = np.linalg.solve(toeplitz_mat, np.ones(toeplitz_mat.shape[0]))

        guesses = {
            "ones": tensor_train.ones(toeplitz_tt.row_modes, ranks=5).orthonormalize(
                canonical="right"
            ),
            "random": tensor_train.rand(toeplitz_tt.row_modes, ranks=5, seed=153),
        }
        for name, guess in guesses.items():
            with self.subTest(guess=name):
                solution = mals(
                    toeplitz_tt,
                    rhs_tt,
                    x0=guess,
                    error_threshold=TEST_DELTA,
                    max_rank=10,
                    max_sweeps=1,
                )

                self.assertLess(self._solution_error(solution, reference), TEST_DELTA)
                self.assertLess(
                    self._residual(toeplitz_tt, solution, rhs_tt), TEST_DELTA
                )

    def test_mals_solves_a_nonsymmetric_system_from_the_default_guess(self):
        """A dense random operator, condition number 1.4e4, solved without ``x0``."""
        size = 2**self.ORDER
        operator_mat = np.random.default_rng(31).random((size, size))
        operator_tt = tensor_train.tt_svd(
            operator_mat.reshape([2] * 2 * self.ORDER),
            physical_indexes=[(2, 2)] * self.ORDER,
            ttm=True,
            error=0,
        )

        # rhs is A @ x for a known x, so the exact solution needs no inversion
        reference = np.linalg.solve(operator_mat, np.ones(size))
        rhs_tt = tensor_train.tt_svd(
            (operator_mat @ reference).reshape([2] * self.ORDER), error=0
        )

        solution = mals(operator_tt, rhs_tt, error_threshold=TEST_DELTA, max_sweeps=2)

        self.assertLess(self._solution_error(solution, reference), TEST_DELTA)
        self.assertLess(self._residual(operator_tt, solution, rhs_tt), TEST_DELTA)

    def test_mals_solves_a_diagonal_system(self):
        """A well-conditioned operator where truncation is the limit."""
        size = 2**self.ORDER
        diagonal_mat = np.diag(np.arange(1, size + 1))
        diagonal_tt = tensor_train.tt_svd(
            diagonal_mat.reshape([2] * 2 * self.ORDER),
            physical_indexes=[(2, 2)] * self.ORDER,
            ttm=True,
            error=0,
        )

        rhs_tt = tensor_train.ones(diagonal_tt.row_modes, ranks=10)
        guess = tensor_train.ones(diagonal_tt.row_modes, ranks=10).orthonormalize(
            canonical="right"
        )

        solution = mals(
            diagonal_tt,
            rhs_tt,
            x0=guess,
            # the decimation produces the residual asserted below; at
            # TEST_DELTA the run lands at 9.7e-07, above that limit
            error_threshold=TEST_DELTA / 10,
            max_sweeps=1,
        )

        self.assertLess(self._residual(diagonal_tt, solution, rhs_tt), TEST_DELTA)

    def test_mals_on_a_system_past_double_precision(self):
        """Hilbert 256: condition number 6.4e20, numerical rank 20.

        The operator is singular to working precision, so the solution vector
        is not determined by the data. The assertions cover what holds
        regardless: the residual is small and the solution is not accurate.
        """
        size = 2**self.ORDER
        hilbert_mat = sp.linalg.hilbert(size)
        # folded to modes of 4 so the exact TTM has half as many cores
        hilbert_tt = tensor_train.tt_svd(
            hilbert_mat.reshape([4] * 2 * (self.ORDER // 2)),
            physical_indexes=[(4, 4)] * (self.ORDER // 2),
            ttm=True,
            error=0,
        )

        rhs_tt = tensor_train.ones(hilbert_tt.row_modes, ranks=10)
        guess = tensor_train.ones(hilbert_tt.row_modes, ranks=10).orthonormalize(
            canonical="right"
        )

        solution = mals(
            hilbert_tt, rhs_tt, x0=guess, error_threshold=TEST_DELTA, max_sweeps=2
        )
        vector = solution.get_full_tensor().reshape(-1)

        # a near-singular local solve can return non-finite entries; it does not
        self.assertTrue(np.all(np.isfinite(vector)))

        self.assertEqual(solution.bonds[0], 1)
        self.assertEqual(solution.bonds[-1], 1)
        for index, core in enumerate(solution.cores):
            with self.subTest(core=index):
                self.assertEqual(core.shape[0], solution.bonds[index])
                self.assertEqual(core.shape[3], solution.bonds[index + 1])

        self.assertLess(self._residual(hilbert_tt, solution, rhs_tt), TEST_DELTA * 100)

        # a small residual at this conditioning does not imply an accurate solution
        reference = np.linalg.solve(hilbert_mat, rhs_tt.get_full_tensor().reshape(-1))
        self.assertGreater(self._solution_error(solution, reference), 0.5)

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
        initial_tt = tensor_train.ones(toeplitz_tt.row_modes, ranks=5).orthonormalize(
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
