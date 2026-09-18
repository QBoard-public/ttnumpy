"""Tests for the ALS one-site linear solver."""

import unittest

import numpy as np

from ttnumpy import tensor_train
from ttnumpy.solvers.als import als
from solver_test_utils import qtt_laplacian_1d


class TestAls(unittest.TestCase):
    """Tests for the fixed-rank ALS solver."""

    def test_als_solves_with_sufficient_rank(self):
        """With enough rank in the guess ALS must reach the exact solution."""

        order = 6
        laplacian_tt, rhs_tt, _ = qtt_laplacian_1d(order)
        size = 2**order
        h = 1.0 / (size + 1)
        laplacian = (
            np.diag(-2.0 * np.ones(size))
            + np.diag(np.ones(size - 1), 1)
            + np.diag(np.ones(size - 1), -1)
        ) / h**2
        exact = np.linalg.solve(laplacian, np.ones(size))

        # the solution is a parabola of exact QTT rank 3
        guess = tensor_train.ones([2] * order, ranks=3).orthonormalize(
            canonical="right"
        )
        solution, info = als(
            laplacian_tt,
            rhs_tt,
            x0=guess,
            max_sweeps=10,
            residual_tol=1e-9,
            symmetric=True,
            return_info=True,
        )

        self.assertTrue(info.converged)
        rel_err = np.linalg.norm(
            solution.get_full_tensor().reshape(-1) - exact
        ) / np.linalg.norm(exact)
        self.assertLess(rel_err, 1e-9)

        # ALS is fixed-rank: bonds may shrink but must never grow
        for bond, guess_bond in zip(solution.bonds, guess.bonds):
            self.assertLessEqual(bond, guess_bond)

    def test_als_stagnates_when_rank_starved(self):
        """ALS cannot grow ranks, so a deficient guess must stall.

        This is the weakness AMEn's residual enrichment removes; keep it as
        the reference case for that comparison.
        """

        order = 6
        laplacian_tt, rhs_tt, _ = qtt_laplacian_1d(order)

        guess = tensor_train.ones([2] * order, ranks=1).orthonormalize(
            canonical="right"
        )
        solution, info = als(
            laplacian_tt,
            rhs_tt,
            x0=guess,
            max_sweeps=15,
            residual_tol=1e-9,
            symmetric=True,
            return_info=True,
        )

        self.assertFalse(info.converged)
        self.assertEqual(info.stop_reason, "stagnated")
        # rank never grew, so the residual stays far from the target
        self.assertEqual(max(solution.bonds), 1)
        self.assertGreater(info.residuals[-1], 1e-3)

    def test_als_iterative_matches_direct(self):
        """The matrix-free local solve must agree with the explicit one."""

        order = 6
        laplacian_tt, rhs_tt, _ = qtt_laplacian_1d(order)
        guess = tensor_train.ones([2] * order, ranks=6).orthonormalize(
            canonical="right"
        )

        direct = als(
            laplacian_tt,
            rhs_tt,
            x0=guess.copy(),
            max_sweeps=4,
            residual_tol=1e-9,
            symmetric=True,
            local_solver="direct",
        )
        # a tiny direct_solve_size forces every local system onto MINRES
        iterative = als(
            laplacian_tt,
            rhs_tt,
            x0=guess.copy(),
            max_sweeps=4,
            residual_tol=1e-9,
            symmetric=True,
            local_solver="minres",
            direct_solve_size=4,
        )

        reference = direct.get_full_tensor().reshape(-1)
        rel_diff = np.linalg.norm(
            iterative.get_full_tensor().reshape(-1) - reference
        ) / np.linalg.norm(reference)
        self.assertLess(rel_diff, 1e-8)

    def test_als_keeps_train_consistent(self):
        """Core shapes, bonds and the orthogonality centre must stay in sync."""

        order = 5
        laplacian_tt, rhs_tt, _ = qtt_laplacian_1d(order)
        guess = tensor_train.ones([2] * order, ranks=4).orthonormalize(
            canonical="right"
        )

        solution = als(laplacian_tt, rhs_tt, x0=guess, max_sweeps=3, symmetric=True)

        self.assertEqual(solution.orth_center, 0)
        self.assertEqual(solution.max_rank, max(solution.bonds))
        self.assertEqual(solution.bonds[0], 1)
        self.assertEqual(solution.bonds[-1], 1)
        for k, core in enumerate(solution.cores):
            self.assertEqual(core.shape[0], solution.bonds[k])
            self.assertEqual(core.shape[3], solution.bonds[k + 1])
