# Copyright (c) 2026 QBoard LLC. Licensed under the AGPLv3. See LICENSE.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the AMEn one-site solver and its residual/enrichment machinery."""

import unittest
from unittest import mock

import numpy as np
import numpy.testing as npt

from ttnumpy import tensor_train
from ttnumpy.solvers import one_site
from ttnumpy.solvers.common import TruncationMode, relative_residual
from ttnumpy.solvers.amen import (
    _enrich_left,
    _enrich_right,
    _truncate_core_left,
    _truncate_core_right,
    amen,
    fit_residual_train,
    init_residual_train,
)
from solver_test_utils import qtt_laplacian_1d


class TestAmen(unittest.TestCase):
    """Tests for the residual train, enrichment and the AMEn solver."""

    def test_residual_train_fits_residual(self):
        """The residual train must converge to b - A x as its rank allows."""

        order = 5
        rng = np.random.default_rng(2)
        operator_tt = tensor_train.tt_svd(
            rng.random((2**order, 2**order)).reshape([2] * 2 * order),
            physical_indexes=[(2, 2)] * order,
            ttm=True,
            error=0,
        )
        rhs_tt = tensor_train.tt_svd(rng.random(2**order).reshape([2] * order), error=0)
        x = tensor_train.rand([2] * order, ranks=3, seed=9).orthonormalize(
            canonical="right"
        )

        # z fits b - A x; note relative_residual builds A x - b, the negative
        target = rhs_tt.add(operator_tt.dot(x).multiply(-1), truncate=False)
        target_full = target.get_full_tensor().reshape(-1)
        target_norm = target.tt_norm()

        def fit_error(kickrank, sweeps):
            z = init_residual_train(rhs_tt, kickrank=kickrank, seed=4)
            fit_residual_train(operator_tt, rhs_tt, x, z, sweeps=sweeps)
            return (
                np.linalg.norm(z.get_full_tensor().reshape(-1) - target_full)
                / target_norm
            )

        # a train of full TT rank represents the residual exactly
        self.assertLess(fit_error(4, 1), 1e-12)

        # a rank-starved fit is inexact but still a sensible approximation
        poor = fit_error(1, 1)
        self.assertLess(poor, 0.5)
        self.assertGreater(poor, 1e-6)

        # more rank cannot hurt
        self.assertLess(fit_error(2, 1), poor + 1e-12)

        # extra sweeps must not make the fit worse
        z = init_residual_train(rhs_tt, kickrank=2, seed=4)
        errors = []
        for _ in range(3):
            fit_residual_train(operator_tt, rhs_tt, x, z, sweeps=1)
            errors.append(
                np.linalg.norm(z.get_full_tensor().reshape(-1) - target_full)
                / target_norm
            )
        for previous, current in zip(errors, errors[1:]):
            self.assertLessEqual(current, previous * (1.0 + 1e-9))

    def test_residual_train_state_after_fit(self):
        """A fitted residual train must be right-canonical and consistent.

        AMEn's forward sweep starts from exactly this state.
        """

        order = 5
        laplacian_tt, rhs_tt, _ = qtt_laplacian_1d(order)
        x = tensor_train.rand([2] * order, ranks=2, seed=3).orthonormalize(
            canonical="right"
        )

        z = init_residual_train(rhs_tt, kickrank=3, seed=1)
        fit_residual_train(laplacian_tt, rhs_tt, x, z, sweeps=2)

        self.assertEqual(z.orth_center, 0)
        self.assertEqual(z.max_rank, max(z.bonds))
        for k, core in enumerate(z.cores):
            self.assertEqual(core.shape[0], z.bonds[k])
            self.assertEqual(core.shape[3], z.bonds[k + 1])

        # every core past the centre must be right-orthonormal
        for k in range(1, len(z)):
            core = z.cores[k][:, :, 0, :].reshape(z.bonds[k], -1)
            np.testing.assert_allclose(
                core @ core.conj().T, np.eye(z.bonds[k]), atol=1e-10
            )

    def test_init_residual_train(self):
        """Residual train initialization must validate and orthonormalize."""

        rhs_tt = tensor_train.ones([2] * 4)

        z = init_residual_train(rhs_tt, kickrank=3, seed=7)
        self.assertEqual(z.physical_dims(), rhs_tt.physical_dims())
        self.assertLessEqual(max(z.bonds), 3)
        for k in range(1, len(z)):
            core = z.cores[k][:, :, 0, :].reshape(z.bonds[k], -1)
            np.testing.assert_allclose(
                core @ core.conj().T, np.eye(z.bonds[k]), atol=1e-10
            )

        # the same seed must reproduce the same train
        other = init_residual_train(rhs_tt, kickrank=3, seed=7)
        for core, other_core in zip(z.cores, other.cores):
            np.testing.assert_array_equal(core, other_core)

        with self.assertRaises(ValueError):
            init_residual_train(rhs_tt, kickrank=0)
        with self.assertRaises(ValueError):
            init_residual_train(rhs_tt, kickrank=-2)

    def test_truncate_core_helpers(self):
        """Truncation factors must reconstruct the core and stay orthogonal."""

        rng = np.random.default_rng(0)
        core = rng.random((3, 2, 4))
        threshold = TruncationMode.THRESHOLD

        left, carry = _truncate_core_right(core.copy(), threshold, 0.0, None)
        np.testing.assert_allclose(
            np.tensordot(left, carry, axes=(2, 0)), core, atol=1e-12
        )
        # left factor has orthonormal columns
        mat = left.reshape(-1, left.shape[2])
        np.testing.assert_allclose(mat.conj().T @ mat, np.eye(mat.shape[1]), atol=1e-12)

        right, carry_left = _truncate_core_left(core.copy(), threshold, 0.0, None)
        np.testing.assert_allclose(
            np.tensordot(carry_left, right, axes=(1, 0)), core, atol=1e-12
        )
        mat = right.reshape(right.shape[0], -1)
        np.testing.assert_allclose(mat @ mat.conj().T, np.eye(mat.shape[0]), atol=1e-12)

        # max_rank caps the split
        left, carry = _truncate_core_right(core.copy(), threshold, 0.0, 2)
        self.assertEqual(left.shape[2], 2)
        self.assertEqual(carry.shape, (2, 4))

    def test_enrich_helpers_preserve_value(self):
        """Enrichment must widen a bond without changing the tensor.

        This is the zero-pad invariant: the appended directions multiply
        zero rows of the carried factor, so the represented tensor is
        untouched. Getting it wrong silently corrupts the solution.
        """

        order = 5
        rng = np.random.default_rng(8)

        # forward: identity carry makes core . carry the current core k
        x = tensor_train.rand([2] * order, ranks=3, seed=4).orthonormalize(
            canonical="right"
        )
        full_before = x.get_full_tensor()
        k = 1
        core = x.cores[k][:, :, 0, :].copy()
        carry = np.eye(x.bonds[k + 1])
        block = rng.random((x.bonds[k], 2, 3))
        _enrich_right(x, k, core, carry, block)
        np.testing.assert_allclose(x.get_full_tensor(), full_before, atol=1e-12)
        self.assertEqual(x.orth_center, k + 1)
        for site, site_core in enumerate(x.cores):
            self.assertEqual(site_core.shape[0], x.bonds[site])
            self.assertEqual(site_core.shape[3], x.bonds[site + 1])

        # backward mirror
        x = tensor_train.rand([2] * order, ranks=3, seed=4).orthonormalize(
            canonical="right"
        )
        full_before = x.get_full_tensor()
        k = 3
        core = x.cores[k][:, :, 0, :].copy()
        carry = np.eye(x.bonds[k])
        block = rng.random((3, 2, x.bonds[k + 1]))
        _enrich_left(x, k, core, carry, block)
        np.testing.assert_allclose(x.get_full_tensor(), full_before, atol=1e-12)
        self.assertEqual(x.orth_center, k - 1)
        for site, site_core in enumerate(x.cores):
            self.assertEqual(site_core.shape[0], x.bonds[site])
            self.assertEqual(site_core.shape[3], x.bonds[site + 1])

    def test_enrich_hard_cap_is_value_exact(self):
        """max_rank drops only surplus enrichment directions (H3): the bond is
        capped and the represented tensor is unchanged."""
        order = 5

        # forward
        rng = np.random.default_rng(11)
        base = tensor_train.rand([2] * order, ranks=3, seed=4).orthonormalize(
            canonical="right"
        )
        full_before = base.get_full_tensor()
        k = 2
        core = base.cores[k][:, :, 0, :].copy()
        carry = np.eye(base.bonds[k + 1])
        block = rng.random((base.bonds[k], 2, 5))

        uncapped = base.copy()
        _enrich_right(uncapped, k, core, carry, block)
        natural = uncapped.bonds[k + 1]
        self.assertGreater(natural, base.bonds[k + 1])  # enrichment actually widened

        capped = base.copy()
        _enrich_right(capped, k, core, carry, block, max_rank=natural - 1)
        self.assertEqual(capped.bonds[k + 1], natural - 1)
        np.testing.assert_allclose(capped.get_full_tensor(), full_before, atol=1e-12)
        for site, site_core in enumerate(capped.cores):
            self.assertEqual(site_core.shape[0], capped.bonds[site])
            self.assertEqual(site_core.shape[3], capped.bonds[site + 1])

        # backward mirror
        base = tensor_train.rand([2] * order, ranks=3, seed=4).orthonormalize(
            canonical="right"
        )
        full_before = base.get_full_tensor()
        k = 2
        core = base.cores[k][:, :, 0, :].copy()
        carry = np.eye(base.bonds[k])
        block = rng.random((5, 2, base.bonds[k + 1]))

        uncapped = base.copy()
        _enrich_left(uncapped, k, core, carry, block)
        natural = uncapped.bonds[k]
        self.assertGreater(natural, base.bonds[k])

        capped = base.copy()
        _enrich_left(capped, k, core, carry, block, max_rank=natural - 1)
        self.assertEqual(capped.bonds[k], natural - 1)
        np.testing.assert_allclose(capped.get_full_tensor(), full_before, atol=1e-12)
        for site, site_core in enumerate(capped.cores):
            self.assertEqual(site_core.shape[0], capped.bonds[site])
            self.assertEqual(site_core.shape[3], capped.bonds[site + 1])

    def test_amen_converges_where_als_stagnates(self):
        """The headline property: enrichment removes ALS's rank stagnation.

        Same operator, same rank-1 initial guess as
        test_als_stagnates_when_rank_starved -- ALS stalls at a residual
        above 1e-3 there, AMEn must reach 1e-8.
        """

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

        guess = tensor_train.ones([2] * order, ranks=1).orthonormalize(
            canonical="right"
        )
        solution, info = amen(
            laplacian_tt,
            rhs_tt,
            x0=guess,
            kickrank=3,
            error_threshold=1e-12,
            residual_tol=1e-8,
            max_sweeps=20,
            symmetric=True,
            return_info=True,
            seed=0,
        )

        self.assertTrue(info.converged)
        rel_err = np.linalg.norm(
            solution.get_full_tensor().reshape(-1) - exact
        ) / np.linalg.norm(exact)
        self.assertLess(rel_err, 1e-8)
        # the rank trace is populated, one entry per sweep
        self.assertEqual(len(info.max_bonds), info.num_sweeps_run)
        self.assertGreater(max(solution.bonds), 1)

    def test_amen_default_guess_and_nonsymmetric(self):
        """AMEn must work from the trivial default guess, GMRES path included."""

        order = 5
        rng = np.random.default_rng(5)
        operator_mat = np.eye(2**order) + 0.3 * rng.random((2**order, 2**order))
        operator_tt = tensor_train.tt_svd(
            operator_mat.reshape([2] * 2 * order),
            physical_indexes=[(2, 2)] * order,
            ttm=True,
            error=0,
        )
        rhs_vec = rng.random(2**order)
        rhs_tt = tensor_train.tt_svd(rhs_vec.reshape([2] * order), error=0)
        exact = np.linalg.solve(operator_mat, rhs_vec)

        solution, info = amen(
            operator_tt,
            rhs_tt,
            kickrank=4,
            error_threshold=1e-12,
            residual_tol=1e-9,
            max_sweeps=25,
            return_info=True,
            seed=3,
        )

        self.assertTrue(info.converged)
        rel_err = np.linalg.norm(
            solution.get_full_tensor().reshape(-1) - exact
        ) / np.linalg.norm(exact)
        self.assertLess(rel_err, 1e-9)

    def test_threshold_mode_solves_matrix_free(self):
        """error_threshold runs must reach the matrix-free path too.

        The iterative path used to be gated on residual_tol, so a threshold
        run assembled the explicit local matrix however large it was.
        """
        order = 6
        laplacian_tt, rhs_tt, _ = qtt_laplacian_1d(order)

        with mock.patch.object(
            one_site,
            "local_explicit_matrix_one_site",
            side_effect=AssertionError("explicit matrix assembled"),
        ) as explicit:
            solution = amen(
                laplacian_tt,
                rhs_tt,
                error_threshold=1e-10,
                # no residual target: this is the threshold truncation mode
                residual_tol=None,
                max_sweeps=4,
                # every local system here is above the direct-solve cutoff
                direct_solve_size=1,
                symmetric=True,
                seed=0,
            )

        explicit.assert_not_called()
        self.assertLess(relative_residual(laplacian_tt, rhs_tt, solution), 1e-4)

    def test_amen_max_rank_bound(self):
        """max_rank is a hard cap on every bond, enrichment included."""

        order = 6
        laplacian_tt, rhs_tt, _ = qtt_laplacian_1d(order)

        solution, info = amen(
            laplacian_tt,
            rhs_tt,
            kickrank=2,
            max_rank=2,
            error_threshold=1e-12,
            residual_tol=1e-8,
            max_sweeps=6,
            symmetric=True,
            return_info=True,
            seed=2,
        )

        # hard cap: bonds never exceed max_rank, not even by the enrichment
        self.assertLessEqual(max(solution.bonds), 2)
        for max_bond in info.max_bonds:
            self.assertLessEqual(max_bond, 2)

    def test_amen_hard_cap_allows_convergence(self):
        """A hard cap at the solution's rank still converges: the enrichment is
        squeezed to fit, the solution is not. Convergence also proves the
        capped enrichment stayed value-exact."""

        order = 6
        laplacian_tt, rhs_tt, _ = qtt_laplacian_1d(order)
        guess = tensor_train.ones(laplacian_tt.row_modes, ranks=1)

        solution, info = amen(
            laplacian_tt,
            rhs_tt,
            x0=guess,
            kickrank=4,
            max_rank=3,
            error_threshold=1e-12,
            residual_tol=1e-8,
            max_sweeps=20,
            symmetric=True,
            return_info=True,
            seed=1,
        )

        self.assertTrue(info.converged)
        self.assertLessEqual(max(solution.bonds), 3)

    def test_amen_residual_based_truncation(self):
        """With residual_tol set, core truncation follows the residual budget
        and error_threshold is ignored -- the same two-mode format as MALS
        (C1)."""

        order = 6
        laplacian_tt, rhs_tt, _ = qtt_laplacian_1d(order)
        guess = tensor_train.ones(laplacian_tt.row_modes, ranks=1)
        shared = dict(
            kickrank=4,
            residual_tol=1e-8,
            max_sweeps=20,
            symmetric=True,
            return_info=True,
            seed=1,
        )

        # an absurdly loose error_threshold would collapse sigma-relative
        # truncation to rank 1 and stall; residual-based truncation ignores it
        loose, loose_info = amen(
            laplacian_tt, rhs_tt, x0=guess, error_threshold=0.5, **shared
        )
        tight, _ = amen(laplacian_tt, rhs_tt, x0=guess, error_threshold=1e-12, **shared)

        self.assertTrue(loose_info.converged)
        self.assertLessEqual(loose_info.residuals[-1], 1e-8)
        self.assertGreater(max(loose.bonds), 1)
        # error_threshold has no effect once residual_tol drives the rank
        self.assertEqual(loose.bonds, tight.bonds)
        np.testing.assert_allclose(
            loose.get_full_tensor(), tight.get_full_tensor(), atol=1e-12
        )

        # residual_damp below 1 is rejected, as in mals
        with self.assertRaises(ValueError):
            amen(laplacian_tt, rhs_tt, residual_damp=0.5)

    def test_amen_seed_reproducible(self):
        """The same seed must give bit-identical runs."""

        laplacian_tt, rhs_tt, _ = qtt_laplacian_1d(5)

        first = amen(
            laplacian_tt, rhs_tt, kickrank=2, max_sweeps=2, symmetric=True, seed=11
        )
        second = amen(
            laplacian_tt, rhs_tt, kickrank=2, max_sweeps=2, symmetric=True, seed=11
        )
        for core, other_core in zip(first.cores, second.cores):
            np.testing.assert_array_equal(core, other_core)

    def test_amen_input_validation(self):
        """AMEn-specific arguments must be validated fast."""
        laplacian_tt, rhs_tt, _ = qtt_laplacian_1d(3)

        with self.assertRaises(ValueError):
            amen(laplacian_tt, rhs_tt, kickrank=0)
        with self.assertRaises(ValueError):
            amen(laplacian_tt, rhs_tt, kickrank=-1)
        with self.assertRaises(ValueError):
            amen(laplacian_tt, rhs_tt, max_rank=0)
        # shared structural validation still applies
        with self.assertRaises(tensor_train.DimensionMismatch):
            amen(laplacian_tt, tensor_train.ones([2] * 4))

    def test_amen_leaves_the_global_numpy_rng_alone(self):
        """Solving must not cost the caller their own reproducibility.

        AMEn is the one solver that draws random numbers, for its residual
        enrichment train. It reaches them through ``tensor_train.rand``, so a
        generator scoped to the train rather than to the process is what keeps
        a solve from resetting the caller's global stream.
        """

        laplacian_tt, rhs_tt, initial_tt = qtt_laplacian_1d(4)

        def draw_after(**kwargs) -> np.ndarray:
            np.random.seed(20240617)
            amen(
                laplacian_tt,
                rhs_tt,
                x0=initial_tt.copy(),
                max_sweeps=1,
                symmetric=True,
                **kwargs,
            )
            return np.random.rand(4)

        # neither the unseeded nor the seeded path may disturb the caller
        npt.assert_array_equal(draw_after(), draw_after())
        npt.assert_array_equal(draw_after(seed=0), draw_after(seed=17))
