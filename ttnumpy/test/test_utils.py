# Copyright (c) 2026 QBoard LLC. Licensed under the AGPLv3. See LICENSE.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the tensor-train utility functions.

Every train here is seeded: these helpers are compared against exact
expectations, so an unlucky unseeded draw would be an unreproducible failure.

The four sweep helpers rebind the elements of the list they are given, which is
how the callers in :mod:`ttnumpy.tensor_train` use them. Tests therefore hand
over ``list(tt.cores)`` and keep ``tt`` itself as the untouched reference.
"""

import unittest

import numpy as np
import numpy.testing as npt

from ttnumpy import tensor_train
from ttnumpy import utils

TEST_DELTA = 1e-10
RTOL = 1e-10
ATOL = 1e-12


def _reference_train(seed: int) -> tensor_train.TensorTrain:
    """A small non-square train, distinct per seed."""
    return tensor_train.rand((2, 3, 4), ranks=5, seed=seed)


class TestTensorNorm(unittest.TestCase):
    """``tensor_norm`` is the Frobenius norm, over real and complex input."""

    def test_norm_of_a_reshaped_identity(self):
        # ||I_4||_F = sqrt(4) = 2, and the tensor has 4 axes
        eye_tensor = np.eye(4).reshape(2, 2, 2, 2)
        npt.assert_allclose(utils.tensor_norm(eye_tensor), 2.0, rtol=RTOL)

    def test_norm_of_an_all_ones_tensor(self):
        npt.assert_allclose(
            utils.tensor_norm(np.ones((3, 3, 3))), np.sqrt(27.0), rtol=RTOL
        )

    def test_norm_of_a_zero_tensor_is_zero(self):
        npt.assert_allclose(utils.tensor_norm(np.zeros((5, 5, 5, 5))), 0.0, atol=ATOL)

    def test_norm_of_a_complex_tensor_uses_the_conjugate(self):
        """sum(x * x) would give sqrt(9 - 16); the norm must be sqrt(9 + 16)."""
        npt.assert_allclose(utils.tensor_norm(np.array([3.0, 4.0j])), 5.0, rtol=RTOL)


class TestTruncatedBond(unittest.TestCase):
    """``truncated_bond`` bounds the discarded tail, floored at 1, capped by rank."""

    SINGULAR_VALUES = np.array([4.0, 2.0, 1.0, 0.5])

    def test_keeps_the_smallest_rank_whose_tail_fits_the_budget(self):
        # dropping the last value costs 0.5, which fits; dropping the last two
        # costs sqrt(1 + 0.25) = 1.118, which does not
        self.assertEqual(utils.truncated_bond(self.SINGULAR_VALUES, 1.0), 3)

    def test_bounds_the_tail_rather_than_each_singular_value(self):
        """Many values just below delta must be kept, not discarded together.

        Thresholding on individual values would keep none of them and leave a
        discarded tail of ``sqrt(m) * delta`` -- the overshoot this criterion
        exists to prevent.
        """
        delta = 0.1
        singular_values = np.concatenate([[1.0], np.full(100, 0.099)])

        bond = utils.truncated_bond(singular_values, delta)

        # exactly one value can be dropped: 0.099 fits the budget, two do not
        self.assertEqual(bond, 100)
        self.assertLessEqual(np.linalg.norm(singular_values[bond:]), delta)
        self.assertGreater(np.linalg.norm(singular_values[bond - 1 :]), delta)

    def test_discards_a_tail_that_fits_the_budget(self):
        self.assertEqual(utils.truncated_bond(np.array([1.0, 0.05]), 0.1), 1)
        self.assertEqual(utils.truncated_bond(np.array([1.0, 0.5]), 0.1), 2)

    def test_floors_at_one_when_every_value_is_below_the_threshold(self):
        self.assertEqual(utils.truncated_bond(self.SINGULAR_VALUES, 100.0), 1)

    def test_is_capped_by_max_rank(self):
        self.assertEqual(utils.truncated_bond(self.SINGULAR_VALUES, 0.0, max_rank=2), 2)

    def test_rejects_a_non_positive_max_rank(self):
        for max_rank in (0, -3):
            with self.subTest(max_rank=max_rank):
                with self.assertRaises(ValueError):
                    utils.truncated_bond(self.SINGULAR_VALUES, 1.0, max_rank=max_rank)

    def test_discards_a_noise_level_tail_even_with_a_zero_budget(self):
        """A rank-deficient spectrum carries noise where its zeros are."""
        self.assertEqual(utils.truncated_bond(np.array([1.0, 1e-17, 3e-18]), 0.0), 1)

    def test_the_bond_does_not_follow_what_lapack_left_in_the_noise(self):
        """Four forms of a mathematically zero tail give the same bond."""
        # forms different BLAS builds report; before the cutoff: bonds 3, 1, 2, 3
        for tail in ([1e-17, 3e-18], [0.0, 0.0], [5e-17, 0.0], [1e-20, 1e-160]):
            with self.subTest(tail=tail):
                values = np.array([1.0, *tail])
                self.assertEqual(utils.truncated_bond(values, 0.0), 1)

    def test_the_noise_cutoff_is_relative_to_the_largest_value(self):
        """The cutoff is a fraction of ``singular_values[0]``, not an absolute."""
        # for two values led by 1e8 the cutoff is 4.4e-08
        self.assertEqual(utils.truncated_bond(np.array([1e8, 1e-9]), 0.0), 1)
        self.assertEqual(utils.truncated_bond(np.array([1e8, 1e-5]), 0.0), 2)

    def test_the_noise_cutoff_leaves_a_well_scaled_spectrum_alone(self):
        """For this spectrum the cutoff is 3.6e-15 and the smallest value 0.5."""
        self.assertEqual(utils.truncated_bond(self.SINGULAR_VALUES, 0.0), 4)


class TestOrthonormalize(unittest.TestCase):
    """QR sweeps must orthonormalize without changing the tensor."""

    def test_left_sweep_makes_every_swept_core_left_orthonormal(self):
        train = _reference_train(1)
        cores = utils.left_orthonormalize(list(train.cores), 0, 2)
        result = tensor_train.TensorTrain(cores)

        for index, core in enumerate(result.cores[:-1]):
            with self.subTest(core=index):
                left, row, col, right = core.shape
                matrix = core.reshape(left * row * col, right)
                npt.assert_allclose(
                    matrix.T @ matrix, np.eye(right), rtol=RTOL, atol=ATOL
                )

    def test_left_sweep_preserves_the_tensor(self):
        train = _reference_train(2)
        reference = train.get_full_tensor()
        result = tensor_train.TensorTrain(
            utils.left_orthonormalize(list(train.cores), 0, 2)
        )
        npt.assert_allclose(result.get_full_tensor(), reference, rtol=RTOL, atol=ATOL)

    def test_right_sweep_makes_every_swept_core_right_orthonormal(self):
        train = _reference_train(3)
        cores = utils.right_orthonormalize(list(train.cores), 2, 0)
        result = tensor_train.TensorTrain(cores)

        for index, core in enumerate(result.cores[1:], start=1):
            with self.subTest(core=index):
                left, row, col, right = core.shape
                matrix = core.reshape(left, row * col * right)
                npt.assert_allclose(
                    matrix @ matrix.T, np.eye(left), rtol=RTOL, atol=ATOL
                )

    def test_right_sweep_preserves_the_tensor(self):
        train = _reference_train(4)
        reference = train.get_full_tensor()
        result = tensor_train.TensorTrain(
            utils.right_orthonormalize(list(train.cores), 2, 0)
        )
        npt.assert_allclose(result.get_full_tensor(), reference, rtol=RTOL, atol=ATOL)


class TestSvdSweeps(unittest.TestCase):
    """SVD sweeps truncate to a requested rank, exactly when nothing is cut."""

    def test_left_svd_at_machine_precision_preserves_the_tensor(self):
        train = _reference_train(5)
        reference = train.get_full_tensor()
        result = tensor_train.TensorTrain(utils.left_svd(list(train.cores), TEST_DELTA))
        npt.assert_allclose(result.get_full_tensor(), reference, rtol=RTOL, atol=ATOL)

    def test_right_svd_at_machine_precision_preserves_the_tensor(self):
        train = _reference_train(6)
        reference = train.get_full_tensor()
        result = tensor_train.TensorTrain(
            utils.right_svd(list(train.cores), TEST_DELTA)
        )
        npt.assert_allclose(result.get_full_tensor(), reference, rtol=RTOL, atol=ATOL)

    def test_left_svd_respects_max_rank(self):
        train = _reference_train(7)
        result = tensor_train.TensorTrain(
            utils.left_svd(list(train.cores), TEST_DELTA, 3)
        )
        self.assertLessEqual(max(result.bonds), 3)

    def test_right_svd_respects_max_rank(self):
        train = _reference_train(8)
        result = tensor_train.TensorTrain(
            utils.right_svd(list(train.cores), TEST_DELTA, 3)
        )
        self.assertLessEqual(max(result.bonds), 3)

    def test_left_svd_floors_bonds_at_one(self):
        """An error above every singular value must still leave a valid train."""
        train = _reference_train(9).orthonormalize("right")
        result = tensor_train.TensorTrain(utils.left_svd(list(train.cores), 1e3))
        self.assertEqual(result.bonds, [1, 1, 1, 1])

    def test_right_svd_floors_bonds_at_one(self):
        train = _reference_train(10).orthonormalize("left")
        result = tensor_train.TensorTrain(utils.right_svd(list(train.cores), 1e3))
        self.assertEqual(result.bonds, [1, 1, 1, 1])

    def test_svd_sweeps_reject_a_non_positive_max_rank(self):
        train = _reference_train(11)
        for sweep in (utils.left_svd, utils.right_svd):
            with self.subTest(sweep=sweep.__name__):
                with self.assertRaises(ValueError):
                    sweep(list(train.cores), TEST_DELTA, 0)

    def test_sweeps_do_not_rebind_the_caller_s_list(self):
        """Handing over a copy must leave the original train's list intact.

        The helpers rebind list elements, so a caller that wants to keep its
        train has to pass ``list(cores)`` - this pins that contract down.
        """
        train = _reference_train(12)
        original_cores = list(train.cores)
        reference = train.get_full_tensor()

        utils.left_svd(list(train.cores), TEST_DELTA)

        self.assertEqual([id(c) for c in train.cores], [id(c) for c in original_cores])
        npt.assert_array_equal(train.get_full_tensor(), reference)
