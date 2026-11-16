# Copyright (c) 2026 QBoard LLC. Licensed under the AGPLv3. See LICENSE.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the tensor-train utility functions.

Every train here is seeded: these helpers are compared against exact
expectations, so an unlucky unseeded draw would be an unreproducible failure.

The four sweep helpers rebind the elements of the list they are given, which is
how the callers in :mod:`ttnumpy.tensor_train` use them. Tests therefore hand
over ``list(tt.cores)`` and keep ``tt`` itself as the untouched reference.
"""

import tracemalloc
import unittest
from unittest import mock

import numpy as np
import numpy.testing as npt
import scipy.linalg

from ttnumpy import tensor_train
from ttnumpy import utils
from ttnumpy.logger_config import PACKAGE_LOGGER_NAME

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

    def test_the_noise_cutoff_follows_the_precision_of_the_values(self):
        """The cutoff uses the machine epsilon of the singular values' dtype."""
        # for two values led by 1 the cutoff is 4.4e-16 in float64, 2.4e-07 in
        # float32; integers count as float64
        for dtype, bond in ((np.float64, 2), (np.float32, 1)):
            with self.subTest(dtype=np.dtype(dtype).name):
                values = np.array([1.0, 1e-9], dtype=dtype)
                self.assertEqual(utils.truncated_bond(values, 0.0), bond)
        self.assertEqual(utils.truncated_bond(np.array([3, 1]), 0.0), 2)


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


def _doubled_trains():
    """Name -> (train of bond 3, the same tensor doubled and stored with bond 6)."""
    rng = np.random.default_rng(20)
    vector = tensor_train.rand([2] * 6, ranks=3, seed=21)
    operator = tensor_train.rand([(2, 3)] * 4, ranks=3, seed=22)
    trains = {
        "TT": vector,
        "TTM": operator,
        "complex": tensor_train.TensorTrain(
            [
                core * np.exp(2j * np.pi * rng.uniform(size=core.shape))
                for core in vector.cores
            ]
        ),
        "float32": tensor_train.TensorTrain(
            [core.astype(np.float32) for core in vector.cores]
        ),
    }
    return {name: (train, train.add(train)) for name, train in trains.items()}


_RANDOMIZED_SWEEPS = (
    utils.left_randomized_orthonormalize,
    utils.right_randomized_orthonormalize,
)


class TestRandomizedOrthonormalize(unittest.TestCase):
    """Sketched sweeps return an orthonormal train with bonds within the rank."""

    def test_sweeps_recover_a_train_stored_with_larger_bonds(self):
        """A sweep of rank 3 returns a tensor of TT ranks 3 stored with bonds 6."""
        for name, (train, doubled) in _doubled_trains().items():
            for sweep in _RANDOMIZED_SWEEPS:
                with self.subTest(train=name, sweep=sweep.__name__):
                    rtol = 1e-4 if name == "float32" else RTOL
                    reference = 2 * train.get_full_tensor()

                    cores = sweep(list(doubled.cores), 3, np.random.default_rng(0))
                    result = tensor_train.TensorTrain(cores)

                    self.assertLessEqual(result.max_rank, 3)
                    self.assertEqual(cores[0].dtype, train.cores[0].dtype)
                    npt.assert_allclose(
                        result.get_full_tensor(),
                        reference,
                        rtol=rtol,
                        atol=rtol * np.abs(reference).max(),
                    )

    def test_left_sweep_makes_every_core_but_the_last_left_orthonormal(self):
        _, doubled = _doubled_trains()["complex"]
        cores = utils.left_randomized_orthonormalize(
            list(doubled.cores), 4, np.random.default_rng(1)
        )

        for index, core in enumerate(cores[:-1]):
            with self.subTest(core=index):
                matrix = core.reshape(-1, core.shape[3])
                npt.assert_allclose(
                    matrix.conj().T @ matrix,
                    np.eye(core.shape[3]),
                    rtol=RTOL,
                    atol=ATOL,
                )

    def test_right_sweep_makes_every_core_but_the_first_right_orthonormal(self):
        _, doubled = _doubled_trains()["complex"]
        cores = utils.right_randomized_orthonormalize(
            list(doubled.cores), 4, np.random.default_rng(2)
        )

        for index, core in enumerate(cores[1:], start=1):
            with self.subTest(core=index):
                matrix = core.reshape(core.shape[0], -1)
                npt.assert_allclose(
                    matrix @ matrix.conj().T,
                    np.eye(core.shape[0]),
                    rtol=RTOL,
                    atol=ATOL,
                )

    def test_bonds_stay_within_the_full_rank_bound(self):
        """A bond exceeds neither the rank nor the mode sizes on either side of it."""
        _, doubled = _doubled_trains()["TT"]
        for sweep in _RANDOMIZED_SWEEPS:
            with self.subTest(sweep=sweep.__name__):
                cores = sweep(list(doubled.cores), 5, np.random.default_rng(3))

                self.assertEqual(
                    tensor_train.TensorTrain(cores).bonds, [1, 2, 4, 5, 4, 2, 1]
                )

    def test_sweeps_below_the_rank_of_the_train_keep_its_largest_part(self):
        """A sweep of rank 6 has under a tenth of the error of SVD rounding to rank 3.

        The train has bonds up to 9 and decaying singular values.
        """
        grid = np.indices([3] * 8).sum(axis=0)
        train = tensor_train.tt_svd(1.0 / (1.0 + grid), error=1e-14)
        rounded = train.truncate(max_rank=3)
        svd_error = rounded.add(train.multiply(-1)).tt_norm()
        for sweep in _RANDOMIZED_SWEEPS:
            with self.subTest(sweep=sweep.__name__):
                result = tensor_train.TensorTrain(
                    sweep(list(train.cores), 6, np.random.default_rng(7))
                )

                self.assertEqual(result.max_rank, 6)
                error = result.add(train.multiply(-1)).tt_norm()
                self.assertLess(error, 0.1 * svd_error)

    def test_sweeps_keep_a_train_of_600_cores(self):
        """A sweep of rank 16 returns a full-rank train of bond 16.

        The sketch of core ``i`` is a product of ``600 - i`` random matrices
        with entries of variance 1. It overflows unless each product is
        rescaled.
        """
        rng = np.random.default_rng(30)
        cores = [
            rng.standard_normal(
                (1 if index == 0 else 16, 2, 1, 1 if index == 599 else 16)
            )
            / np.sqrt(32)
            for index in range(600)
        ]
        train = tensor_train.TensorTrain(utils.right_orthonormalize(cores, 599, 0))
        for sweep in _RANDOMIZED_SWEEPS:
            with self.subTest(sweep=sweep.__name__):
                result = tensor_train.TensorTrain(
                    sweep(list(train.cores), 16, np.random.default_rng(6))
                )

                difference = result.add(train.multiply(-1)).tt_norm()
                self.assertLess(difference, 1e-10 * train.tt_norm())

    def test_equal_generators_give_equal_cores(self):
        _, doubled = _doubled_trains()["TT"]
        for sweep in _RANDOMIZED_SWEEPS:
            with self.subTest(sweep=sweep.__name__):
                first = sweep(list(doubled.cores), 4, np.random.default_rng(4))
                second = sweep(list(doubled.cores), 4, np.random.default_rng(4))

                for one, other in zip(first, second):
                    npt.assert_array_equal(one, other)

    def test_sweeps_reject_a_non_positive_rank(self):
        _, doubled = _doubled_trains()["TT"]
        for sweep in _RANDOMIZED_SWEEPS:
            with self.subTest(sweep=sweep.__name__):
                with self.assertRaisesRegex(ValueError, "rank must be positive"):
                    sweep(list(doubled.cores), 0, np.random.default_rng(5))


def _hadamard_pairs():
    """Name -> two trains with equal modes, bonds 3 and 2."""
    rng = np.random.default_rng(40)
    vector = tensor_train.rand([2] * 6, ranks=3, seed=41)
    return {
        "TT": (vector, tensor_train.rand([2] * 6, ranks=2, seed=42)),
        "TTM": (
            tensor_train.rand([(2, 3)] * 4, ranks=3, seed=43),
            tensor_train.rand([(2, 3)] * 4, ranks=2, seed=44),
        ),
        "complex": (
            tensor_train.TensorTrain(
                [
                    core * np.exp(2j * np.pi * rng.uniform(size=core.shape))
                    for core in vector.cores
                ]
            ),
            tensor_train.rand([2] * 6, ranks=2, seed=45),
        ),
    }


class TestRandomizedHadamard(unittest.TestCase):
    """The sketched sweep of an element-wise product, read from its two factors."""

    def test_equals_the_left_sweep_of_the_built_product(self):
        """Rank 3 is below the bonds of the product, which are up to 6."""
        for name, (lhs, rhs) in _hadamard_pairs().items():
            with self.subTest(trains=name):
                built = utils.left_randomized_orthonormalize(
                    list(lhs.multiply(rhs).cores), 3, np.random.default_rng(0)
                )
                expected = tensor_train.TensorTrain(built).get_full_tensor()

                cores = utils.randomized_hadamard(
                    lhs.cores, rhs.cores, 3, np.random.default_rng(0)
                )

                self.assertEqual(cores[0].dtype, built[0].dtype)
                npt.assert_allclose(
                    tensor_train.TensorTrain(cores).get_full_tensor(),
                    expected,
                    rtol=RTOL,
                    atol=RTOL * np.abs(expected).max(),
                )

    def test_rank_of_the_product_gives_the_product(self):
        for name, (lhs, rhs) in _hadamard_pairs().items():
            with self.subTest(trains=name):
                expected = lhs.get_full_tensor() * rhs.get_full_tensor()

                cores = utils.randomized_hadamard(
                    lhs.cores, rhs.cores, 6, np.random.default_rng(1)
                )

                npt.assert_allclose(
                    tensor_train.TensorTrain(cores).get_full_tensor(),
                    expected,
                    rtol=RTOL,
                    atol=RTOL * np.abs(expected).max(),
                )

    def test_every_core_but_the_last_is_left_orthonormal(self):
        lhs, rhs = _hadamard_pairs()["complex"]
        cores = utils.randomized_hadamard(
            lhs.cores, rhs.cores, 4, np.random.default_rng(2)
        )

        for index, core in enumerate(cores[:-1]):
            with self.subTest(core=index):
                matrix = core.reshape(-1, core.shape[3])
                npt.assert_allclose(
                    matrix.conj().T @ matrix,
                    np.eye(core.shape[3]),
                    rtol=RTOL,
                    atol=ATOL,
                )

    def test_peak_memory_is_below_one_core_of_the_product(self):
        """Factors of bond 32 have a product with cores of 2 * 1024**2 entries."""
        lhs = tensor_train.rand([2] * 12, ranks=32, seed=46)
        rhs = tensor_train.rand([2] * 12, ranks=32, seed=47)
        product_core = 8 * 2 * 1024**2

        tracemalloc.start()
        utils.randomized_hadamard(lhs.cores, rhs.cores, 8, np.random.default_rng(3))
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        self.assertLess(peak, product_core / 4)

    def test_keeps_a_product_of_800_cores(self):
        """A sweep of rank 16 returns the product of two factors of bond 4.

        The sketch of core ``i`` is a product of ``800 - i`` random matrices
        with entries of variance 1. It overflows unless each product is
        rescaled.
        """
        factors = []
        for seed in (48, 49):
            rng = np.random.default_rng(seed)
            cores = [
                rng.standard_normal(
                    (1 if index == 0 else 4, 2, 1, 1 if index == 799 else 4)
                )
                / np.sqrt(8)
                for index in range(800)
            ]
            factors.append(
                tensor_train.TensorTrain(utils.right_orthonormalize(cores, 799, 0))
            )
        product = factors[0].multiply(factors[1])

        result = tensor_train.TensorTrain(
            utils.randomized_hadamard(
                factors[0].cores, factors[1].cores, 16, np.random.default_rng(4)
            )
        )

        difference = result.add(product.multiply(-1)).tt_norm()
        self.assertLess(difference, 1e-10 * product.tt_norm())

    def test_rejects_a_non_positive_rank(self):
        lhs, rhs = _hadamard_pairs()["TT"]
        with self.assertRaisesRegex(ValueError, "rank must be positive"):
            utils.randomized_hadamard(lhs.cores, rhs.cores, 0, np.random.default_rng(5))


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


def _svd_failing_in_gesdd(failures: int):
    """``scipy.linalg.svd`` whose first ``failures`` gesdd calls do not converge.

    Returns the replacement and the list of drivers it is called with.
    """
    drivers = []

    def svd(matrix, **kwargs):
        driver = kwargs.get("lapack_driver", "gesdd")
        drivers.append(driver)
        if driver == "gesdd" and drivers.count("gesdd") <= failures:
            raise scipy.linalg.LinAlgError("SVD did not converge")
        return scipy.linalg.svd(matrix, **kwargs)

    return svd, drivers


class TestSvdFallback(unittest.TestCase):
    """A ``gesdd`` convergence failure in an SVD sweep is repeated with ``gesvd``."""

    def test_only_the_failed_unfolding_is_repeated_with_gesvd(self):
        train = _reference_train(13)
        reference = train.get_full_tensor()
        for sweep in (utils.left_svd, utils.right_svd):
            with self.subTest(sweep=sweep.__name__):
                svd, drivers = _svd_failing_in_gesdd(failures=1)
                with (
                    mock.patch.object(utils, "svd", svd),
                    self.assertLogs(PACKAGE_LOGGER_NAME, level="WARNING") as captured,
                ):
                    cores = sweep(list(train.cores), TEST_DELTA)

                # three cores give two unfoldings; the second one is not retried
                self.assertEqual(drivers, ["gesdd", "gesvd", "gesdd"])
                self.assertEqual(len(captured.records), 1)
                result = tensor_train.TensorTrain(cores)
                npt.assert_allclose(
                    result.get_full_tensor(), reference, rtol=RTOL, atol=ATOL
                )

    def test_a_failure_on_non_finite_entries_raises_value_error(self):
        """NaN and inf entries are reported instead of decomposed with gesvd."""
        for value in (np.nan, np.inf):
            with self.subTest(value=value):
                cores = list(_reference_train(14).cores)
                cores[0] = cores[0].copy()
                cores[0][0, 1, 0, 2] = value
                svd, drivers = _svd_failing_in_gesdd(failures=1)
                # LinAlgError subclasses ValueError, so the message is checked
                with mock.patch.object(utils, "svd", svd):
                    with self.assertRaisesRegex(ValueError, "NaN or infinite"):
                        utils.left_svd(cores, TEST_DELTA)
                self.assertEqual(drivers, ["gesdd"])
