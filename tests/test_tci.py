# Copyright (c) 2026 QBoard LLC. Licensed under the AGPLv3. See LICENSE.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the tci driver."""

import unittest

import numpy as np
import numpy.testing as npt
from tci_test_utils import RATIONAL_DIMS, dense, rational

from ttnumpy import tensor_train
from ttnumpy.interpolation import black_box, tci, tci_form


def _two_deltas(indices) -> float:
    """1 at the all-zero and the all-one configuration, 0 elsewhere."""
    return float(all(i == 0 for i in indices) or all(i == 1 for i in indices))


def _quantics_peak(indices) -> float:
    """0.1 plus a Gaussian peak of width 0.005 at x = 0.3, x read from the bits."""
    x = sum(bit / 2.0 ** (position + 1) for position, bit in enumerate(indices))
    return 0.1 + np.exp(-(((x - 0.3) / 0.005) ** 2))


def _heaviside_corner(indices) -> float:
    """1 plus the product of Heaviside steps at 0, on ten nodes per variable.

    The nodes span [-1, 1] and miss 0, so the product is 1 where every
    variable is positive and 0 elsewhere.
    """
    nodes = np.linspace(-1.0, 1.0, 10)
    return 1.0 + float(all(nodes[i] > 0.0 for i in indices))


def _train_error(tensor_ci: tci_form.TensorCI, full: np.ndarray) -> float:
    """Largest deviation of the contracted train from ``full``."""
    return np.abs(tensor_ci.to_tensor_train().get_full_tensor() - full).max()


class TestTci(unittest.TestCase):
    """The driver sweeps until the ranks settle within the error threshold."""

    def test_finds_ranks_above_the_local_dimension(self):
        """The middle ranks, 10, exceed d = 6 and take more than one half-sweep.

        The first half-sweep exhausts slices of six columns, with zero error.
        """
        error_threshold = 1e-12
        full = dense(rational, RATIONAL_DIMS)

        tensor_ci, info = tci.tci(
            rational, RATIONAL_DIMS, [[0] * 5], error_threshold, return_info=True
        )

        self.assertTrue(info.converged)
        self.assertEqual(info.num_half_sweeps_run, 3)
        self.assertEqual(info.max_ranks, (6, 10, 10))
        self.assertEqual(info.pivot_errors[0], 0.0)
        self.assertLessEqual(max(info.pivot_errors[1:]), error_threshold)
        self.assertEqual(tensor_ci.pivots.ranks, [1, 6, 10, 10, 6, 1])
        error = np.abs(tensor_ci.to_tensor_train().get_full_tensor() - full).max()
        self.assertLess(error, 10 * error_threshold * np.abs(full).max())

    def test_reproduces_an_exactly_low_rank_train(self):
        train = tensor_train.rand([3, 4, 3, 4, 3], ranks=4, seed=7)
        full = train.get_full_tensor()

        tensor_ci, info = tci.tci(
            black_box.tt_as_black_box(train),
            (3, 4, 3, 4, 3),
            [[0] * 5],
            1e-12,
            return_info=True,
        )

        self.assertTrue(info.converged)
        self.assertEqual(tensor_ci.pivots.ranks, [1, 3, 4, 4, 3, 1])
        npt.assert_allclose(
            tensor_ci.to_tensor_train().get_full_tensor(), full, rtol=1e-10, atol=1e-12
        )

    def test_the_error_threshold_is_relative_to_the_sampled_values(self):
        """Scaling by a power of two is exact, so the runs match pivot for pivot."""
        small, small_info = tci.tci(
            rational, RATIONAL_DIMS, [[0] * 5], return_info=True
        )
        large, large_info = tci.tci(
            lambda idx: 2.0**40 * rational(idx),
            RATIONAL_DIMS,
            [[0] * 5],
            return_info=True,
        )

        for bond in range(1, 5):
            with self.subTest(bond=bond):
                self.assertEqual(large.pivots.left(bond), small.pivots.left(bond))
                self.assertEqual(large.pivots.right(bond), small.pivots.right(bond))
        self.assertEqual(large_info.pivot_errors, small_info.pivot_errors)

    def test_single_precision_and_complex_values(self):
        nodes = np.linspace(-1.0, 1.0, 8)
        cases = {
            # the default error_threshold, 1e-7, lies below float32 precision
            "float32": (
                lambda idx: np.float32(np.sin(sum(nodes[i] for i in idx))),
                1e-5,
            ),
            "complex": (
                lambda idx: np.exp(1j * sum(nodes[i] for i in idx))
                + np.exp(2j * sum(nodes[i] for i in idx)),
                1e-7,
            ),
        }
        for name, (function, error_threshold) in cases.items():
            with self.subTest(values=name):
                tensor_ci, info = tci.tci(
                    function, (8, 8, 8, 8), [[0] * 4], error_threshold, return_info=True
                )
                self.assertTrue(info.converged)
                self.assertEqual(tensor_ci.pivots.ranks, [1, 2, 2, 2, 1])

    def test_a_binding_max_rank_is_reported_as_not_converged(self):
        """Rank 3 is below the 10 that the error threshold needs."""
        error_threshold = 1e-12
        with self.assertLogs("ttnumpy", level="WARNING"):
            tensor_ci, info = tci.tci(
                rational,
                RATIONAL_DIMS,
                [[0] * 5],
                error_threshold,
                max_rank=3,
                return_info=True,
            )

        self.assertLessEqual(max(tensor_ci.pivots.ranks), 3)
        self.assertFalse(info.converged)
        self.assertGreater(info.pivot_errors[-1], error_threshold)

    def test_converges_while_the_pivots_keep_changing(self):
        """Pivots that keep changing between half-sweeps do not block convergence."""
        rng = np.random.default_rng(8)
        coarse, fine = np.linspace(-1.0, 1.0, 5), np.linspace(0.0, 1.0, 16)
        cases = [
            (
                "tied entries",
                lambda idx: np.exp(-sum(coarse[i] ** 2 for i in idx))
                + 0.1 * np.cos(sum(coarse[i] for i in idx)),
                5,
                5,
                [1, 3, 3, 3, 3, 1],
            ),
            (
                "no ties",
                lambda idx: np.exp(-sum(fine[i] ** 2 for i in idx))
                * np.cos(5.0 * sum(fine[i] for i in idx))
                + 0.1 * np.prod([fine[i] + 0.5 for i in idx]),
                16,
                6,
                [1, 3, 3, 3, 3, 3, 1],
            ),
        ]
        for name, function, dim, n_sites, ranks in cases:
            with self.subTest(case=name):
                tensor_ci, info = tci.tci(
                    function,
                    (dim,) * n_sites,
                    [[dim // 2] * n_sites],
                    1e-10,
                    return_info=True,
                )
                self.assertTrue(info.converged)
                self.assertEqual(tensor_ci.pivots.ranks, ranks)
                train = tensor_ci.to_tensor_train()
                for index in rng.integers(0, dim, size=(200, n_sites)):
                    self.assertAlmostEqual(
                        train.get_element(list(index)), function(tuple(index)), 12
                    )

    def test_running_out_of_half_sweeps_is_reported(self):
        """One half-sweep leaves the ranks at d = 6, below the true 10."""
        with self.assertLogs("ttnumpy", level="WARNING"):
            _, info = tci.tci(
                rational,
                RATIONAL_DIMS,
                [[0] * 5],
                1e-12,
                max_half_sweeps=1,
                return_info=True,
            )

        self.assertFalse(info.converged)
        self.assertEqual(info.num_half_sweeps_run, 1)

    def test_returns_the_tensor_ci_alone_unless_info_is_asked_for(self):
        result = tci.tci(rational, RATIONAL_DIMS, [[0] * 5])
        self.assertIsInstance(result, tci_form.TensorCI)

    def test_uses_a_given_black_box_as_it_is(self):
        box = black_box.BlackBox(rational)

        tensor_ci = tci.tci(box, RATIONAL_DIMS, [[0] * 5])

        self.assertIs(tensor_ci.box, box)

    def test_starts_from_every_initial_pivot(self):
        """From one delta the other is missed; from both, the train is exact."""
        dims = (2,) * 6
        full = dense(_two_deltas, dims)

        one = tci.tci(_two_deltas, dims, [[0] * 6])
        both = tci.tci(_two_deltas, dims, [[0] * 6, [1] * 6])

        self.assertEqual(_train_error(one, full), 1.0)
        self.assertEqual(both.pivots.ranks, [1, 2, 2, 2, 2, 2, 1])
        self.assertLess(_train_error(both, full), 1e-12)

    def test_rejects_initial_pivots_where_the_tensor_vanishes(self):
        nodes = np.linspace(0.0, 1.0, 4)

        def product(indices) -> float:
            return float(np.prod(nodes[list(indices)]))

        with self.assertRaises(ValueError):
            tci.tci(product, (4, 4, 4), [[0, 0, 0], [0, 1, 2]])
        tensor_ci = tci.tci(product, (4, 4, 4), [[0, 0, 0], [1, 1, 1]])

        self.assertEqual(tensor_ci.pivots.ranks, [1, 1, 1, 1])

    def test_rejects_invalid_arguments_before_evaluating(self):
        for dims, initial_pivots, options in [
            (RATIONAL_DIMS, [[0] * 5], {"error_threshold": -1.0}),
            (RATIONAL_DIMS, [[0] * 5], {"error_threshold": float("nan")}),
            (RATIONAL_DIMS, [[0] * 5], {"max_rank": 0}),
            (RATIONAL_DIMS, [[0] * 5], {"max_half_sweeps": 0}),
            (RATIONAL_DIMS, [[0] * 4], {}),
            (RATIONAL_DIMS, [], {}),
            (RATIONAL_DIMS, [[0] * 5, [0] * 4], {}),
            (RATIONAL_DIMS, [[0, 0, 0, 0, 6]], {}),
            # the two-site slices around a site of dimension 1 cannot grow a rank
            ((6, 1, 6, 6, 6), [[0] * 5], {}),
        ]:
            with self.subTest(
                dims=dims, initial_pivots=initial_pivots, options=options
            ):
                box = black_box.BlackBox(rational)
                with self.assertRaises(ValueError):
                    tci.tci(box, dims, initial_pivots, **options)
                self.assertEqual(box.n_evals, 0)

    def test_rejects_initial_pivots_that_are_not_integer_sequences(self):
        """A flat list is the single configuration of an older signature."""
        box = black_box.BlackBox(rational)
        for initial_pivots in ([[0.5] * 5], [0] * 5):
            with self.subTest(initial_pivots=initial_pivots):
                with self.assertRaises(TypeError):
                    tci.tci(box, RATIONAL_DIMS, initial_pivots)
        self.assertEqual(box.n_evals, 0)


class TestAddGlobalPivots(unittest.TestCase):
    """Global pivots reach features that 2-site sweeps alone miss."""

    def test_recovers_a_delta_that_tci_missed(self):
        """The two-delta example of the reference, Appendix B.3.5."""
        dims = (2,) * 6
        full = dense(_two_deltas, dims)
        tensor_ci = tci.tci(_two_deltas, dims, [[0] * 6])
        self.assertEqual(_train_error(tensor_ci, full), 1.0)

        result, info = tci.add_global_pivots(tensor_ci, [[1] * 6], return_info=True)

        self.assertIs(result, tensor_ci)
        self.assertTrue(info.converged)
        self.assertLess(_train_error(tensor_ci, full), 1e-12)

    def test_any_pivot_in_the_upper_region_of_a_step_recovers_it(self):
        """1 + H(x1) H(x2) H(x3), started where every variable is negative.

        Every 2-site slice through the start pins one variable to a negative
        node, so no slice reaches the region where all three are positive and
        tci converges on the constant 1.
        """
        dims = (10,) * 3
        full = dense(_heaviside_corner, dims)
        for pivot in ([9, 9, 9], [5, 6, 8], [5, 5, 5]):
            with self.subTest(pivot=pivot):
                tensor_ci, info = tci.tci(
                    _heaviside_corner, dims, [[0, 0, 0]], return_info=True
                )
                self.assertTrue(info.converged)
                self.assertEqual(_train_error(tensor_ci, full), 1.0)

                tci.add_global_pivots(tensor_ci, [pivot])

                self.assertEqual(tensor_ci.pivots.ranks, [1, 2, 2, 1])
                self.assertLess(_train_error(tensor_ci, full), 1e-12)

    def test_sweeps_grow_a_peak_from_one_pivot(self):
        """Inserting the centre of the peak gives rank 2; the sweeps grow it."""
        dims = (2,) * 10
        full = dense(_quantics_peak, dims)
        tensor_ci = tci.tci(_quantics_peak, dims, [[0] * 10])
        self.assertGreater(_train_error(tensor_ci, full), 0.5)
        # 307 / 1024 is the grid point nearest x = 0.3
        centre = [int(bit) for bit in np.binary_repr(307, 10)]

        tci.add_global_pivots(tensor_ci, [centre])

        self.assertGreater(max(tensor_ci.pivots.ranks), 2)
        self.assertLess(_train_error(tensor_ci, full), 1e-6 * np.abs(full).max())

    def test_zero_and_repeated_proposals_add_nothing(self):
        dims = (2,) * 6
        full = dense(_two_deltas, dims)
        tensor_ci = tci.tci(_two_deltas, dims, [[0] * 6, [1] * 6])
        ranks = tensor_ci.pivots.ranks

        tci.add_global_pivots(tensor_ci, [[0, 1, 0, 1, 0, 1], [1] * 6, [1] * 6])

        self.assertEqual(tensor_ci.pivots.ranks, ranks)
        self.assertLess(_train_error(tensor_ci, full), 1e-12)

    def test_returns_the_form_alone_unless_info_is_asked_for(self):
        tensor_ci = tci.tci(rational, RATIONAL_DIMS, [[0] * 5])

        self.assertIs(tci.add_global_pivots(tensor_ci, [[5] * 5]), tensor_ci)

    def test_rejects_invalid_arguments_before_evaluating(self):
        tensor_ci = tci.tci(rational, RATIONAL_DIMS, [[0] * 5])
        n_evals = tensor_ci.box.n_evals
        for configurations, options in [
            ([], {}),
            ([[0] * 4], {}),
            ([[0, 0, 0, 0, 6]], {}),
            ([[5] * 5], {"error_threshold": -1.0}),
            ([[5] * 5], {"max_rank": 0}),
            ([[5] * 5], {"max_half_sweeps": 0}),
        ]:
            with self.subTest(configurations=configurations, options=options):
                with self.assertRaises(ValueError):
                    tci.add_global_pivots(tensor_ci, configurations, **options)
        with self.assertRaises(TypeError):
            tci.add_global_pivots(tensor_ci, [[0.5] * 5])
        self.assertEqual(tensor_ci.box.n_evals, n_evals)


if __name__ == "__main__":
    unittest.main()
