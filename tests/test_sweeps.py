# Copyright (c) 2026 QBoard LLC. Licensed under the AGPLv3. See LICENSE.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the 2-site, 1-site and 0-site updates and their sweeps."""

import unittest

import numpy as np
import numpy.testing as npt
from tci_test_utils import RATIONAL_DIMS, dense, dense_box, rational

from ttnumpy.interpolation import black_box, sweeps, tci, tci_form

RTOL = 1e-12

DIMS = (2, 3, 2, 2)


def _product_tensor(seed: int) -> np.ndarray:
    """A rank-one tensor, the outer product of one vector per site."""
    rng = np.random.default_rng(seed)
    factors = [rng.standard_normal(dim) for dim in DIMS]
    tensor = factors[0]
    for factor in factors[1:]:
        tensor = np.multiply.outer(tensor, factor)
    return tensor


def _tensor_ci(box, configuration, dims=DIMS) -> tci_form.TensorCI:
    """A TCI form of rank one, with every pivot taken from ``configuration``."""
    pivots = tci_form.PivotLists.from_configuration(configuration)
    return tci_form.TensorCI(box, dims, pivots)


def _bond_pivots(tensor_ci: tci_form.TensorCI, bond: int) -> tuple:
    """The row and column pivots of ``tensor_ci`` at ``bond``."""
    return tensor_ci.pivots.left(bond), tensor_ci.pivots.right(bond)


def _sweep(tensor_ci: tci_form.TensorCI, half_sweeps: int, tolerance=0.0) -> None:
    """Alternate forward and backward half-sweeps of 2-site updates."""
    n_sites = len(tensor_ci.dims)
    forward, backward = range(1, n_sites), range(n_sites - 1, 0, -1)
    for half_sweep in range(half_sweeps):
        for bond in backward if half_sweep % 2 else forward:
            sweeps.two_site_update(tensor_ci, bond, tolerance)


def _swept_random_tensor_ci(seed: int) -> tci_form.TensorCI:
    """A random tensor over DIMS after three half-sweeps, at ranks [1, 2, 4, 2, 1]."""
    box = dense_box(np.random.default_rng(seed).standard_normal(DIMS))
    tensor_ci = _tensor_ci(box, [1, 0, 1, 1])
    _sweep(tensor_ci, 3)
    return tensor_ci


def _sine_box(dtype) -> black_box.BlackBox:
    """sin(x1 + ... + xn) on eight nodes per site, returned as ``dtype``."""
    nodes = np.linspace(-1.0, 1.0, 8)
    return black_box.BlackBox(lambda idx: dtype(np.sin(sum(nodes[i] for i in idx))))


def _unnested_rational_tensor_ci() -> tci_form.TensorCI:
    """The rational tensor after tci, with every pivot matrix cut at 1e-4.

    The cut leaves ranks [1, 5, 5, 5, 5, 1], below d = 6, and pivots that are
    not fully nested.
    """
    tensor_ci = tci.tci(rational, RATIONAL_DIMS, [[0] * 5], 1e-12)
    tolerance = 1e-4 * tensor_ci.box.max_modulus
    for bond in range(1, len(RATIONAL_DIMS)):
        sweeps.zero_site_update(tensor_ci, bond, tolerance)
    return tensor_ci


class TestTwoSiteUpdate(unittest.TestCase):
    """A 2-site update replaces one bond's pivots with those of its slice."""

    CONFIGURATION = [1, 0, 1, 1]

    def test_new_lists_are_nested_with_their_neighbours(self):
        box = dense_box(np.random.default_rng(2).standard_normal(DIMS))
        tensor_ci = _tensor_ci(box, self.CONFIGURATION)

        sweeps.two_site_update(tensor_ci, 2)

        self.assertTrue(tensor_ci.pivots.is_left_nested_at(2))
        self.assertTrue(tensor_ci.pivots.is_right_nested_at(2))

    def test_leaves_every_other_bond_alone(self):
        box = dense_box(np.random.default_rng(3).standard_normal(DIMS))
        tensor_ci = _tensor_ci(box, self.CONFIGURATION)
        others = [bond for bond in range(1, len(DIMS)) if bond != 2]
        before = {bond: _bond_pivots(tensor_ci, bond) for bond in others}

        sweeps.two_site_update(tensor_ci, 2)

        for bond in others:
            with self.subTest(bond=bond):
                self.assertEqual(_bond_pivots(tensor_ci, bond), before[bond])

    def test_a_product_tensor_needs_one_pivot(self):
        tensor_ci = _tensor_ci(dense_box(_product_tensor(4)), self.CONFIGURATION)

        result = sweeps.two_site_update(tensor_ci, 2, tolerance=1e-10)

        self.assertEqual(result.rank, 1)
        self.assertLess(result.pivot_error, 1e-10)

    def test_a_slice_below_tolerance_leaves_the_lists_untouched(self):
        """An empty bond would not describe a tensor train."""
        tensor_ci = _tensor_ci(dense_box(np.zeros(DIMS)), self.CONFIGURATION)
        before = _bond_pivots(tensor_ci, 2)

        result = sweeps.two_site_update(tensor_ci, 2, tolerance=1e-8)

        self.assertEqual(result.rank, 0)
        self.assertEqual(_bond_pivots(tensor_ci, 2), before)

    def test_costs_one_request_per_entry_of_the_two_site_slice(self):
        """Below rank d it differs from the 1-site and 0-site counts."""
        tensor_ci = _unnested_rational_tensor_ci()
        pivots, box = tensor_ci.pivots, tensor_ci.box
        entries = (
            len(pivots.left(1))
            * RATIONAL_DIMS[1]
            * RATIONAL_DIMS[2]
            * len(pivots.right(3))
        )
        n_requests = box.n_requests

        sweeps.two_site_update(tensor_ci, 2)

        self.assertEqual(box.n_requests - n_requests, entries)

    def test_rejects_a_bond_without_two_free_sites_before_sampling(self):
        box = dense_box(np.ones(DIMS))
        tensor_ci = _tensor_ci(box, self.CONFIGURATION)
        for bond in (0, len(DIMS)):
            with self.subTest(bond=bond):
                with self.assertRaises(ValueError):
                    sweeps.two_site_update(tensor_ci, bond)
        self.assertEqual(box.n_requests, 0)

    def test_a_zero_tolerance_keeps_no_rounding_noise(self):
        """The noise level follows the precision of the sampled values.

        sin(x1 + x2 + x3 + x4) has rank 2 at every bond, and these sweeps reach
        it with the pivots fully nested.
        """
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=dtype):
                box = _sine_box(dtype)
                tensor_ci = _tensor_ci(box, [7, 7, 7, 0], dims=(8, 8, 8, 8))

                _sweep(tensor_ci, 4)

                self.assertEqual(tensor_ci.pivots.ranks, [1, 2, 2, 2, 1])
                self.assertTrue(tensor_ci.pivots.is_fully_nested())
                self.assertLess(box.n_evals, 8**4)


class TestOneSiteUpdate(unittest.TestCase):
    """A 1-site update replaces a bond of a site by pivots of its site tensor."""

    def test_forward_rows_extend_the_bond_before_and_columns_are_kept(self):
        tensor_ci = _unnested_rational_tensor_ci()
        for site in range(len(RATIONAL_DIMS) - 1):
            with self.subTest(site=site):
                old_right = set(tensor_ci.pivots.right(site + 1))

                result = sweeps.one_site_update(tensor_ci, site, forward=True)

                self.assertEqual(result.bond, site + 1)
                self.assertTrue(tensor_ci.pivots.is_left_nested_at(site + 1))
                self.assertLessEqual(set(tensor_ci.pivots.right(site + 1)), old_right)

    def test_backward_columns_extend_the_bond_after_and_rows_are_kept(self):
        tensor_ci = _unnested_rational_tensor_ci()
        for site in range(len(RATIONAL_DIMS) - 1, 0, -1):
            with self.subTest(site=site):
                old_left = set(tensor_ci.pivots.left(site))

                result = sweeps.one_site_update(tensor_ci, site, forward=False)

                self.assertEqual(result.bond, site)
                self.assertTrue(tensor_ci.pivots.is_right_nested_at(site))
                self.assertLessEqual(set(tensor_ci.pivots.left(site)), old_left)

    def test_costs_one_request_per_entry_of_the_site_tensor(self):
        """Below rank d it differs from the 2-site and 0-site counts."""
        tensor_ci = _unnested_rational_tensor_ci()
        box = tensor_ci.box
        for site, forward in ((1, True), (3, False)):
            with self.subTest(site=site, forward=forward):
                pivots = tensor_ci.pivots
                entries = (
                    len(pivots.left(site))
                    * RATIONAL_DIMS[site]
                    * len(pivots.right(site + 1))
                )
                n_requests = box.n_requests

                sweeps.one_site_update(tensor_ci, site, forward)

                self.assertEqual(box.n_requests - n_requests, entries)

    def test_rejects_a_site_outside_its_direction_before_sampling(self):
        box = dense_box(np.ones(DIMS))
        tensor_ci = _tensor_ci(box, [1, 0, 1, 1])
        for site, forward in ((len(DIMS) - 1, True), (-1, True), (0, False)):
            with self.subTest(site=site, forward=forward):
                with self.assertRaises(ValueError):
                    sweeps.one_site_update(tensor_ci, site, forward)
        self.assertEqual(box.n_requests, 0)


class TestZeroSiteUpdate(unittest.TestCase):
    """A 0-site update keeps the pivots that its bond's pivot matrix selects."""

    def test_a_swept_form_keeps_every_pivot_without_evaluating(self):
        """After a sweep each pivot matrix is cached and nonsingular."""
        tensor_ci = _swept_random_tensor_ci(2)
        ranks, box = tensor_ci.pivots.ranks, tensor_ci.box
        n_evals, n_requests = box.n_evals, box.n_requests

        for bond in range(1, len(DIMS)):
            sweeps.zero_site_update(tensor_ci, bond)

        self.assertEqual(tensor_ci.pivots.ranks, ranks)
        self.assertEqual(box.n_evals, n_evals)
        self.assertEqual(box.n_requests - n_requests, sum(r * r for r in ranks[1:-1]))

    def test_drops_a_linearly_dependent_pivot(self):
        """Two pivots of a rank-one tensor give a pivot matrix of rank one."""
        tensor = _product_tensor(4)
        first, second = (1, 0, 1, 1), (0, 2, 0, 1)
        pivots = tci_form.PivotLists(
            [
                ([first[:1]], [first[1:]]),
                ([first[:2], second[:2]], [first[2:], second[2:]]),
                ([first[:3]], [first[3:]]),
            ]
        )
        tensor_ci = tci_form.TensorCI(dense_box(tensor), DIMS, pivots)

        result = sweeps.zero_site_update(tensor_ci, 2)

        self.assertEqual(result.rank, 1)
        npt.assert_allclose(
            tensor_ci.to_tensor_train().get_full_tensor(), tensor, rtol=RTOL
        )

    def test_a_matrix_below_tolerance_leaves_the_lists_untouched(self):
        tensor_ci = _tensor_ci(dense_box(np.zeros(DIMS)), [1, 0, 1, 1])
        before = _bond_pivots(tensor_ci, 2)

        result = sweeps.zero_site_update(tensor_ci, 2, tolerance=1e-8)

        self.assertEqual(result.rank, 0)
        self.assertEqual(_bond_pivots(tensor_ci, 2), before)

    def test_rejects_a_bond_outside_the_internal_ones(self):
        tensor_ci = _tensor_ci(dense_box(np.ones(DIMS)), [1, 0, 1, 1])
        for bond in (0, len(DIMS)):
            with self.subTest(bond=bond):
                with self.assertRaises(ValueError):
                    sweeps.zero_site_update(tensor_ci, bond)


def _interpolated_rational(factor: float = 1.0) -> tci_form.TensorCI:
    """``factor`` times the rational tensor, interpolated to 1e-12."""
    return tci.tci(
        lambda indices: factor * rational(indices), RATIONAL_DIMS, [[0] * 5], 1e-12
    )


def _same_pivots(first: tci_form.TensorCI, second: tci_form.TensorCI) -> bool:
    """Whether two forms hold the same pivots at every bond."""
    return all(
        _bond_pivots(first, bond) == _bond_pivots(second, bond)
        for bond in range(1, len(first.dims))
    )


def _bumps(seed: int, dim: int, n_sites: int) -> tuple:
    """Three seeded Gaussian bumps on a constant, and the centre of the first.

    Returns:
        tuple: The function of a configuration, and a configuration near the
        centre of the first bump.
    """
    rng = np.random.default_rng(seed)
    centres = rng.uniform(0.0, dim - 1, size=(3, n_sites))
    widths = rng.uniform(0.1, 3.0, 3)
    heights = rng.uniform(-1.0, 1.0, 3)

    def function(indices) -> float:
        squares = ((np.asarray(indices, dtype=float) - centres) ** 2).sum(axis=1)
        return float(heights @ np.exp(-squares / widths) + 0.3)

    return function, [int(index) for index in np.round(centres[0])]


def _assert_rejects_invalid_thresholds(test, sweep) -> None:
    """``sweep(tensor_ci, error_threshold, max_rank)`` refuses before sampling."""
    swept = _swept_random_tensor_ci(2)
    unsampled = _tensor_ci(dense_box(np.ones(DIMS)), [1, 0, 1, 1])
    for tensor_ci, error_threshold, max_rank in (
        (swept, -1.0, None),
        (swept, float("nan"), None),
        (swept, 0.0, 0),
        (unsampled, 0.0, None),
    ):
        with test.subTest(error_threshold=error_threshold, max_rank=max_rank):
            n_requests = tensor_ci.box.n_requests
            with test.assertRaises(ValueError):
                sweep(tensor_ci, error_threshold, max_rank)
            test.assertEqual(tensor_ci.box.n_requests, n_requests)


def _assert_the_threshold_is_relative(test, sweep) -> None:
    """``sweep(tensor_ci, error_threshold, max_rank)`` scales by the samples.

    Scaling by a power of two is exact, so the forms match pivot for pivot.
    """
    small, large = _interpolated_rational(), _interpolated_rational(2.0**40)

    small_error = sweep(small, 1e-4, None)
    large_error = sweep(large, 1e-4, None)

    test.assertLess(max(small.pivots.ranks), 10)
    test.assertTrue(_same_pivots(small, large))
    test.assertEqual(small_error, large_error)


def _assert_max_rank_caps_every_bond(test, sweep) -> None:
    """``sweep(tensor_ci, error_threshold, max_rank)`` keeps at most max_rank."""
    tensor_ci = _interpolated_rational()

    pivot_error = sweep(tensor_ci, 0.0, 3)

    test.assertEqual(max(tensor_ci.pivots.ranks), 3)
    test.assertGreater(pivot_error, 0.0)


class TestOneSiteSweep(unittest.TestCase):
    """1-site sweeps repeat until the pivots are fully nested."""

    def test_leaves_the_pivots_fully_nested(self):
        """Nested pivots make the train equal the tensor on every site tensor."""
        tensor_ci = _unnested_rational_tensor_ci()
        ranks = tensor_ci.pivots.ranks
        self.assertFalse(tensor_ci.pivots.is_fully_nested())

        with self.assertNoLogs("ttnumpy", level="WARNING"):
            sweeps.one_site_sweep(tensor_ci)

        pivots = tensor_ci.pivots
        self.assertEqual(pivots.ranks, ranks)
        self.assertTrue(pivots.is_fully_nested())
        train = tensor_ci.to_tensor_train()
        for site, dim in enumerate(RATIONAL_DIMS):
            with self.subTest(site=site):
                configurations = [
                    row + column
                    for row in tci_form.append_site(pivots.left(site), dim)
                    for column in pivots.right(site + 1)
                ]
                npt.assert_allclose(
                    [train.get_element(list(index)) for index in configurations],
                    [rational(index) for index in configurations],
                    rtol=1e-10,
                )

    def test_the_threshold_is_relative_to_the_sampled_values(self):
        _assert_the_threshold_is_relative(self, sweeps.one_site_sweep)

    def test_max_rank_caps_every_bond(self):
        _assert_max_rank_caps_every_bond(self, sweeps.one_site_sweep)

    def test_rejects_invalid_thresholds_before_sampling(self):
        _assert_rejects_invalid_thresholds(self, sweeps.one_site_sweep)

    def test_rejects_a_non_positive_max_sweeps_before_sampling(self):
        tensor_ci = _swept_random_tensor_ci(2)
        n_requests = tensor_ci.box.n_requests

        with self.assertRaisesRegex(ValueError, "max_sweeps"):
            sweeps.one_site_sweep(tensor_ci, max_sweeps=0)

        self.assertEqual(tensor_ci.box.n_requests, n_requests)


class TestZeroSiteSweep(unittest.TestCase):
    """A 0-site sweep updates every bond from its own pivot matrix."""

    def test_at_the_interpolation_threshold_it_changes_nothing(self):
        """The pivot matrices of a converged run are all of full rank."""
        tensor_ci = _interpolated_rational()
        before = _interpolated_rational()

        pivot_error = sweeps.zero_site_sweep(tensor_ci, error_threshold=1e-12)

        self.assertTrue(_same_pivots(tensor_ci, before))
        self.assertEqual(pivot_error, 0.0)

    def test_a_looser_threshold_keeps_subsets_without_evaluating(self):
        tensor_ci = _interpolated_rational()
        before = _interpolated_rational()
        n_evals = tensor_ci.box.n_evals

        pivot_error = sweeps.zero_site_sweep(tensor_ci, error_threshold=1e-4)

        self.assertEqual(tensor_ci.pivots.ranks, [1, 5, 5, 5, 5, 1])
        self.assertLessEqual(pivot_error, 1e-4)
        self.assertEqual(tensor_ci.box.n_evals, n_evals)
        for bond in range(1, len(RATIONAL_DIMS)):
            with self.subTest(bond=bond):
                left, right = _bond_pivots(tensor_ci, bond)
                old_left, old_right = _bond_pivots(before, bond)
                self.assertLessEqual(set(left), set(old_left))
                self.assertLessEqual(set(right), set(old_right))

    def test_the_threshold_is_relative_to_the_sampled_values(self):
        _assert_the_threshold_is_relative(self, sweeps.zero_site_sweep)

    def test_max_rank_caps_every_bond(self):
        _assert_max_rank_caps_every_bond(self, sweeps.zero_site_sweep)

    def test_rejects_invalid_thresholds_before_sampling(self):
        _assert_rejects_invalid_thresholds(self, sweeps.zero_site_sweep)


class TestRecompress(unittest.TestCase):
    """Recompression lowers the ranks to a threshold and leaves them nested."""

    def test_lowers_the_ranks_within_the_threshold(self):
        error_threshold = 1e-4
        full = dense(rational, RATIONAL_DIMS)
        tensor_ci = _interpolated_rational()

        pivot_error = sweeps.recompress(tensor_ci, error_threshold)

        self.assertEqual(tensor_ci.pivots.ranks, [1, 5, 5, 5, 5, 1])
        self.assertTrue(tensor_ci.pivots.is_fully_nested())
        self.assertLessEqual(pivot_error, error_threshold)
        error = np.abs(tensor_ci.to_tensor_train().get_full_tensor() - full).max()
        self.assertLess(error, 10 * error_threshold * np.abs(full).max())

    def test_a_binding_max_rank_leaves_the_pivots_nested(self):
        tensor_ci = _interpolated_rational()

        pivot_error = sweeps.recompress(tensor_ci, 0.0, max_rank=3)

        self.assertEqual(tensor_ci.pivots.ranks, [1, 3, 3, 3, 3, 1])
        self.assertTrue(tensor_ci.pivots.is_fully_nested())
        self.assertGreater(pivot_error, 0.0)

    def test_sweeps_until_a_backward_half_keeps_every_rank(self):
        """Here one 1-site sweep still lowers ranks and leaves the pivots unnested."""
        function, pivot = _bumps(210, dim=2, n_sites=6)
        stopped, finished = (
            tci.tci(function, (2,) * 6, [pivot], 1e-12) for _ in range(2)
        )

        with self.assertLogs("ttnumpy", level="WARNING"):
            sweeps.recompress(stopped, 0.03, max_sweeps=1)
        pivot_error = sweeps.recompress(finished, 0.03)

        self.assertFalse(stopped.pivots.is_fully_nested())
        self.assertTrue(finished.pivots.is_fully_nested())
        self.assertLessEqual(pivot_error, 0.03)
