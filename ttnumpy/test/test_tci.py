# Copyright (c) 2026 QBoard LLC. Licensed under the AGPLv3. See LICENSE.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the two-site update and the tci driver.

Tensors here are written out or seeded, since the slice is compared entry by
entry against the function it samples.
"""

import unittest

import numpy as np
import numpy.testing as npt

from ttnumpy import tensor_train
from ttnumpy.interpolation import black_box, tci, tci_form

RTOL = 1e-12

DIMS = (2, 3, 2, 2)


def _dense_box(tensor: np.ndarray) -> black_box.BlackBox:
    """Sample a dense tensor by index."""
    return black_box.BlackBox(lambda indices: float(tensor[tuple(indices)]))


def _digits_box() -> black_box.BlackBox:
    """Return the configuration read as decimal digits, so entries spell it."""
    return black_box.BlackBox(
        lambda indices: float("".join(str(index) for index in indices))
    )


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
    """Alternate forward and backward half-sweeps, forward first."""
    n_sites = len(tensor_ci.dims)
    forward, backward = range(1, n_sites), range(n_sites - 1, 0, -1)
    for half_sweep in range(half_sweeps):
        for bond in backward if half_sweep % 2 else forward:
            tci.update_bond(tensor_ci, bond, tolerance)


def _sine_box(dtype) -> black_box.BlackBox:
    """sin(x1 + ... + xn) on eight nodes per site, returned as ``dtype``."""
    nodes = np.linspace(-1.0, 1.0, 8)
    return black_box.BlackBox(lambda idx: dtype(np.sin(sum(nodes[i] for i in idx))))


class TestTwoSiteSlice(unittest.TestCase):
    """The slice samples the tensor over the free sites at one bond."""

    CONFIGURATION = [1, 0, 1, 1]

    def test_entries_are_the_function_at_row_plus_column(self):
        tensor = np.random.default_rng(0).standard_normal(DIMS)
        tensor_ci = _tensor_ci(_dense_box(tensor), self.CONFIGURATION)

        matrix, rows, columns = tci.two_site_slice(tensor_ci, 2)

        for r, row in enumerate(rows):
            for c, column in enumerate(columns):
                with self.subTest(row=row, column=column):
                    npt.assert_allclose(matrix[r, c], tensor[row + column], rtol=RTOL)

    def test_row_and_column_order_follows_the_site_extensions(self):
        """Entries spell their configuration, so the order is read directly."""
        tensor_ci = _tensor_ci(_digits_box(), self.CONFIGURATION)

        matrix, rows, columns = tci.two_site_slice(tensor_ci, 2)

        self.assertEqual(rows, [(1, 0), (1, 1), (1, 2)])
        self.assertEqual(columns, [(0, 1), (1, 1)])
        npt.assert_allclose(
            matrix,
            [[1001.0, 1011.0], [1101.0, 1111.0], [1201.0, 1211.0]],
            rtol=RTOL,
        )

    def test_rows_and_columns_span_the_sites_their_bond_implies(self):
        tensor_ci = _tensor_ci(_dense_box(np.ones(DIMS)), self.CONFIGURATION)
        for bond in range(1, len(DIMS)):
            with self.subTest(bond=bond):
                _, rows, columns = tci.two_site_slice(tensor_ci, bond)
                self.assertTrue(all(len(row) == bond for row in rows))
                self.assertTrue(
                    all(len(column) == len(DIMS) - bond for column in columns)
                )

    def test_rows_extend_the_left_neighbour_and_columns_the_right(self):
        tensor_ci = _tensor_ci(_dense_box(np.ones(DIMS)), self.CONFIGURATION)

        _, rows, columns = tci.two_site_slice(tensor_ci, 2)

        self.assertTrue(all(row[:-1] in tensor_ci.pivots.left(1) for row in rows))
        self.assertTrue(
            all(column[1:] in tensor_ci.pivots.right(3) for column in columns)
        )

    def test_costs_one_evaluation_per_entry(self):
        """The evaluation count is what a run of TCI is charged for."""
        box = _dense_box(np.random.default_rng(1).standard_normal(DIMS))

        matrix, _, _ = tci.two_site_slice(_tensor_ci(box, self.CONFIGURATION), 2)

        self.assertEqual(box.n_evals, matrix.size)

    def test_rejects_a_bond_without_two_free_sites(self):
        tensor_ci = _tensor_ci(_dense_box(np.ones(DIMS)), self.CONFIGURATION)
        for bond in (0, len(DIMS)):
            with self.subTest(bond=bond):
                with self.assertRaises(ValueError):
                    tci.two_site_slice(tensor_ci, bond)


class TestUpdateBond(unittest.TestCase):
    """The update replaces one bond's pivots with those of its slice."""

    CONFIGURATION = [1, 0, 1, 1]

    def test_selects_the_largest_entry_when_one_pivot_is_kept(self):
        """A lone spike fixes which row and column a rank-one update takes."""
        tensor = np.zeros(DIMS)
        tensor[1, 2, 1, 0] = 5.0
        tensor_ci = _tensor_ci(_dense_box(tensor), [1, 2, 1, 0])

        result = tci.update_bond(tensor_ci, 2, max_rank=1)

        self.assertEqual(result.rank, 1)
        self.assertEqual(tensor_ci.pivots.left(2), ((1, 2),))
        self.assertEqual(tensor_ci.pivots.right(2), ((1, 0),))

    def test_new_lists_are_nested_with_their_neighbours(self):
        box = _dense_box(np.random.default_rng(2).standard_normal(DIMS))
        tensor_ci = _tensor_ci(box, self.CONFIGURATION)

        tci.update_bond(tensor_ci, 2)

        self.assertTrue(tensor_ci.pivots.is_left_nested_at(2))
        self.assertTrue(tensor_ci.pivots.is_right_nested_at(2))

    def test_leaves_every_other_bond_alone(self):
        box = _dense_box(np.random.default_rng(3).standard_normal(DIMS))
        tensor_ci = _tensor_ci(box, self.CONFIGURATION)
        others = [bond for bond in range(1, len(DIMS)) if bond != 2]
        before = {bond: _bond_pivots(tensor_ci, bond) for bond in others}

        tci.update_bond(tensor_ci, 2)

        for bond in others:
            with self.subTest(bond=bond):
                self.assertEqual(_bond_pivots(tensor_ci, bond), before[bond])

    def test_a_product_tensor_needs_one_pivot(self):
        tensor_ci = _tensor_ci(_dense_box(_product_tensor(4)), self.CONFIGURATION)

        result = tci.update_bond(tensor_ci, 2, tolerance=1e-10)

        self.assertEqual(result.rank, 1)
        self.assertLess(result.pivot_error, 1e-10)

    def test_max_rank_caps_the_pivots_kept(self):
        box = _dense_box(np.random.default_rng(5).standard_normal(DIMS))
        tensor_ci = _tensor_ci(box, self.CONFIGURATION)

        result = tci.update_bond(tensor_ci, 2, max_rank=1)

        self.assertEqual(result.rank, 1)
        self.assertEqual(tensor_ci.pivots.ranks[2], 1)

    def test_a_slice_below_tolerance_leaves_the_lists_untouched(self):
        """An empty bond would not describe a tensor train."""
        tensor_ci = _tensor_ci(_dense_box(np.zeros(DIMS)), self.CONFIGURATION)
        before = _bond_pivots(tensor_ci, 2)

        result = tci.update_bond(tensor_ci, 2, tolerance=1e-8)

        self.assertEqual(result.rank, 0)
        self.assertEqual(_bond_pivots(tensor_ci, 2), before)

    def test_sweeping_to_convergence_leaves_every_bond_nested(self):
        """sin(x1 + x2 + x3 + x4) has rank 2 at every internal bond."""
        tensor_ci = _tensor_ci(_sine_box(float), [7, 7, 7, 0], dims=(8, 8, 8, 8))

        _sweep(tensor_ci, 4, tolerance=1e-10)

        self.assertEqual(tensor_ci.pivots.ranks, [1, 2, 2, 2, 1])
        self.assertTrue(tensor_ci.pivots.is_fully_nested())

    def test_a_zero_tolerance_keeps_no_rounding_noise(self):
        """The noise level follows the precision of the sampled values."""
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=dtype):
                box = _sine_box(dtype)
                tensor_ci = _tensor_ci(box, [7, 7, 7, 0], dims=(8, 8, 8, 8))

                _sweep(tensor_ci, 4)

                self.assertEqual(tensor_ci.pivots.ranks, [1, 2, 2, 2, 1])
                self.assertLess(box.n_evals, 8**4)


def _rational(indices) -> float:
    """1 / (1 + 10 (x1 + ... + x5)) on six nodes in [0, 1] per site."""
    nodes = np.linspace(0.0, 1.0, 6)
    return 1.0 / (1.0 + 10.0 * sum(nodes[i] for i in indices))


RATIONAL_DIMS = (6, 6, 6, 6, 6)


def _dense(function, dims) -> np.ndarray:
    """Every element of the tensor behind ``function``."""
    return np.array([function(index) for index in np.ndindex(*dims)]).reshape(dims)


class TestTci(unittest.TestCase):
    """The driver sweeps until the ranks settle within the error threshold."""

    def test_finds_ranks_above_the_local_dimension(self):
        """The middle ranks, 10, exceed d = 6 and take more than one sweep."""
        error_threshold = 1e-12
        full = _dense(_rational, RATIONAL_DIMS)

        tensor_ci, info = tci.tci(
            _rational, RATIONAL_DIMS, [0] * 5, error_threshold, return_info=True
        )

        self.assertTrue(info.converged)
        self.assertLessEqual(info.pivot_error, error_threshold)
        self.assertEqual(tensor_ci.pivots.ranks, [1, 6, 10, 10, 6, 1])
        error = np.abs(tensor_ci.to_tensor_train().get_full_tensor() - full).max()
        self.assertLess(error, 10 * error_threshold * np.abs(full).max())

    def test_reproduces_an_exactly_low_rank_train(self):
        train = tensor_train.rand([3, 4, 3, 4, 3], ranks=4, seed=7)
        full = train.get_full_tensor()

        tensor_ci, info = tci.tci(
            black_box.tt_as_black_box(train),
            (3, 4, 3, 4, 3),
            [0] * 5,
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
        small = tci.tci(_rational, RATIONAL_DIMS, [0] * 5)
        large = tci.tci(lambda idx: 2.0**40 * _rational(idx), RATIONAL_DIMS, [0] * 5)

        for bond in range(1, 5):
            with self.subTest(bond=bond):
                self.assertEqual(large.pivots.left(bond), small.pivots.left(bond))
                self.assertEqual(large.pivots.right(bond), small.pivots.right(bond))

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
                    function, (8, 8, 8, 8), [0] * 4, error_threshold, return_info=True
                )
                self.assertTrue(info.converged)
                self.assertEqual(tensor_ci.pivots.ranks, [1, 2, 2, 2, 1])

    def test_a_binding_max_rank_is_reported_as_not_converged(self):
        """Rank 3 is below the 10 that the error threshold needs."""
        error_threshold = 1e-12
        with self.assertLogs("ttnumpy", level="WARNING"):
            tensor_ci, info = tci.tci(
                _rational,
                RATIONAL_DIMS,
                [0] * 5,
                error_threshold,
                max_rank=3,
                return_info=True,
            )

        self.assertLessEqual(max(tensor_ci.pivots.ranks), 3)
        self.assertFalse(info.converged)
        self.assertGreater(info.pivot_error, error_threshold)

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
                    [dim // 2] * n_sites,
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
                _rational,
                RATIONAL_DIMS,
                [0] * 5,
                1e-12,
                max_half_sweeps=1,
                return_info=True,
            )

        self.assertFalse(info.converged)
        self.assertEqual(info.num_half_sweeps_run, 1)

    def test_returns_the_tensor_ci_alone_unless_info_is_asked_for(self):
        result = tci.tci(_rational, RATIONAL_DIMS, [0] * 5)
        self.assertIsInstance(result, tci_form.TensorCI)

    def test_uses_a_given_black_box_as_it_is(self):
        box = black_box.BlackBox(_rational)

        tensor_ci = tci.tci(box, RATIONAL_DIMS, [0] * 5)

        self.assertIs(tensor_ci.box, box)

    def test_rejects_an_initial_pivot_where_the_tensor_vanishes(self):
        nodes = np.linspace(0.0, 1.0, 4)
        with self.assertRaises(ValueError):
            tci.tci(lambda idx: np.prod(nodes[list(idx)]), (4, 4, 4), [0, 0, 0])

    def test_rejects_invalid_arguments_before_evaluating(self):
        for dims, initial_pivot, options in [
            (RATIONAL_DIMS, [0] * 5, {"error_threshold": -1.0}),
            (RATIONAL_DIMS, [0] * 5, {"error_threshold": float("nan")}),
            (RATIONAL_DIMS, [0] * 5, {"max_rank": 0}),
            (RATIONAL_DIMS, [0] * 5, {"max_half_sweeps": 0}),
            (RATIONAL_DIMS, [0] * 4, {}),
            # the two-site slices around a site of dimension 1 cannot grow a rank
            ((6, 1, 6, 6, 6), [0] * 5, {}),
        ]:
            with self.subTest(dims=dims, initial_pivot=initial_pivot, options=options):
                box = black_box.BlackBox(_rational)
                with self.assertRaises(ValueError):
                    tci.tci(box, dims, initial_pivot, **options)
                self.assertEqual(box.n_evals, 0)


if __name__ == "__main__":
    unittest.main()
