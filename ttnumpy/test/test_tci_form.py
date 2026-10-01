# Copyright (c) 2026 QBoard LLC. Licensed under the AGPLv3. See LICENSE.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the TCI form: pivot bookkeeping and contraction into a tensor train.

The nesting cases come from Table 1 of SciPost Phys. 18, 104 (2025), together
with the two ways of breaking it that the paper names in the surrounding text.
Tensors are seeded, since the train is compared entry by entry against the
tensor it interpolates.
"""

import itertools
import unittest

import numpy as np
import numpy.testing as npt

from ttnumpy import tensor_train
from ttnumpy.interpolation import black_box, tci, tci_form

RTOL = 1e-10
ATOL = 1e-12

#: Table 1 of the paper: one (left, right) pair per internal bond, bonds 1 to 4.
TABLE_1 = [
    ([(1,)], [(1, 0, 0, 1)]),
    ([(1, 0), (1, 1)], [(0, 0, 1), (1, 0, 1)]),
    ([(1, 1, 0), (1, 0, 1)], [(0, 1), (1, 1)]),
    ([(1, 1, 0, 0)], [(1,)]),
]

#: Dimensions of the five sites of Table 1.
TABLE_1_DIMS = (2, 2, 2, 2, 2)


def _table_1_with(bond, row, column):
    """Table 1 with one extra row and column pivot at ``bond``."""
    bonds = [(list(left), list(right)) for left, right in TABLE_1]
    bonds[bond - 1][0].append(row)
    bonds[bond - 1][1].append(column)
    return tci_form.PivotLists(bonds)


def _dense_box(tensor: np.ndarray) -> black_box.BlackBox:
    """Sample a dense tensor by index."""
    return black_box.BlackBox(lambda indices: tensor[indices])


def _slice_configurations(pivots: tci_form.PivotLists, dims) -> list:
    """Every configuration of every ``T_s`` the pivots define."""
    return [
        left + (value,) + right
        for site, dim in enumerate(dims)
        for left, value, right in itertools.product(
            pivots.left(site), range(dim), pivots.right(site + 1)
        )
    ]


def _sweep(tensor_ci: tci_form.TensorCI, half_sweeps: int, tolerance: float) -> None:
    """Alternate forward and backward half-sweeps of two-site updates."""
    n_sites = len(tensor_ci.dims)
    for half_sweep in range(half_sweeps):
        forward = half_sweep % 2 == 0
        bonds = range(1, n_sites) if forward else range(n_sites - 1, 0, -1)
        for bond in bonds:
            tci.update_bond(tensor_ci, bond, tolerance)


class TestSplitConfiguration(unittest.TestCase):
    """A configuration is cut at every bond into halves that rejoin to it."""

    def test_halves_rejoin_at_every_bond(self):
        configuration = (1, 0, 2, 1)
        lefts, rights = tci_form.split_configuration(configuration)

        self.assertEqual(len(lefts), len(configuration) + 1)
        for bond, (head, tail) in enumerate(zip(lefts, rights)):
            with self.subTest(bond=bond):
                self.assertEqual(head + tail, configuration)
                self.assertEqual(len(head), bond)

    def test_outer_bonds_are_empty(self):
        lefts, rights = tci_form.split_configuration([3, 1])
        self.assertEqual(lefts[0], ())
        self.assertEqual(rights[-1], ())

    def test_numpy_integers_are_normalized(self):
        """Pivot indices arrive from NumPy, and tuples of them must compare equal."""
        lefts, _ = tci_form.split_configuration(np.array([1, 0, 2]))
        self.assertEqual(lefts[2], (1, 0))
        for index in lefts[2]:
            self.assertNotIsInstance(index, np.integer)


class TestSiteExtension(unittest.TestCase):
    """Extension order fixes the row and column order of a two-site slice."""

    def test_append_site_varies_the_new_site_fastest(self):
        rows = tci_form.append_site([(0,), (1,), (2,)], 2)
        self.assertEqual(rows, [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)])

    def test_prepend_site_varies_the_multi_index_fastest(self):
        columns = tci_form.prepend_site([(0,), (1,), (2,)], 2)
        self.assertEqual(columns, [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)])

    def test_extension_multiplies_the_count_by_the_dimension(self):
        indices = [(0, 1), (1, 1)]
        self.assertEqual(len(tci_form.append_site(indices, 4)), 8)
        self.assertEqual(len(tci_form.prepend_site(indices, 4)), 8)

    def test_table_1_bond_2_is_a_full_extension_of_bond_1(self):
        """Every pivot list of Table 1 is a subset of an extension."""
        lists = tci_form.PivotLists(TABLE_1)
        self.assertEqual(tci_form.append_site(lists.left(1), 2), list(lists.left(2)))

    def test_extending_the_empty_multi_index_enumerates_one_site(self):
        self.assertEqual(tci_form.append_site([()], 3), [(0,), (1,), (2,)])
        self.assertEqual(tci_form.prepend_site([()], 3), [(0,), (1,), (2,)])


class TestNesting(unittest.TestCase):
    """Nesting is prefix closure on the left and suffix closure on the right."""

    def test_table_1_is_fully_nested(self):
        self.assertTrue(tci_form.PivotLists(TABLE_1).is_fully_nested())

    def test_a_row_pivot_whose_prefix_is_absent_breaks_left_nesting(self):
        """The paper names adding (0, 0) to I_2 as a way to break Table 1.

        The added column pivot keeps the bond square and its own suffix is
        present at the next bond, so right nesting still holds.
        """
        lists = _table_1_with(2, row=(0, 0), column=(1, 1, 1))

        self.assertFalse(lists.is_left_nested_at(2))
        self.assertTrue(lists.is_right_nested_at(2))
        self.assertFalse(lists.is_fully_nested())

    def test_a_column_pivot_whose_suffix_is_absent_breaks_right_nesting(self):
        """A column pivot whose suffix is absent from the next bond.

        The paper's example adds (1, 1, 0) to J_3. At that bond both
        extensions of I_1 are already present, leaving no square counterpart,
        so the same breakage is placed at bond 3.
        """
        lists = _table_1_with(3, row=(1, 0, 0), column=(1, 0))

        self.assertFalse(lists.is_right_nested_at(3))
        self.assertTrue(lists.is_left_nested_at(3))
        self.assertFalse(lists.is_fully_nested())

    def test_nesting_is_defined_only_at_internal_bonds(self):
        """The chain of eq. (37) links bonds 1 to n_sites-1 on both sides."""
        lists = tci_form.PivotLists(TABLE_1)
        for bond in (0, lists.n_sites):
            with self.subTest(bond=bond):
                with self.assertRaises(ValueError):
                    lists.is_left_nested_at(bond)
                with self.assertRaises(ValueError):
                    lists.is_right_nested_at(bond)


class TestPivotLists(unittest.TestCase):
    """The container fixes the bond convention and guards its invariants."""

    def test_from_configuration_is_rank_one_and_fully_nested(self):
        lists = tci_form.PivotLists.from_configuration([1, 0, 1, 1])

        self.assertEqual(lists.n_sites, 4)
        self.assertEqual(lists.ranks, [1, 1, 1, 1, 1])
        self.assertTrue(lists.is_fully_nested())

    def test_halves_of_an_internal_bond_tile_the_sites(self):
        lists = tci_form.PivotLists.from_configuration([1, 0, 1, 1])
        for bond in range(1, lists.n_sites):
            with self.subTest(bond=bond):
                self.assertEqual(
                    lists.left(bond)[0] + lists.right(bond)[0], (1, 0, 1, 1)
                )

    def test_ranks_report_the_bond_dimensions(self):
        self.assertEqual(tci_form.PivotLists(TABLE_1).ranks, [1, 1, 2, 2, 1, 1])

    def test_outer_bonds_are_the_empty_dummy(self):
        lists = tci_form.PivotLists(TABLE_1)
        self.assertEqual(lists.left(0), ((),))
        self.assertEqual(lists.right(lists.n_sites), ((),))

    def test_the_undefined_halves_of_the_outer_bonds_raise(self):
        """right(0) and left(n_sites) name no pivots, so asking for them fails."""
        lists = tci_form.PivotLists(TABLE_1)
        with self.assertRaises(ValueError):
            lists.right(0)
        with self.assertRaises(ValueError):
            lists.left(lists.n_sites)

    def test_negative_bonds_raise_rather_than_wrap(self):
        lists = tci_form.PivotLists(TABLE_1)
        with self.assertRaises(ValueError):
            lists.left(-1)
        with self.assertRaises(ValueError):
            lists.right(-1)

    def test_returned_pivots_cannot_modify_the_stored_ones(self):
        lists = tci_form.PivotLists(TABLE_1)
        with self.assertRaises(AttributeError):
            lists.left(2).append((0, 0))

    def test_replace_installs_new_pivots(self):
        lists = tci_form.PivotLists(TABLE_1)
        lists.replace(2, [(1, 1)], [(0, 0, 1)])

        self.assertEqual(lists.left(2), ((1, 1),))
        self.assertEqual(lists.right(2), ((0, 0, 1),))
        self.assertEqual(lists.ranks, [1, 1, 1, 2, 1, 1])

    def test_replace_normalizes_numpy_integers(self):
        lists = tci_form.PivotLists(TABLE_1)
        lists.replace(2, [np.array([1, 1])], [np.array([0, 0, 1])])
        for index in lists.left(2)[0] + lists.right(2)[0]:
            self.assertNotIsInstance(index, np.integer)

    def test_a_rejected_replace_leaves_the_pivots_unchanged(self):
        lists = tci_form.PivotLists(TABLE_1)
        before = (lists.left(2), lists.right(2))

        for left, right in [
            ([(1, 1)], []),  # empty column side
            ([(1, 1), (1, 0)], [(0, 0, 1)]),  # not square
            ([(1,)], [(0, 0, 1)]),  # row pivot spans the wrong sites
            ([(1, 0), (1, 0)], [(0, 0, 1), (1, 0, 1)]),  # repeated row pivot
            ([(1, 0), (1, 1)], [(0, 0, 1), (0, 0, 1)]),  # repeated column pivot
            ([(1, -1)], [(0, 0, 1)]),  # negative index
        ]:
            with self.subTest(left=left, right=right):
                with self.assertRaises(ValueError):
                    lists.replace(2, left, right)
                self.assertEqual((lists.left(2), lists.right(2)), before)

    def test_non_integer_indices_are_refused(self):
        """A float is refused rather than truncated to the integer below it."""
        lists = tci_form.PivotLists(TABLE_1)
        with self.assertRaises(TypeError):
            lists.replace(2, [(1, 1.0)], [(0, 0, 1)])
        with self.assertRaises(TypeError):
            tci_form.PivotLists.from_configuration([0, 2.9999, 1])

    def test_replace_rejects_an_outer_bond(self):
        lists = tci_form.PivotLists(TABLE_1)
        for bond in (0, lists.n_sites):
            with self.subTest(bond=bond):
                with self.assertRaises(ValueError):
                    lists.replace(bond, [()], [()])

    def test_construction_applies_the_same_checks(self):
        bonds = [(list(left), list(right)) for left, right in TABLE_1]
        bonds[1] = ([(1, 0), (1, 1), (0, 0)], bonds[1][1])
        with self.assertRaises(ValueError):
            tci_form.PivotLists(bonds)

    def test_a_one_site_train_has_no_internal_bonds(self):
        lists = tci_form.PivotLists.from_configuration([3])
        self.assertEqual(lists.n_sites, 1)
        self.assertEqual(lists.ranks, [1, 1])
        self.assertTrue(lists.is_fully_nested())

    def test_rejects_an_empty_configuration(self):
        with self.assertRaises(ValueError):
            tci_form.PivotLists.from_configuration([])


class TestTensorCI(unittest.TestCase):
    """The pivots must index a tensor train over the given dims."""

    def test_rejects_dims_of_the_wrong_length(self):
        box = _dense_box(np.ones(TABLE_1_DIMS))
        with self.assertRaises(ValueError):
            tci_form.TensorCI(box, TABLE_1_DIMS[:-1], tci_form.PivotLists(TABLE_1))

    def test_rejects_a_single_site(self):
        box = _dense_box(np.ones(3))
        with self.assertRaises(ValueError):
            tci_form.TensorCI(box, (3,), tci_form.PivotLists.from_configuration([0]))

    def test_rejects_dimensions_that_are_not_integers(self):
        pivots = tci_form.PivotLists.from_configuration([0, 0, 0])
        with self.assertRaises(TypeError):
            tci_form.TensorCI(_dense_box(np.ones((3, 3, 3))), (3.0, 3.0, 3.0), pivots)

    def test_rejects_pivots_outside_the_dims(self):
        box = _dense_box(np.ones((3, 3, 3)))
        for configuration in ([0, 3, 1], [0, 0, 5]):
            with self.subTest(configuration=configuration):
                pivots = tci_form.PivotLists.from_configuration(configuration)
                with self.assertRaises(ValueError):
                    tci_form.TensorCI(box, (3, 3, 3), pivots)

    def test_the_train_follows_updates_to_the_pivots(self):
        dims = (2, 3, 2, 2)
        tensor = np.random.default_rng(0).standard_normal(dims)
        pivots = tci_form.PivotLists.from_configuration([1, 0, 1, 1])
        tensor_ci = tci_form.TensorCI(_dense_box(tensor), dims, pivots)

        tci.update_bond(tensor_ci, 2)

        self.assertEqual(tensor_ci.to_tensor_train().bonds, [1, 1, 2, 1, 1])

    def test_repr_reports_the_ranks(self):
        tensor_ci = tci_form.TensorCI(
            _dense_box(np.ones(TABLE_1_DIMS)),
            TABLE_1_DIMS,
            tci_form.PivotLists(TABLE_1),
        )
        self.assertIn("ranks=[1, 1, 2, 2, 1, 1]", repr(tensor_ci))


class TestToTensorTrain(unittest.TestCase):
    """The train is the TCI form contracted over its pivots."""

    def test_cores_follow_the_ranks_and_dims(self):
        tensor = np.random.default_rng(1).standard_normal(TABLE_1_DIMS)
        tensor_ci = tci_form.TensorCI(
            _dense_box(tensor), TABLE_1_DIMS, tci_form.PivotLists(TABLE_1)
        )

        train = tensor_ci.to_tensor_train()

        self.assertEqual(train.bonds, [1, 1, 2, 2, 1, 1])
        self.assertEqual(train.row_modes, list(TABLE_1_DIMS))

    def test_exact_on_every_slice_when_fully_nested(self):
        """Eq. (38) of the paper, on the fully nested pivots of its Table 1."""
        rng = np.random.default_rng(2)
        real = rng.standard_normal(TABLE_1_DIMS)
        tensors = {
            "real": real,
            "complex": real + 1j * rng.standard_normal(TABLE_1_DIMS),
        }
        pivots = tci_form.PivotLists(TABLE_1)
        configurations = _slice_configurations(pivots, TABLE_1_DIMS)

        for name, tensor in tensors.items():
            with self.subTest(dtype=name):
                tensor_ci = tci_form.TensorCI(_dense_box(tensor), TABLE_1_DIMS, pivots)
                full = tensor_ci.to_tensor_train().get_full_tensor()
                npt.assert_allclose(
                    [full[index] for index in configurations],
                    [tensor[index] for index in configurations],
                    rtol=RTOL,
                    atol=ATOL,
                )

    def test_reproduces_a_low_rank_tensor_after_sweeping(self):
        """Middle rank 5 exceeds the dimension 3, so it takes several sweeps."""
        dims = (3, 3, 3, 3)
        train = tensor_train.rand(list(dims), ranks=5, seed=3)
        full = train.get_full_tensor()
        tensor_ci = tci_form.TensorCI(
            black_box.tt_as_black_box(train),
            dims,
            tci_form.PivotLists.from_configuration([0, 0, 0, 0]),
        )

        _sweep(tensor_ci, 4, tolerance=1e-12 * np.abs(full).max())

        self.assertEqual(tensor_ci.pivots.ranks, [1, 3, 5, 3, 1])
        self.assertTrue(tensor_ci.pivots.is_fully_nested())
        npt.assert_allclose(
            tensor_ci.to_tensor_train().get_full_tensor(), full, rtol=RTOL, atol=ATOL
        )

    def test_a_half_sweep_leaves_nothing_new_to_evaluate(self):
        """Every T_s and P_b lies in a slice that the last updates sampled."""
        dims = (4, 4, 4, 4)
        train = tensor_train.rand(list(dims), ranks=3, seed=4)
        for half_sweeps in (1, 2):
            with self.subTest(half_sweeps=half_sweeps):
                tensor_ci = tci_form.TensorCI(
                    black_box.tt_as_black_box(train),
                    dims,
                    tci_form.PivotLists.from_configuration([0, 0, 0, 0]),
                )
                _sweep(tensor_ci, half_sweeps, tolerance=1e-12)
                evaluated = tensor_ci.box.n_evals

                tensor_ci.to_tensor_train()

                self.assertEqual(tensor_ci.box.n_evals, evaluated)

    def test_integer_values_give_floating_point_cores(self):
        box = black_box.BlackBox(lambda indices: sum(indices) + 1)
        pivots = tci_form.PivotLists.from_configuration([1, 1, 1])
        tensor_ci = tci_form.TensorCI(box, (3, 3, 3), pivots)
        for core in tensor_ci.to_tensor_train().cores:
            self.assertEqual(core.dtype, np.float64)

    def test_a_singular_pivot_matrix_raises(self):
        tensor_ci = tci_form.TensorCI(
            _dense_box(np.zeros(TABLE_1_DIMS)),
            TABLE_1_DIMS,
            tci_form.PivotLists(TABLE_1),
        )
        with self.assertRaises(np.linalg.LinAlgError):
            tensor_ci.to_tensor_train()


if __name__ == "__main__":
    unittest.main()
