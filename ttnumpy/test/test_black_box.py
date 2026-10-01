# Copyright (c) 2026 QBoard LLC. Licensed under the AGPLv3. See LICENSE.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the sampling wrapper around a black-box function.

The trains here are seeded, since the wrapper is compared element by element
against the train it wraps.
"""

import unittest
from unittest.mock import Mock

import numpy as np
import numpy.testing as npt

from ttnumpy import tensor_train
from ttnumpy.interpolation import black_box

RTOL = 1e-12


def _spy() -> Mock:
    """A callable that records the configurations it receives."""
    return Mock(return_value=1.0)


class TestCaching(unittest.TestCase):
    """A configuration is evaluated once however often it is requested."""

    def test_counters_start_at_zero(self):
        box = black_box.BlackBox(_spy())
        self.assertEqual((box.n_evals, box.n_requests, box.n_hits), (0, 0, 0))

    def test_repeated_configuration_is_evaluated_once(self):
        spy = _spy()
        box = black_box.BlackBox(spy)

        for _ in range(5):
            box([1, 0, 2])

        self.assertEqual(spy.call_count, 1)
        self.assertEqual(box.n_evals, 1)
        self.assertEqual(box.n_requests, 5)
        self.assertEqual(box.n_hits, 4)

    def test_distinct_configurations_are_evaluated_separately(self):
        spy = _spy()
        box = black_box.BlackBox(spy)

        box([0, 0])
        box([0, 1])
        box([0, 0])

        self.assertEqual(box.n_evals, 2)
        self.assertEqual(box.n_requests, 3)
        self.assertEqual(box.n_hits, 1)

    def test_cached_value_matches_the_first_evaluation(self):
        values = iter([7.0, 99.0])
        box = black_box.BlackBox(lambda indices: next(values))
        first = box([0])
        self.assertEqual(box([0]), first)


class TestKeyNormalization(unittest.TestCase):
    """Configurations are keyed on their integer values, not their container."""

    def test_sequence_types_share_one_cache_entry(self):
        spy = _spy()
        box = black_box.BlackBox(spy)

        box([1, 2, 3])
        box((1, 2, 3))
        box(np.array([1, 2, 3]))

        self.assertEqual(box.n_evals, 1)
        self.assertEqual(box.n_requests, 3)

    def test_the_wrapped_function_receives_python_ints(self):
        """Callees see Python ints, whatever integer type the caller used."""
        spy = _spy()
        box = black_box.BlackBox(spy)

        box(np.array([1, 2, 3]))

        received = spy.call_args.args[0]
        self.assertIsInstance(received, tuple)
        for index in received:
            self.assertIsInstance(index, int)
            self.assertNotIsInstance(index, np.integer)

    def test_non_integer_indices_are_refused(self):
        """A float is refused rather than truncated to the integer below it."""
        spy = _spy()
        box = black_box.BlackBox(spy)

        with self.assertRaises(TypeError):
            box([0, 2.9999])
        spy.assert_not_called()

    def test_different_configurations_of_equal_length_do_not_collide(self):
        spy = _spy()
        box = black_box.BlackBox(spy)
        box([1, 2])
        box([2, 1])
        self.assertEqual(box.n_evals, 2)


class TestSample(unittest.TestCase):
    """A block is sampled at every row multi-index joined to every column."""

    def test_entries_join_a_row_to_a_column(self):
        """Entries spell their configuration, so the order is read directly."""
        box = black_box.BlackBox(
            lambda indices: float("".join(str(index) for index in indices))
        )

        matrix = box.sample([(1,), (2,)], [(0, 3), (4, 5), (6, 7)])

        npt.assert_array_equal(matrix, [[103.0, 145.0, 167.0], [203.0, 245.0, 267.0]])

    def test_every_entry_is_one_request(self):
        box = black_box.BlackBox(_spy())

        box.sample([(0,), (1,)], [(0,), (1,), (2,)])
        box.sample([(0,), (1,)], [(0,), (1,), (2,)])

        self.assertEqual(box.n_requests, 12)
        self.assertEqual(box.n_evals, 6)


class TestTensorTrainAdapter(unittest.TestCase):
    """A tensor train is a valid black box, which makes it a test oracle."""

    DIMS = (2, 3, 4)

    def test_reproduces_every_element_of_the_train(self):
        train = tensor_train.rand(self.DIMS, ranks=3, seed=0)
        box = black_box.tt_as_black_box(train)
        full = train.get_full_tensor()

        for index in np.ndindex(*self.DIMS):
            npt.assert_allclose(box(index), full[index], rtol=RTOL)

        self.assertEqual(box.n_evals, int(np.prod(self.DIMS)))

    def test_accepts_numpy_integer_indices(self):
        """Pivot lists are numpy arrays, so their entries reach the sampler."""
        train = tensor_train.rand(self.DIMS, ranks=2, seed=1)
        box = black_box.tt_as_black_box(train)

        npt.assert_allclose(box(np.array([1, 2, 3])), box([1, 2, 3]), rtol=RTOL)

    def test_rejects_a_tensor_train_matrix(self):
        matrix_train = tensor_train.eye([2, 2])
        with self.assertRaises(tensor_train.WrongTTType):
            black_box.tt_as_black_box(matrix_train)


class TestValues(unittest.TestCase):
    """Values pass through unchanged."""

    def test_complex_values_round_trip(self):
        box = black_box.BlackBox(lambda indices: 1.0 + 2.0j)
        self.assertEqual(box([0]), 1.0 + 2.0j)

    def test_non_finite_values_are_refused_and_not_cached(self):
        for value in [np.nan, np.inf, complex(0.0, np.nan)]:
            with self.subTest(value=value):
                box = black_box.BlackBox(Mock(return_value=value))
                with self.assertRaises(ValueError):
                    box([0])
                self.assertEqual(box.n_evals, 0)

    def test_max_modulus_covers_every_evaluated_value(self):
        values = iter([3.0, -5.0, 2.0j])
        box = black_box.BlackBox(lambda indices: next(values))
        self.assertEqual(box.max_modulus, 0.0)

        for index in range(3):
            box([index])

        self.assertEqual(box.max_modulus, 5.0)

    def test_repr_reports_the_counters(self):
        box = black_box.BlackBox(_spy())
        box([0])
        self.assertIn("n_evals=1", repr(box))


if __name__ == "__main__":
    unittest.main()
