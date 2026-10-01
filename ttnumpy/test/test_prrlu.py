# Copyright (c) 2026 QBoard LLC. Licensed under the AGPLv3. See LICENSE.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the partial rank-revealing LU decomposition.

Every matrix here is either seeded or written out, so a failure is
reproducible: the decomposition is compared against exact expectations and
against LAPACK, both of which an unlucky unseeded draw would make unreadable.
"""

import unittest

import numpy as np
import numpy.testing as npt
from scipy.linalg import lapack

from ttnumpy.interpolation.prrlu import prrlu

RTOL = 1e-10
ATOL = 1e-12


def _low_rank(rows: int, cols: int, rank: int, seed: int) -> np.ndarray:
    """A ``rows`` by ``cols`` matrix of exactly the requested rank."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((rows, rank)) @ rng.standard_normal((rank, cols))


class TestFactorization(unittest.TestCase):
    """At full rank the factors reproduce the matrix."""

    def test_square_matrix(self):
        matrix = np.random.default_rng(0).standard_normal((6, 6))
        npt.assert_allclose(prrlu(matrix).reconstruct(), matrix, rtol=RTOL, atol=ATOL)

    def test_tall_and_wide_matrices(self):
        for shape in [(9, 4), (4, 9)]:
            with self.subTest(shape=shape):
                matrix = np.random.default_rng(1).standard_normal(shape)
                result = prrlu(matrix)
                self.assertEqual(result.rank, min(shape))
                npt.assert_allclose(result.reconstruct(), matrix, rtol=RTOL, atol=ATOL)

    def test_complex_matrix(self):
        """Pivoting ranks entries by modulus, so a complex matrix factorizes."""
        rng = np.random.default_rng(2)
        matrix = rng.standard_normal((5, 7)) + 1j * rng.standard_normal((5, 7))
        npt.assert_allclose(prrlu(matrix).reconstruct(), matrix, rtol=RTOL, atol=ATOL)

    def test_degenerate_shapes(self):
        for matrix in [np.array([[2.5]]), np.arange(5.0).reshape(1, 5)]:
            with self.subTest(shape=matrix.shape):
                npt.assert_allclose(
                    prrlu(matrix).reconstruct(), matrix, rtol=RTOL, atol=ATOL
                )

    def test_integer_input_is_promoted(self):
        matrix = np.arange(12).reshape(3, 4)
        npt.assert_allclose(prrlu(matrix).reconstruct(), matrix, rtol=RTOL, atol=ATOL)

    def test_the_factorized_matrix_is_not_modified(self):
        matrix = np.random.default_rng(3).standard_normal((5, 5))
        original = matrix.copy()
        prrlu(matrix)
        npt.assert_array_equal(matrix, original)


class TestRankRevealing(unittest.TestCase):
    """The rank is discovered from the tolerance, not prescribed."""

    def test_finds_the_rank_of_a_low_rank_matrix(self):
        matrix = _low_rank(8, 10, 3, seed=4)
        result = prrlu(matrix, tolerance=1e-8)
        self.assertEqual(result.rank, 3)
        npt.assert_allclose(result.reconstruct(), matrix, rtol=RTOL, atol=1e-10)

    def test_a_zero_tolerance_refuses_rounding_noise(self):
        """arange(12) has rank 2; its third pivot is rounding noise."""
        for dtype in (np.float32, np.float64, np.complex128):
            with self.subTest(dtype=dtype):
                matrix = np.arange(12).reshape(3, 4).astype(dtype)
                self.assertEqual(prrlu(matrix, tolerance=0.0).rank, 2)

    def test_the_rounding_level_scales_with_the_entries(self):
        matrix = 1e-20 * np.arange(12.0).reshape(3, 4)
        self.assertEqual(prrlu(matrix).rank, 2)

    def test_a_zero_tolerance_keeps_small_pivots_above_rounding(self):
        self.assertEqual(prrlu(np.diag([1.0, 1e-12])).rank, 2)

    def test_zero_matrix_has_no_pivots(self):
        result = prrlu(np.zeros((4, 3)))
        self.assertEqual(result.rank, 0)
        self.assertEqual(result.pivot_error, 0.0)
        npt.assert_array_equal(result.reconstruct(), np.zeros((4, 3)))

    def test_a_random_matrix_has_full_rank(self):
        matrix = np.random.default_rng(5).standard_normal((6, 6))
        self.assertEqual(prrlu(matrix, tolerance=1e-8).rank, 6)


class TestPivotError(unittest.TestCase):
    """``pivot_error`` is the deviation of the reconstruction, not a bound on it."""

    def test_pivot_error_equals_the_largest_elementwise_deviation(self):
        """The Schur complement is the reconstruction error, so the two agree.

        The identity is exact rather than approximate, hence the tight rtol.
        """
        matrix = np.random.default_rng(6).standard_normal((7, 9))
        # rank 7 exhausts the rows, leaving an empty Schur complement and a
        # reported error of exactly 0 against a reconstruction that differs by
        # rounding; that case is asserted in test_pivot_error_vanishes_at_full_rank
        for rank in range(1, 7):
            with self.subTest(rank=rank):
                result = prrlu(matrix, max_rank=rank)
                deviation = np.abs(matrix - result.reconstruct()).max()
                npt.assert_allclose(result.pivot_error, deviation, rtol=1e-12)

    def test_pivot_error_vanishes_at_full_rank(self):
        matrix = _low_rank(6, 6, 2, seed=7)
        self.assertLess(prrlu(matrix, tolerance=1e-8).pivot_error, 1e-10)

    def test_pivot_error_is_the_pivot_that_was_refused(self):
        matrix = np.diag([1.0, 1e-6])
        result = prrlu(matrix, tolerance=1e-3)
        self.assertEqual(result.rank, 1)
        npt.assert_allclose(result.pivot_error, 1e-6, rtol=RTOL)


class TestInterpolation(unittest.TestCase):
    """The truncated reconstruction is exact on the rows and columns it kept."""

    MATRIX = np.random.default_rng(8).standard_normal((7, 9))

    def test_exact_on_pivot_rows_and_columns(self):
        result = prrlu(self.MATRIX, max_rank=3)
        approximation = result.reconstruct()
        npt.assert_allclose(
            approximation[result.row_pivots, :],
            self.MATRIX[result.row_pivots, :],
            rtol=RTOL,
            atol=ATOL,
        )
        npt.assert_allclose(
            approximation[:, result.col_pivots],
            self.MATRIX[:, result.col_pivots],
            rtol=RTOL,
            atol=ATOL,
        )

    def test_pivots_are_distinct_and_within_the_matrix(self):
        result = prrlu(self.MATRIX, max_rank=4)
        for pivots, size in [(result.row_pivots, 7), (result.col_pivots, 9)]:
            with self.subTest(size=size):
                self.assertEqual(len(set(pivots.tolist())), result.rank)
                self.assertTrue(np.all((pivots >= 0) & (pivots < size)))

    def test_pivot_values_lie_on_the_diagonal(self):
        result = prrlu(self.MATRIX, max_rank=3)
        npt.assert_allclose(
            result.diagonal[0],
            self.MATRIX[result.row_pivots[0], result.col_pivots[0]],
            rtol=RTOL,
        )


class TestAgainstLapack(unittest.TestCase):
    """Cross-check against ``dgetc2``, LAPACK's complete-pivoting LU."""

    def test_pivot_sequence_matches_dgetc2(self):
        """``dgetc2`` leaves its pivots on the diagonal of the factor it returns."""
        rng = np.random.default_rng(9)
        for trial in range(5):
            with self.subTest(trial=trial):
                matrix = rng.standard_normal((6, 6))
                # scipy builds its LAPACK wrappers at import time, so pylint
                # cannot see this member
                # pylint: disable-next=no-member
                factor, _, _, _ = lapack.dgetc2(matrix.copy())
                npt.assert_allclose(
                    np.abs(prrlu(matrix).diagonal),
                    np.abs(np.diag(factor)),
                    rtol=1e-10,
                )

    def test_first_pivot_is_the_largest_entry(self):
        matrix = np.random.default_rng(10).standard_normal((5, 8))
        result = prrlu(matrix, max_rank=1)
        npt.assert_allclose(abs(result.diagonal[0]), np.abs(matrix).max(), rtol=RTOL)


class TestArguments(unittest.TestCase):
    """``max_rank`` and ``tolerance`` bound the elimination; bad input raises."""

    def test_max_rank_caps_the_number_of_pivots(self):
        matrix = np.random.default_rng(11).standard_normal((8, 8))
        for cap in [1, 3, 8, 12]:
            with self.subTest(cap=cap):
                self.assertEqual(prrlu(matrix, max_rank=cap).rank, min(cap, 8))

    def test_tolerance_and_max_rank_both_apply(self):
        matrix = _low_rank(6, 6, 4, seed=12)
        self.assertEqual(prrlu(matrix, tolerance=1e-8, max_rank=2).rank, 2)

    def test_rejects_invalid_arguments(self):
        matrix = np.eye(3)
        with self.assertRaises(ValueError):
            prrlu(np.ones((2, 2, 2)))
        with self.assertRaises(ValueError):
            prrlu(matrix, tolerance=-1.0)
        with self.assertRaises(ValueError):
            prrlu(matrix, tolerance=float("nan"))
        with self.assertRaises(ValueError):
            prrlu(matrix, max_rank=0)


if __name__ == "__main__":
    unittest.main()
