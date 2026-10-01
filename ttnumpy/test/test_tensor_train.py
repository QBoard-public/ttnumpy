# Copyright (c) 2026 QBoard LLC. Licensed under the AGPLv3. See LICENSE.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Test module for tensor train type"""

import unittest

import numpy as np
import numpy.testing as npt

from ttnumpy import tensor_train

TEST_DELTA = 1e-10
DECIMAL = 10

# Seeded so a failing core layout is always reproducible.
_RNG_CORES = np.random.default_rng(40)


class TestTTMethods(unittest.TestCase):
    """Tests for base tensor train methods."""

    def test_tt_initialization(self):
        """Test for initialization method."""

        # GHZ state (check https://arxiv.org/abs/1603.03039 chapter 3.2)
        ising_bond = 2
        ising_phys = 2

        first_cite = np.array([1, 0, 0, 1]).reshape((1, ising_phys, ising_bond))
        last_cite = np.array([1, 0, 0, 1]).reshape((ising_bond, ising_phys, 1))
        cite_i = np.array([[1, 0], [0, 0], [0, 0], [0, 1]]).reshape(
            (ising_bond, ising_phys, ising_bond)
        )
        ghz = [first_cite, cite_i, last_cite]
        tt = tensor_train.TensorTrain(ghz)

        self.assertEqual(tt.bonds, [1] + [2] * 2 + [1])
        self.assertEqual(tt.physical_dims(), [2] * 3)

        npt.assert_array_almost_equal(
            tt[0], first_cite.reshape(1, ising_phys, 1, ising_bond), decimal=DECIMAL
        )
        npt.assert_array_almost_equal(
            tt[-1], last_cite.reshape(ising_bond, ising_phys, 1, 1), decimal=DECIMAL
        )

        # for same indexes should be 1, for different 0
        self.assertEqual(tt.get_element((0, 0, 0)), 1)
        self.assertEqual(tt.get_element((1, 1, 1)), 1)
        self.assertEqual(tt.get_element((0, 1, 0)), 0)

    def test_ttm_initialization(self):
        """Test for initialization method."""

        # Hamiltonian of Ising spin chain (check https://en.wikipedia.org/wiki/Ising_model)
        # H = S_1 \otimes S_2 + S_2 \otimes S_3, S = |0><0| - |1><1|
        ising_bond = 3
        ising_phys = 2

        S_z = np.array([[1, 0], [0, -1]])
        zero = np.zeros((2, 2))
        eye = np.eye(2)

        core_i = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 0, 0, -1, 0, 0],
                [0, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, -1, 1],
            ]
        ).reshape((ising_bond, ising_phys, ising_phys, ising_bond))
        first_core = np.array(
            [[[0, 1, 1], [0, 0, 0]], [[0, 0, 0], [0, -1, 1]]]
        ).reshape((1, ising_phys, ising_phys, ising_bond))
        last_core = np.array([eye, S_z, zero]).reshape(
            (ising_bond, ising_phys, ising_phys, 1)
        )
        ising_H = [first_core, core_i, last_core]
        tt = tensor_train.TensorTrain(ising_H)

        self.assertEqual(tt.bonds, [1] + [ising_bond] * 2 + [1])
        self.assertEqual(tt.physical_dims(), [(ising_phys, ising_phys)] * 3)

        npt.assert_array_almost_equal(tt[0], first_core, decimal=DECIMAL)
        npt.assert_array_almost_equal(tt[0], first_core, decimal=DECIMAL)

        # for all same must be equal to 2
        self.assertEqual(tt.get_element(((0, 0), (0, 0), (0, 0))), 2.0)
        self.assertEqual(tt.get_element(((1, 1), (1, 1), (1, 1))), 2.0)

        # if one non middle pair is different, must be equal to 0
        # if middle pair, must be equal to -2
        self.assertEqual(tt.get_element(((0, 0), (1, 1), (0, 0))), -2.0)
        self.assertEqual(tt.get_element(((1, 1), (1, 1), (0, 0))), 0.0)

        # if any pair contain different sites, must be equal to 0
        self.assertEqual(tt.get_element(((1, 0), (1, 1), (0, 0))), 0.0)
        self.assertEqual(tt.get_element(((1, 0), (1, 0), (1, 0))), 0.0)

    def test_conjugate_transpose(self):
        """Test for conjugate transpose method."""
        cores = (
            [_RNG_CORES.random((1, 2, 3, 2))]
            + [_RNG_CORES.random((2, 2, 3, 2))]
            + [_RNG_CORES.random((2, 2, 3, 1))]
        )
        transpose_cores = [cores[i].transpose(0, 2, 1, 3) for i in range(3)]
        ttm = tensor_train.TensorTrain(cores)
        npt.assert_array_almost_equal(
            ttm.transpose([0])[0], transpose_cores[0], decimal=DECIMAL
        )
        npt.assert_array_almost_equal(ttm.transpose([0])[1], cores[1], decimal=DECIMAL)

        im_cores = [cores[i] * (complex(1, i)) for i in range(3)]
        transpose_im_cores = [im_cores[i].transpose(0, 2, 1, 3) for i in range(3)]
        im_ttm = tensor_train.TensorTrain(im_cores)
        for i in range(3):
            npt.assert_array_almost_equal(
                ttm.transpose()[i], transpose_cores[i], decimal=DECIMAL
            )
            npt.assert_array_almost_equal(
                im_ttm.transpose(conjugate=True).conj()[i],
                transpose_im_cores[i],
                decimal=DECIMAL,
            )

    def test_orthonormalize(self):
        """Test for orthonormalization method."""
        tt = tensor_train.rand(((2, 3), (2, 3), (2, 3)), ranks=2)
        # left
        tt_left_ort = tt.orthonormalize("left")
        self.assertTrue(tt_left_ort.orth_center == len(tt) - 1)

        for i in range(len(tt_left_ort) - 1):
            matrix = tt_left_ort[i].reshape(
                tt_left_ort[i].shape[0]
                * tt_left_ort[i].shape[1]
                * tt_left_ort[i].shape[2],
                tt_left_ort[i].shape[-1],
            )
            npt.assert_array_almost_equal(
                matrix.conj().T @ matrix, np.eye(matrix.shape[-1]), decimal=DECIMAL
            )

        # right
        tt_right_ort = tt.orthonormalize("right")
        self.assertTrue(tt_right_ort.orth_center == 0)

        for i in range(len(tt_right_ort) - 1, 0, -1):
            tens = tt_right_ort[i].reshape(
                tt_right_ort[i].shape[0],
                tt_right_ort[i].shape[1]
                * tt_right_ort[i].shape[2]
                * tt_right_ort[i].shape[-1],
            )
            npt.assert_array_almost_equal(
                tens @ tens.conj().T, np.eye(tens.shape[0]), decimal=DECIMAL
            )

        # mix
        tt_mix_ort = tt.orthonormalize("mix")
        self.assertTrue(tt_mix_ort.orth_center == len(tt) // 2)

        for i in range(len(tt_mix_ort) // 2):
            # left
            tens = tt_mix_ort[i].reshape(
                tt_mix_ort[i].shape[0]
                * tt_mix_ort[i].shape[1]
                * tt_mix_ort[i].shape[2],
                tt_mix_ort[i].shape[-1],
            )
            npt.assert_array_almost_equal(
                tens.conj().T @ tens, np.eye(tens.shape[-1]), decimal=DECIMAL
            )

            # right
            tens = tt_mix_ort[-i - 1].reshape(
                tt_mix_ort[-i - 1].shape[0],
                tt_mix_ort[-i - 1].shape[1]
                * tt_mix_ort[-i - 1].shape[2]
                * tt_mix_ort[-i - 1].shape[-1],
            )
            npt.assert_array_almost_equal(
                tens @ tens.conj().T, np.eye(tens.shape[0]), decimal=DECIMAL
            )

    def test_orthonormalize_copy_semantics(self):
        """Orthonormalization with inplace=False must not touch the original."""
        tt = tensor_train.rand((2, 3, 4), ranks=5, seed=101)
        cores_before = [core.copy() for core in tt.cores]

        tt_orth = tt.orthonormalize("left")
        self.assertIsNot(tt_orth, tt)
        self.assertIsNone(tt.orth_center)
        for core_before, core in zip(cores_before, tt.cores):
            npt.assert_array_equal(core_before, core)

        # early exit (already satisfied) must also return an independent copy
        tt_again = tt_orth.orthonormalize("left")
        self.assertIsNot(tt_again, tt_orth)
        self.assertIsNot(tt_again.cores[0], tt_orth.cores[0])
        self.assertEqual(tt_again.orth_center, tt_orth.orth_center)

    def test_orthonormalize_partial_sweep(self):
        """Partial sweeps must keep bonds consistent and not claim a false center."""
        # partial sweep on a train with no known center: cores beyond the
        # swept range stay non-orthogonal, so no canonical form is reached
        tt = tensor_train.rand((2, 3, 4, 5), ranks=6, seed=102)
        tt_part = tt.orthonormalize("left", center_position=1)
        self.assertIsNone(tt_part.orth_center)

        # bonds must always match the actual core shapes
        for tt_check in (
            tt_part,
            tt.orthonormalize("right", center_position=2),
            tt.orthonormalize("mix", center_position=1),
        ):
            derived = [core.shape[0] for core in tt_check.cores] + [1]
            self.assertEqual(tt_check.bonds, derived)
            self.assertEqual(tt_check.max_rank, max(derived))
            npt.assert_array_almost_equal(
                tt_check.get_full_tensor(), tt.get_full_tensor(), decimal=DECIMAL
            )

        # moving the center from a known position keeps the canonical form
        tt_mix = tt.orthonormalize("mix", center_position=1)
        tt_moved = tt_mix.orthonormalize("left", center_position=2)
        self.assertEqual(tt_moved.orth_center, 2)
        npt.assert_array_almost_equal(
            tt_moved.tt_norm(),
            np.linalg.norm(tt.get_full_tensor()),
            decimal=DECIMAL,
        )

    def test_orthonormalize_center_position_validation(self):
        """Out-of-range center positions must raise."""
        tt = tensor_train.rand((2, 3, 4), ranks=5, seed=103)
        for canonical in ("left", "right", "mix"):
            with self.assertRaises(ValueError):
                tt.orthonormalize(canonical, center_position=3)
            with self.assertRaises(ValueError):
                tt.orthonormalize(canonical, center_position=-1)

    def test_svd(self):
        """Test for tensor train svd decomposition method."""
        rand_tt = tensor_train.rand((2, 3, 4), ranks=5, seed=104)
        orth_tt = rand_tt.orthonormalize("left")
        svd_tt = orth_tt.compress_svd(TEST_DELTA)
        npt.assert_almost_equal(
            svd_tt.get_full_tensor(), rand_tt.get_full_tensor(), decimal=DECIMAL
        )
        self.assertTrue(svd_tt.orth_center == 0)

        rand_tt = tensor_train.rand(((2, 2), (3, 3), (4, 4)), ranks=6)
        orth_tt = rand_tt.orthonormalize("right")
        svd_tt = orth_tt.compress_svd(TEST_DELTA, 3)
        npt.assert_array_less(svd_tt.bonds, 4)
        self.assertTrue(svd_tt.orth_center == 2)

    def test_tt_truncation(self):
        """Test for truncation method."""
        tt_1 = tensor_train.rand(physical_indexes=[3, 3, 3, 3], ranks=5, seed=105)
        tt_2 = tensor_train.rand(physical_indexes=[3, 3, 3, 3], ranks=5, seed=106)
        tt_sum = tt_1.add(tt_2, truncate=False)

        tt_trunc = tt_sum.truncate(max_rank=4)
        self.assertEqual(tt_trunc.orth_center, 0)
        npt.assert_array_less(tt_trunc.bonds, 5)

        tt_1 = tensor_train.rand(
            physical_indexes=[(4, 4), (4, 4), (4, 4)], ranks=6, seed=107
        )
        tt_2 = tensor_train.rand(
            physical_indexes=[(4, 4), (4, 4), (4, 4)], ranks=6, seed=108
        )
        tt_mult = tt_1 @ tt_2

        tt_trunc = tt_mult.truncate(max_rank=4)
        self.assertEqual(tt_trunc.orth_center, 0)
        npt.assert_array_less(tt_trunc.bonds, 5)

        tt_1 = tensor_train.rand(
            physical_indexes=[(4, 3), (4, 3), (4, 3), (4, 3)], ranks=6, seed=109
        )
        tt_2 = tensor_train.rand(physical_indexes=[3, 3, 3, 3], ranks=5, seed=110)
        tt_matvec = tt_1 @ tt_2

        tt_trunc = tt_matvec.truncate(max_rank=4, canonical="left")

        self.assertEqual(tt_trunc.orth_center, len(tt_trunc) - 1)
        npt.assert_array_less(tt_trunc.bonds, 5)

        # check orthonormality of the cores after truncation
        for i in range(len(tt_trunc) - 1):
            matrix = tt_trunc[i].reshape(
                tt_trunc[i].shape[0] * tt_trunc[i].shape[1] * tt_trunc[i].shape[2],
                tt_trunc[i].shape[-1],
            )
            npt.assert_array_almost_equal(
                matrix.conj().T @ matrix, np.eye(matrix.shape[-1]), decimal=DECIMAL
            )

    def test_tt_norm(self):
        """Test for norm method."""
        ones = np.ones((5, 5, 5))
        tt_ones = tensor_train.tt_svd(ones, error=TEST_DELTA)
        npt.assert_array_almost_equal(
            tt_ones.tt_norm(), np.linalg.norm(ones), decimal=DECIMAL
        )

        tt = tensor_train.rand(physical_indexes=[2, 2, 2], ranks=3, seed=111)
        npt.assert_array_almost_equal(
            tt.tt_norm(), np.linalg.norm(tt.get_full_tensor()), decimal=DECIMAL
        )

        ttm = tensor_train.rand(
            physical_indexes=[(3, 3), (3, 3), (3, 3)], ranks=5, seed=112
        )
        npt.assert_array_almost_equal(
            ttm.tt_norm(), np.linalg.norm(ttm.get_full_tensor()), decimal=DECIMAL
        )

    def test_tt_norm_orth_center_fast_path(self):
        """When orth_center is set, the norm comes from the center core alone."""
        tt = tensor_train.rand(physical_indexes=[2, 3, 4, 5], ranks=6, seed=113)
        dense_norm = np.linalg.norm(tt.get_full_tensor())
        for center in range(len(tt)):
            tt_canonical = tt.orthonormalize("mix", center_position=center)
            self.assertEqual(tt_canonical.orth_center, center)
            npt.assert_array_almost_equal(
                tt_canonical.tt_norm(), dense_norm, decimal=DECIMAL
            )

        # TTM with a center
        ttm = tensor_train.rand(
            physical_indexes=[(3, 3), (3, 3), (3, 3)], ranks=5, seed=114
        )
        ttm_canonical = ttm.orthonormalize("mix")
        npt.assert_array_almost_equal(
            ttm_canonical.tt_norm(),
            np.linalg.norm(ttm.get_full_tensor()),
            decimal=DECIMAL,
        )

        # complex train with a center
        tt_complex = tt.multiply(1 + 1j).orthonormalize("mix")
        npt.assert_array_almost_equal(
            tt_complex.tt_norm(), np.sqrt(2) * dense_norm, decimal=DECIMAL
        )

    def test_qtt(self):
        """Test for QTT conversion methods."""

        # standard qtt with mode 2
        power = 3
        core_num = 4
        tt_mode2 = tensor_train.rand(
            physical_indexes=[2**power] * core_num, ranks=5, seed=115
        )
        qtt_mode2 = tt_mode2.tt_to_qtt(error=TEST_DELTA)
        npt.assert_array_equal(qtt_mode2.physical_dims(), ([2] * power) * core_num)
        npt.assert_array_almost_equal(
            qtt_mode2.get_full_tensor().reshape(-1),
            tt_mode2.get_full_tensor().reshape(-1),
            decimal=DECIMAL,
        )
        tt_back = qtt_mode2.qtt_to_tt(
            physical_indexes=[2**power] * core_num, error=TEST_DELTA
        )
        npt.assert_array_almost_equal(
            tt_back.get_full_tensor().reshape(-1),
            tt_mode2.get_full_tensor().reshape(-1),
            decimal=DECIMAL,
        )

        # qtt with mode 3
        tt_mode3 = tensor_train.rand(
            physical_indexes=[3**power] * core_num, ranks=5, seed=116
        )
        qtt_mode3 = tt_mode3.tt_to_qtt(mode_size=3, error=TEST_DELTA)
        npt.assert_array_equal(qtt_mode3.physical_dims(), ([3] * power) * core_num)
        npt.assert_array_almost_equal(
            qtt_mode3.get_full_tensor().reshape(-1),
            tt_mode3.get_full_tensor().reshape(-1),
            decimal=DECIMAL,
        )
        tt_back = qtt_mode3.qtt_to_tt(
            physical_indexes=[3**power] * core_num, error=TEST_DELTA
        )
        npt.assert_array_almost_equal(
            tt_back.get_full_tensor().reshape(-1),
            tt_mode3.get_full_tensor().reshape(-1),
            decimal=DECIMAL,
        )

        # ttm in qtt form
        tt_ttm = tensor_train.rand(
            physical_indexes=[(2**power, 2**power)] * core_num, ranks=5, seed=117
        )
        qtt_ttm = tt_ttm.tt_to_qtt(error=TEST_DELTA)
        npt.assert_array_equal(qtt_ttm.physical_dims(), ([(2, 2)] * power) * core_num)
        npt.assert_array_almost_equal(
            qtt_ttm.get_full_tensor().reshape(-1),
            tt_ttm.get_full_tensor().reshape(-1),
            decimal=DECIMAL,
        )
        tt_back = qtt_ttm.qtt_to_tt(
            physical_indexes=[(2**power, 2**power)] * core_num, error=TEST_DELTA
        )
        npt.assert_array_almost_equal(
            tt_back.get_full_tensor().reshape(-1),
            tt_ttm.get_full_tensor().reshape(-1),
            decimal=DECIMAL,
        )

        # cyclic test
        tt_cyclic = tensor_train.rand(
            physical_indexes=[2**power] * core_num, ranks=8, seed=118
        )
        initial_vec = tt_cyclic.get_full_tensor().reshape(-1)
        for i in range(5):
            qtt_cyclic = tt_cyclic.tt_to_qtt(error=TEST_DELTA)
            tt_cyclic = qtt_cyclic.qtt_to_tt(
                physical_indexes=[2**power] * core_num, error=TEST_DELTA
            )
        npt.assert_array_almost_equal(
            tt_cyclic.get_full_tensor().reshape(-1),
            initial_vec,
            decimal=DECIMAL,
        )

    def test_qtt_error_bound(self):
        """Truncation error of tt_to_qtt must be bounded by error * ||A||."""
        tensor = np.random.default_rng(0).standard_normal((8, 8, 8))
        tt = tensor_train.tt_svd(tensor)
        norm = np.linalg.norm(tensor)
        for error in (3e-1, 1e-1, 1e-2, 1e-4):
            qtt = tt.tt_to_qtt(error=error)
            actual = np.linalg.norm(qtt.get_full_tensor().reshape(8, 8, 8) - tensor)
            self.assertLessEqual(actual, error * norm)

    def test_qtt_does_not_modify_source(self):
        """tt_to_qtt must not modify the source tensor train."""
        tt = tensor_train.tt_svd(np.random.default_rng(0).standard_normal((4, 4, 4)))
        cores_before = [core.copy() for core in tt.cores]
        orth_center_before = tt.orth_center

        tt.tt_to_qtt(error=TEST_DELTA)

        self.assertEqual(tt.orth_center, orth_center_before)
        for core_before, core in zip(cores_before, tt.cores):
            npt.assert_array_equal(core_before, core)

    def test_qtt_canonical_form(self):
        """The QTT returned by tt_to_qtt is left-canonical with a valid center."""
        tt = tensor_train.rand(physical_indexes=[8, 8, 8], ranks=5, seed=119)
        qtt = tt.tt_to_qtt(error=TEST_DELTA)

        self.assertEqual(qtt.orth_center, len(qtt) - 1)
        for i in range(len(qtt) - 1):
            matrix = qtt[i].reshape(-1, qtt[i].shape[-1])
            npt.assert_array_almost_equal(
                matrix.conj().T @ matrix, np.eye(matrix.shape[-1]), decimal=DECIMAL
            )
        # the center carries the norm, so tt_norm can use the fast path
        npt.assert_array_almost_equal(
            qtt.tt_norm(), np.linalg.norm(tt.get_full_tensor()), decimal=DECIMAL
        )

    def test_qtt_bond_floor(self):
        """Bonds must be floored at 1 instead of producing degenerate cores."""
        tt = tensor_train.rand(physical_indexes=[16, 16], ranks=4, seed=120)
        qtt = tt.tt_to_qtt(error=1e2)
        self.assertEqual(qtt.bonds, [1] * (len(qtt) + 1))

    def test_qtt_mixed_mode_sizes(self):
        """Cores that need no split must not corrupt the neighboring bonds."""
        tt = tensor_train.rand(physical_indexes=[2, 8], ranks=2, seed=121)
        qtt = tt.tt_to_qtt(error=TEST_DELTA)
        self.assertEqual(qtt.physical_dims(), [2, 2, 2, 2])
        npt.assert_array_almost_equal(
            qtt.get_full_tensor().reshape(-1),
            tt.get_full_tensor().reshape(-1),
            decimal=DECIMAL,
        )

        # 243 = 3^5 is falsely rejected by a float-log power check
        tt = tensor_train.rand(physical_indexes=[243, 3], ranks=2, seed=122)
        qtt = tt.tt_to_qtt(mode_size=3, error=TEST_DELTA)
        self.assertEqual(qtt.physical_dims(), [3] * 6)
        npt.assert_array_almost_equal(
            qtt.get_full_tensor().reshape(-1),
            tt.get_full_tensor().reshape(-1),
            decimal=DECIMAL,
        )

    def test_qtt_argument_validation(self):
        """Invalid arguments and non-factorable modes must raise."""
        tt = tensor_train.rand(physical_indexes=[4, 4], ranks=2, seed=123)
        with self.assertRaises(ValueError):
            tt.tt_to_qtt(max_rank=0)
        with self.assertRaises(ValueError):
            tt.tt_to_qtt(mode_size=1)
        with self.assertRaises(tensor_train.DimensionMismatch):
            tt.tt_to_qtt(mode_size=3)

        ttm_rect = tensor_train.rand(
            physical_indexes=[(4, 2), (4, 2)], ranks=2, seed=124
        )
        with self.assertRaises(tensor_train.DimensionMismatch):
            ttm_rect.tt_to_qtt()

    def test_qtt_to_tt_regrouping(self):
        """qtt_to_tt must support groups of different sizes."""
        tt = tensor_train.tt_svd(np.random.default_rng(0).standard_normal((4, 4, 4)))
        qtt = tt.tt_to_qtt(error=TEST_DELTA)  # six cores of mode 2

        for physical_indexes in ([8, 8], [2, 8, 4], [2, 2, 2, 2, 2, 2]):
            tt_back = qtt.qtt_to_tt(physical_indexes, error=TEST_DELTA)
            self.assertEqual(tt_back.physical_dims(), physical_indexes)
            npt.assert_array_almost_equal(
                tt_back.get_full_tensor().reshape(-1),
                tt.get_full_tensor().reshape(-1),
                decimal=DECIMAL,
            )

    def test_qtt_to_tt_validation(self):
        """qtt_to_tt must reject indexes that do not match the QTT cores."""
        tt = tensor_train.rand(physical_indexes=[4, 4, 4], ranks=3, seed=125)
        qtt = tt.tt_to_qtt(error=TEST_DELTA)  # six cores of mode 2

        # leftover cores are not silently dropped
        with self.assertRaises(tensor_train.DimensionMismatch):
            qtt.qtt_to_tt([4, 4], error=TEST_DELTA)

        # requested indexes need more cores than the QTT has
        with self.assertRaises(tensor_train.DimensionMismatch):
            qtt.qtt_to_tt([8, 8, 8], error=TEST_DELTA)

        # index is not a product of the QTT modes
        with self.assertRaises(tensor_train.DimensionMismatch):
            qtt.qtt_to_tt([3, 8, 8], error=TEST_DELTA)

        # TTM: column products are checked at the same position as row products
        ttm = tensor_train.rand(physical_indexes=[(4, 4), (4, 4)], ranks=3, seed=126)
        qttm = ttm.tt_to_qtt(error=TEST_DELTA)
        tt_back = qttm.qtt_to_tt([(4, 4), (4, 4)], error=TEST_DELTA)
        npt.assert_array_almost_equal(
            tt_back.get_full_tensor().reshape(-1),
            ttm.get_full_tensor().reshape(-1),
            decimal=DECIMAL,
        )
        with self.assertRaises(tensor_train.DimensionMismatch):
            qttm.qtt_to_tt([(4, 2), (4, 8)], error=TEST_DELTA)


class TestTTConstructors(unittest.TestCase):
    """Tests for tensor train constructors."""

    def test_tt_svd_respects_the_requested_error_on_a_flat_spectrum(self):
        """``error`` is an upper bound, not a per-singular-value threshold.

        A spectrum sitting just below the per-split budget is the case that
        separates the two: discarding all of it costs ``sqrt(m)`` times the
        budget, so the requested accuracy is only met if the criterion bounds
        the tail rather than each value.
        """
        size, order, error = 64, 3, 1e-2
        rng = np.random.default_rng(0)
        left = np.linalg.qr(rng.standard_normal((size, size)))[0]
        right = np.linalg.qr(rng.standard_normal((size, size)))[0]
        last = np.zeros(size)
        last[0] = 1.0
        delta = error / np.sqrt(order - 1)
        spectrum = np.concatenate([[1.0], np.full(size - 1, 0.99 * delta)])
        tensor = np.einsum("i,ai,bi,c->abc", spectrum, left, right, last)

        tt = tensor_train.tt_svd(tensor, error=error)

        achieved = np.linalg.norm(tensor - tt.get_full_tensor()) / np.linalg.norm(
            tensor
        )
        self.assertLessEqual(achieved, error)

    def test_tt_svd(self):
        """Test for tt-svd method."""
        # W state (check https://arxiv.org/abs/1603.03039 chapter 3.2)
        up, down = np.array([1, 0]), np.array([0, 1])
        first = np.kron(np.kron(down, up), up)
        second = np.kron(np.kron(up, down), up)
        third = np.kron(np.kron(up, up), down)
        tensor = first + second + third

        tt = tensor_train.tt_svd(tensor, physical_indexes=(2, 2, 2))

        self.assertEqual(tt.physical_dims(), [2] * 3)
        npt.assert_array_almost_equal(
            tt.get_full_tensor(), tensor.reshape((2, 2, 2)), decimal=DECIMAL
        )

        # only one 1 is allowed
        self.assertAlmostEqual(tt.get_element((1, 0, 0)), 1, delta=TEST_DELTA)
        self.assertAlmostEqual(tt.get_element((0, 1, 0)), 1, delta=TEST_DELTA)
        self.assertAlmostEqual(tt.get_element((0, 0, 1)), 1, delta=TEST_DELTA)

        self.assertAlmostEqual(tt.get_element((0, 0, 0)), 0, delta=TEST_DELTA)
        self.assertAlmostEqual(tt.get_element((1, 1, 1)), 0, delta=TEST_DELTA)

        # random tensor
        random_tensor = np.random.default_rng(41).random((5, 5, 5, 5))
        tt = tensor_train.tt_svd(random_tensor)
        npt.assert_array_almost_equal(
            tt.get_full_tensor(), random_tensor, decimal=DECIMAL
        )

        tt = tensor_train.tt_svd(random_tensor, max_rank=5)
        npt.assert_array_less(tt.bonds, 6)

        # random operator
        random_tensor = np.random.default_rng(42).random((2, 3, 4, 5, 2, 3, 4, 5))
        tt = tensor_train.tt_svd(
            random_tensor, physical_indexes=((2, 2), (3, 3), (4, 4), (5, 5)), ttm=True
        )
        npt.assert_array_almost_equal(
            tt.get_full_tensor(), random_tensor, decimal=DECIMAL
        )

        tt = tensor_train.tt_svd(random_tensor, max_rank=5)
        npt.assert_array_less(tt.bonds, 6)

    def test_tt_random(self):
        """Test for random tt method."""

        tt = tensor_train.rand(physical_indexes=[2, 2, 2], ranks=3, seed=127)

        self.assertEqual(tt.physical_dims(), [2] * 3)
        npt.assert_array_less(tt.bonds, 4)

        tt = tensor_train.rand(physical_indexes=((2, 2), (3, 3), (4, 4)), ranks=4)

        self.assertEqual(tt.physical_dims(), [(2, 2), (3, 3), (4, 4)])
        npt.assert_array_less(tt.bonds, 5)

    def test_tt_zeros(self):
        """Test for zeros tt method."""

        tt = tensor_train.zeros(physical_indexes=[2, 2, 2])

        self.assertEqual(tt.physical_dims(), [2] * 3)
        npt.assert_array_almost_equal(
            tt.get_full_tensor(), np.zeros((2, 2, 2)), decimal=DECIMAL
        )

        tt = tensor_train.zeros(physical_indexes=[(2, 2), (3, 3), (4, 4)], ranks=3)

        self.assertEqual(tt.physical_dims(), [(2, 2), (3, 3), (4, 4)])
        npt.assert_array_almost_equal(
            tt.get_full_tensor(), np.zeros((2, 3, 4, 2, 3, 4)), decimal=DECIMAL
        )

    def test_tt_ones(self):
        """Test for ones tt method."""

        tt = tensor_train.ones(physical_indexes=[2] * 3, ranks=2)

        self.assertEqual(tt.physical_dims(), [2] * 3)
        npt.assert_array_almost_equal(
            tt.get_full_tensor(), np.ones((2, 2, 2)) * 4, decimal=DECIMAL
        )

        tt = tensor_train.ones(physical_indexes=[(2, 2), (3, 3), (4, 4)], ranks=2)

        self.assertEqual(tt.physical_dims(), [(2, 2), (3, 3), (4, 4)])
        npt.assert_array_almost_equal(
            tt.get_full_tensor(), np.ones((2, 3, 4, 2, 3, 4)) * 4, decimal=DECIMAL
        )

    def test_tt_eye(self):
        """Test for eye tt method."""

        tt = tensor_train.eye(physical_indexes=[2] * 3)

        self.assertTrue(tt.is_ttm())
        self.assertEqual(tt.physical_dims(), [(2, 2)] * 3)
        npt.assert_array_almost_equal(
            tt.get_full_tensor().reshape(8, 8), np.eye(8), decimal=DECIMAL
        )

    def test_tt_canonical(self):
        """Test for canonical tt method."""

        tt_even = tensor_train.canonical_tt(physical_indexes=[2, 2, 2, 2], max_rank=3)

        self.assertEqual(tt_even.physical_dims(), [2] * 4)
        npt.assert_array_less(tt_even.bonds, 4)

        # check unitarity of the cores for even order
        for i in range(len(tt_even) // 2):
            # left
            tens = tt_even[i].reshape(
                tt_even[i].shape[0] * tt_even[i].shape[1], tt_even[i].shape[-1]
            )
            npt.assert_array_almost_equal(
                tens.conj().T @ tens, np.eye(tens.shape[-1]), decimal=DECIMAL
            )

            # right
            tens = tt_even[-i - 1].reshape(
                tt_even[-i - 1].shape[0],
                tt_even[-i - 1].shape[1] * tt_even[-i - 1].shape[-1],
            )
            npt.assert_array_almost_equal(
                tens @ tens.conj().T, np.eye(tens.shape[0]), decimal=DECIMAL
            )

        tt_odd = tensor_train.canonical_tt(physical_indexes=[3, 3, 3, 3, 3], max_rank=6)

        self.assertEqual(tt_odd.physical_dims(), [3] * 5)
        npt.assert_array_less(tt_odd.bonds, 7)

        # check unitarity of the cores for odd order
        for i in range(len(tt_odd) // 2):
            # left
            tens = tt_odd[i].reshape(
                tt_odd[i].shape[0] * tt_odd[i].shape[1], tt_odd[i].shape[-1]
            )
            npt.assert_array_almost_equal(
                tens.conj().T @ tens, np.eye(tens.shape[-1]), decimal=DECIMAL
            )

            # right
            tens = tt_odd[-i - 1].reshape(
                tt_odd[-i - 1].shape[0],
                tt_odd[-i - 1].shape[1] * tt_odd[-i - 1].shape[-1],
            )
            npt.assert_array_almost_equal(
                tens @ tens.conj().T, np.eye(tens.shape[0]), decimal=DECIMAL
            )

        # middle
        tens = tt_odd[len(tt_odd) // 2].reshape(
            tt_odd[len(tt_odd) // 2].shape[0] * tt_odd[len(tt_odd) // 2].shape[1],
            tt_odd[len(tt_odd) // 2].shape[-1],
        )
        npt.assert_array_almost_equal(
            tens.conj().T @ tens, np.eye(tens.shape[-1]), decimal=DECIMAL
        )


class TestTTOperations(unittest.TestCase):
    """Tests for tensor trains operations."""

    def test_tt_addition(self):
        """Test for addition (+) operation."""

        # rand tt addition
        tt1 = tensor_train.rand(physical_indexes=[2, 2, 2], ranks=2, seed=128)
        tt2 = tensor_train.rand(physical_indexes=[2, 2, 2], ranks=2, seed=129)
        tt_sum = tt1.add(tt2, truncate=False)

        self.assertEqual(tt_sum.physical_dims(), [2, 2, 2])
        self.assertEqual(tt_sum.bonds, [1, 4, 4, 1])

        npt.assert_array_almost_equal(
            tt_sum.get_full_tensor(),
            tt1.get_full_tensor() + tt2.get_full_tensor(),
            decimal=DECIMAL,
        )

        # rand tt subtraction
        tt_diff = tt1.add(-tt2, truncate=False)

        self.assertEqual(tt_diff.physical_dims(), [2, 2, 2])
        self.assertEqual(tt_diff.bonds, [1, 4, 4, 1])

        npt.assert_array_almost_equal(
            tt_diff.get_full_tensor(),
            tt1.get_full_tensor() - tt2.get_full_tensor(),
            decimal=DECIMAL,
        )

        # rand tt addition with truncation
        tt_sum_trunc = tt1.add(tt2, truncate=True)

        self.assertEqual(tt_sum_trunc.physical_dims(), [2, 2, 2])
        npt.assert_array_almost_equal(
            tt_sum_trunc.get_full_tensor(),
            tt1.get_full_tensor() + tt2.get_full_tensor(),
            decimal=DECIMAL,
        )

    def test_tt_scalar_product(self):
        """Test for scalar product (*) operation."""

        dim = 3
        max_rank = 3
        mult = 2

        tt_pos = tensor_train.ones(physical_indexes=[2] * dim, ranks=max_rank)

        mul_tt_pos = tt_pos * 2
        self.assertEqual(mul_tt_pos.physical_dims(), [2, 2, 2])
        npt.assert_array_almost_equal(
            mul_tt_pos.get_full_tensor(),
            np.ones((2, 2, 2)) * (max_rank ** (dim - 1) * mult),
            decimal=DECIMAL,
        )

        neg_tt_pos = tt_pos * -1
        npt.assert_array_almost_equal(
            neg_tt_pos.get_full_tensor(),
            np.ones((2, 2, 2)) * (max_rank ** (dim - 1) * -1),
            decimal=DECIMAL,
        )

    def test_tt_scalar_product_canonical(self):
        """Scaling a canonical train must scale the center core only."""
        tt = tensor_train.rand(physical_indexes=[2, 3, 4], ranks=4, seed=130)
        tt_canonical = tt.orthonormalize("mix", center_position=1)

        tt_scaled = tt_canonical * 2.5
        self.assertEqual(tt_scaled.orth_center, tt_canonical.orth_center)
        npt.assert_array_almost_equal(
            tt_scaled.get_full_tensor(),
            2.5 * tt.get_full_tensor(),
            decimal=DECIMAL,
        )
        # cores away from the center are untouched, so the canonical form and
        # the tt_norm fast path stay valid
        npt.assert_array_equal(tt_scaled.cores[0], tt_canonical.cores[0])
        npt.assert_array_equal(tt_scaled.cores[2], tt_canonical.cores[2])
        npt.assert_array_almost_equal(
            tt_scaled.tt_norm(),
            2.5 * np.linalg.norm(tt.get_full_tensor()),
            decimal=DECIMAL,
        )

        # complex scalar on a real canonical train must not fail on the dtype
        tt_rotated = tt_canonical * 2j
        npt.assert_array_almost_equal(
            tt_rotated.get_full_tensor(),
            2j * tt.get_full_tensor(),
            decimal=DECIMAL,
        )

    def test_tt_inner_and_outer_product(self):
        """Inner and outer products are spelled with an explicit transpose."""
        tt_lhs = tensor_train.rand(physical_indexes=[2, 3, 4], ranks=3, seed=131)
        tt_rhs = tensor_train.rand(physical_indexes=[2, 3, 4], ranks=2, seed=132)

        # inner product: TT.transpose() @ TT gives a TT with all modes 1
        inner = tt_lhs.transpose() @ tt_rhs
        self.assertEqual(inner.physical_dims(), [1, 1, 1])
        npt.assert_array_almost_equal(
            inner.get_full_tensor(),
            np.sum(tt_lhs.get_full_tensor() * tt_rhs.get_full_tensor()),
            decimal=DECIMAL,
        )

        # outer product: TT @ TT.transpose() gives a TTM
        outer = tt_lhs @ tt_rhs.transpose()
        self.assertEqual(outer.physical_dims(), [(2, 2), (3, 3), (4, 4)])
        npt.assert_array_almost_equal(
            outer.get_full_tensor().reshape(24, 24),
            np.outer(tt_lhs.get_full_tensor(), tt_rhs.get_full_tensor()),
            decimal=DECIMAL,
        )

        # no implicit transposition: TT @ TT is rejected
        with self.assertRaises(tensor_train.DimensionMismatch):
            tt_lhs.dot(tt_rhs)

    def test_tt_product(self):
        """Test for tt product (*) operation."""

        # Test for tt ones product with different ranks
        tt_lhs = tensor_train.ones(physical_indexes=[2] * 3, ranks=3)
        tt_rhs = tensor_train.ones(physical_indexes=[2] * 3, ranks=4)
        tt_prod = tt_lhs * tt_rhs

        self.assertEqual(tt_prod.physical_dims(), [2, 2, 2])
        self.assertEqual(tt_prod.bonds, [1, 12, 12, 1])
        self.assertEqual(tt_prod.get_element([0, 0, 0]), 12**2)

        # Test for tt zeros product (all elements should be zero)
        tt_lhs = tensor_train.zeros(physical_indexes=[3] * 3, ranks=3)
        tt_rhs = tensor_train.ones(physical_indexes=[3] * 3, ranks=3)
        tt_prod = tt_lhs * tt_rhs

        self.assertEqual(tt_prod.physical_dims(), [3, 3, 3])
        self.assertEqual(tt_prod.bonds, [1, 9, 9, 1])
        self.assertEqual(tt_prod.get_element([1, 1, 1]), 0)

        # Test for random tt product
        tt_lhs = tensor_train.rand(physical_indexes=[2] * 3, ranks=2, seed=133)
        tt_rhs = tensor_train.rand(physical_indexes=[2] * 3, ranks=3, seed=134)
        tt_prod = tt_lhs * tt_rhs

        self.assertEqual(tt_prod.physical_dims(), [2, 2, 2])
        self.assertEqual(tt_prod.bonds, [1, 6, 6, 1])
        npt.assert_array_almost_equal(
            tt_prod.get_full_tensor(),
            tt_lhs.get_full_tensor() * tt_rhs.get_full_tensor(),
            decimal=DECIMAL,
        )

        # Test for ttm product
        ttm_lhs = tensor_train.rand(physical_indexes=[(2, 2)] * 3, ranks=4, seed=135)
        ttm_rhs = tensor_train.rand(physical_indexes=[(2, 2)] * 3, ranks=5, seed=136)
        ttm_prod = ttm_lhs * ttm_rhs

        self.assertEqual(ttm_prod.physical_dims(), [(2, 2), (2, 2), (2, 2)])
        self.assertEqual(ttm_prod.bonds, [1, 20, 20, 1])
        npt.assert_array_almost_equal(
            ttm_prod.get_full_tensor(),
            ttm_lhs.get_full_tensor() * ttm_rhs.get_full_tensor(),
            decimal=DECIMAL,
        )

        tt_lhs = tensor_train.ones(physical_indexes=[2] * 4, ranks=3)
        tt_rhs = tensor_train.ones(physical_indexes=[2] * 3, ranks=4)

        with self.assertRaises(tensor_train.DimensionMismatch):
            tt_prod = tt_lhs * tt_rhs

    def test_tt_matvec(self):
        """Test for matvec (@) operation."""

        ph_dim1 = 2
        ph_dim2 = 3
        core_num = 4
        rank = 3

        # Test for tt operator and tt vector with compatible dimensions
        ttm = tensor_train.rand(
            physical_indexes=[(ph_dim1, ph_dim2)] * core_num, ranks=rank, seed=137
        )
        tt_vec = tensor_train.rand(
            physical_indexes=[ph_dim2] * core_num, ranks=rank, seed=138
        )
        result_tt = ttm.dot(tt_vec, truncate=False)

        self.assertEqual(result_tt.physical_dims(), [ph_dim1] * core_num)
        self.assertEqual(result_tt.bonds, [1] + [rank**2] * (core_num - 1) + [1])
        npt.assert_array_almost_equal(
            result_tt.get_full_tensor(),
            np.tensordot(
                ttm.get_full_tensor(),
                tt_vec.get_full_tensor(),
                axes=([4, 5, 6, 7], [0, 1, 2, 3]),
            ),
            decimal=DECIMAL,
        )

        # Test for reversed multiplication
        ttm = tensor_train.rand(
            physical_indexes=[(ph_dim1, ph_dim2)] * core_num, ranks=rank, seed=139
        )
        tt_vec = tensor_train.rand(
            physical_indexes=[ph_dim1] * core_num, ranks=rank, seed=140
        ).transpose()
        result_tt = (tt_vec.dot(ttm, truncate=False)).transpose()

        self.assertEqual(result_tt.physical_dims(), [ph_dim2] * core_num)
        self.assertEqual(result_tt.bonds, [1] + [rank**2] * (core_num - 1) + [1])
        npt.assert_array_almost_equal(
            result_tt.get_full_tensor(),
            np.tensordot(
                tt_vec.get_full_tensor(),
                ttm.get_full_tensor(),
                axes=([4, 5, 6, 7], [0, 1, 2, 3]),
            ).reshape([ph_dim2] * core_num),
            decimal=DECIMAL,
        )

        # Test for eye operator product
        ttm_eye = tensor_train.eye([ph_dim1] * core_num)
        ttm_rand = tensor_train.rand(
            physical_indexes=[(ph_dim1, ph_dim1)] * core_num, ranks=rank, seed=141
        )
        result_tt = ttm_eye.dot(ttm_rand, truncate=False)

        self.assertEqual(result_tt.physical_dims(), [(ph_dim1, ph_dim1)] * core_num)
        self.assertEqual(result_tt.bonds, [1] + [rank] * (core_num - 1) + [1])
        npt.assert_array_almost_equal(
            result_tt.get_full_tensor(),
            ttm_rand.get_full_tensor(),
            decimal=DECIMAL,
        )

        # Test for overlap
        tt_lhs = tensor_train.ones(physical_indexes=[ph_dim1] * core_num, ranks=rank)
        tt_rhs = tensor_train.ones(physical_indexes=[ph_dim1] * core_num, ranks=rank)
        result = tt_lhs.transpose().dot(tt_rhs, truncate=False)
        self.assertEqual(result.physical_dims(), [1] * core_num)
        self.assertEqual(result.bonds, [1] + [rank**2] * (core_num - 1) + [1])
        scalar_result = ph_dim1**core_num * (rank**2) ** (
            core_num - 1
        )  # 2^4 from physical indexes and 9^3 from ranks
        self.assertAlmostEqual(
            result.get_full_tensor(), scalar_result, delta=TEST_DELTA
        )

        # Test for random operator and vector with truncation
        rng = np.random.default_rng(43)
        op = rng.random((ph_dim2**core_num, ph_dim2**core_num))
        vec = rng.random(ph_dim2**core_num)
        ttm = tensor_train.tt_svd(
            op.reshape([ph_dim2] * (core_num * 2)),
            physical_indexes=[(ph_dim2, ph_dim2)] * core_num,
            ttm=True,
        )
        tt_vec = tensor_train.tt_svd(
            vec.reshape([ph_dim2] * core_num), physical_indexes=[ph_dim2] * core_num
        )
        tt_prod = ttm @ tt_vec
        npt.assert_array_almost_equal(
            tt_prod.get_full_tensor().reshape(ph_dim2**core_num),
            op @ vec,
            decimal=DECIMAL,
        )

        ttm = tensor_train.rand(
            physical_indexes=[(ph_dim1, ph_dim1)] * core_num, ranks=rank, seed=142
        )
        tt_vec = tensor_train.rand(
            physical_indexes=[ph_dim2] * core_num, ranks=rank, seed=143
        )

        with self.assertRaises(tensor_train.DimensionMismatch):
            tt_prod = ttm @ tt_vec

    def test_scalar_product_accepts_numpy_scalars(self):
        """NumPy scalars must scale a train like their Python counterparts.

        ``np.float64`` and ``np.complex128`` subclass ``float``/``complex``, but
        ``np.int64`` subclasses neither, so an isinstance check against the
        Python numeric types alone rejects the integer that any NumPy reduction
        (``arr.sum()``, ``np.prod(shape)``) naturally produces.
        """

        tt_rand = tensor_train.rand(physical_indexes=[2] * 3, ranks=2, seed=144)
        reference = tt_rand.get_full_tensor() * 3

        for scalar in (
            3,
            np.int8(3),
            np.int32(3),
            np.int64(3),
            np.uint8(3),
            3.0,
            np.float32(3.0),
            np.float64(3.0),
            3 + 0j,
            np.complex128(3),
        ):
            with self.subTest(scalar=type(scalar).__name__):
                npt.assert_allclose(
                    (tt_rand * scalar).get_full_tensor(),
                    reference,
                    rtol=1e-12,
                    atol=1e-12,
                )

    def test_scalar_product_rejects_non_numbers(self):
        """A non-numeric operand must still fail fast."""

        tt_rand = tensor_train.rand(physical_indexes=[2] * 3, ranks=2, seed=145)
        for operand in ("2", [2], None, np.array([1.0, 2.0])):
            with self.subTest(operand=type(operand).__name__):
                with self.assertRaises(TypeError):
                    tt_rand.multiply(operand)


class TestTTCanonicalValidation(unittest.TestCase):
    """Every entry point that takes ``canonical`` must reject a bad value."""

    def test_canonical_argument_is_validated(self):
        """orthonormalize, compress_svd and truncate must agree.

        truncate forwards nothing to compress_svd, so without its own guard an
        unknown form silently falls through to a left-canonical result instead
        of failing.
        """

        tt_rand = tensor_train.rand(physical_indexes=[2] * 4, ranks=3, seed=146)
        for method in ("orthonormalize", "compress_svd", "truncate"):
            with self.subTest(method=method):
                with self.assertRaises(ValueError):
                    getattr(tt_rand, method)(canonical="banana")

    def test_valid_canonical_forms_are_accepted(self):
        """The guard must not reject the forms each method documents."""

        tt_rand = tensor_train.rand(physical_indexes=[2] * 4, ranks=3, seed=147)
        reference = tt_rand.get_full_tensor()

        for method, forms in (
            ("orthonormalize", ("left", "right", "mix")),
            ("compress_svd", ("left", "right", "auto")),
            ("truncate", ("left", "right", "auto")),
        ):
            for form in forms:
                with self.subTest(method=method, form=form):
                    result = getattr(tt_rand, method)(canonical=form)
                    npt.assert_allclose(
                        result.get_full_tensor(), reference, rtol=1e-10, atol=1e-12
                    )


class TestTTRandomState(unittest.TestCase):
    """``rand`` must draw its own numbers without touching global state."""

    @staticmethod
    def _global_draw_after(action) -> np.ndarray:
        """Seed the global RNG, run ``action``, then draw from the global RNG."""
        np.random.seed(20240617)
        action()
        return np.random.rand(4)

    def test_rand_leaves_the_global_numpy_rng_alone(self):
        """A library call must not cost the caller their reproducibility.

        ``np.random.seed(None)`` reseeds the global generator from OS entropy,
        so seeding a train from the global stream makes every caller who set
        their own seed irreproducible the moment they build a random train.
        """

        first = self._global_draw_after(
            lambda: tensor_train.rand(physical_indexes=[2] * 3, ranks=2, seed=148)
        )
        second = self._global_draw_after(
            lambda: tensor_train.rand(physical_indexes=[2] * 3, ranks=2, seed=149)
        )
        npt.assert_array_equal(first, second)

    def test_seeded_rand_leaves_the_global_numpy_rng_alone(self):
        """Passing a seed must scope it to the train, not to the process."""

        first = self._global_draw_after(
            lambda: tensor_train.rand(physical_indexes=[2] * 3, ranks=2, seed=5)
        )
        second = self._global_draw_after(
            lambda: tensor_train.rand(physical_indexes=[2] * 3, ranks=2, seed=99)
        )
        npt.assert_array_equal(first, second)

    def test_same_seed_gives_the_same_train(self):
        """The reproducibility the seed does promise must hold."""

        first = tensor_train.rand(physical_indexes=[2] * 3, ranks=3, seed=7)
        second = tensor_train.rand(physical_indexes=[2] * 3, ranks=3, seed=7)
        npt.assert_array_equal(first.get_full_tensor(), second.get_full_tensor())

    def test_different_seeds_give_different_trains(self):
        """Distinct seeds must not collapse onto one train."""

        first = tensor_train.rand(physical_indexes=[2] * 3, ranks=3, seed=7)
        second = tensor_train.rand(physical_indexes=[2] * 3, ranks=3, seed=8)
        self.assertFalse(np.allclose(first.get_full_tensor(), second.get_full_tensor()))

    def test_unseeded_rand_draws_a_fresh_train_each_call(self):
        """Without a seed the trains must still differ from one another."""

        first = tensor_train.rand(physical_indexes=[2] * 3, ranks=3, seed=150)
        second = tensor_train.rand(physical_indexes=[2] * 3, ranks=3, seed=151)
        self.assertFalse(np.allclose(first.get_full_tensor(), second.get_full_tensor()))

    def test_ttm_rand_is_seeded_the_same_way(self):
        """The TTM branch draws from the same generator."""

        first = tensor_train.rand(physical_indexes=[(2, 3)] * 3, ranks=2, seed=3)
        second = tensor_train.rand(physical_indexes=[(2, 3)] * 3, ranks=2, seed=3)
        npt.assert_array_equal(first.get_full_tensor(), second.get_full_tensor())
        self.assertEqual(first.physical_dims(), [(2, 3), (2, 3), (2, 3)])


class TestTTConstructionValidation(unittest.TestCase):
    """``TensorTrain.__init__`` must reject malformed core lists."""

    @staticmethod
    def _core(left, row, right, col=None):
        shape = (left, row, right) if col is None else (left, row, col, right)
        return np.ones(shape)

    def test_rejects_fewer_than_two_cores(self):
        for cores in ([], [np.ones((1, 2, 1))]):
            with self.subTest(n_cores=len(cores)):
                with self.assertRaises(tensor_train.WrongTTFormat):
                    tensor_train.TensorTrain(cores)

    def test_rejects_cores_that_are_not_rank_three_or_four(self):
        for bad in (np.ones((1, 2)), np.ones((1, 2, 2, 2, 1))):
            with self.subTest(ndim=bad.ndim):
                with self.assertRaises(tensor_train.WrongTTFormat):
                    tensor_train.TensorTrain([bad, np.ones((1, 2, 1))])

    def test_rejects_mixed_core_ranks(self):
        """A rank-3 core beside a rank-4 one is neither a TT nor a TTM."""
        with self.assertRaises(tensor_train.WrongTTFormat):
            tensor_train.TensorTrain([np.ones((1, 2, 2)), np.ones((2, 2, 2, 1))])

    def test_rejects_periodic_boundary_on_the_first_core(self):
        with self.assertRaises(tensor_train.WrongTTFormat):
            tensor_train.TensorTrain([self._core(2, 2, 2), self._core(2, 2, 1)])

    def test_rejects_periodic_boundary_on_the_last_core(self):
        with self.assertRaises(tensor_train.WrongTTFormat):
            tensor_train.TensorTrain([self._core(1, 2, 2), self._core(2, 2, 2)])

    def test_rejects_mismatched_neighbouring_bonds(self):
        """The right bond of core i must equal the left bond of core i+1."""
        with self.assertRaises(tensor_train.DimensionMismatch):
            tensor_train.TensorTrain([self._core(1, 2, 3), self._core(2, 2, 1)])


class TestTTArithmeticContracts(unittest.TestCase):
    """Operators must agree with the named methods they delegate to."""

    def setUp(self):
        self.lhs = tensor_train.rand(physical_indexes=[2] * 3, ranks=2, seed=11)
        self.rhs = tensor_train.rand(physical_indexes=[2] * 3, ranks=3, seed=12)

    def test_add_operator_matches_dense_addition(self):
        """``a + b`` is only ever reached through the operator, never .add()."""
        npt.assert_allclose(
            (self.lhs + self.rhs).get_full_tensor(),
            self.lhs.get_full_tensor() + self.rhs.get_full_tensor(),
            rtol=1e-10,
            atol=1e-12,
        )

    def test_sub_operator_matches_dense_subtraction(self):
        npt.assert_allclose(
            (self.lhs - self.rhs).get_full_tensor(),
            self.lhs.get_full_tensor() - self.rhs.get_full_tensor(),
            rtol=1e-10,
            atol=1e-12,
        )

    def test_scalar_multiplication_commutes(self):
        """``3 * a`` goes through __rmul__, ``a * 3`` through __mul__."""
        npt.assert_allclose(
            (3 * self.lhs).get_full_tensor(),
            (self.lhs * 3).get_full_tensor(),
            rtol=1e-12,
            atol=1e-14,
        )

    def test_multiply_with_truncation_preserves_the_value(self):
        """``truncate=True`` may shrink the bonds but not change the tensor."""
        exact = self.lhs.multiply(self.rhs, truncate=False)
        truncated = self.lhs.multiply(self.rhs, truncate=True)

        npt.assert_allclose(
            truncated.get_full_tensor(),
            exact.get_full_tensor(),
            rtol=1e-9,
            atol=1e-11,
        )
        self.assertLessEqual(max(truncated.bonds), max(exact.bonds))

    def test_add_rejects_a_non_tensor_train(self):
        for operand in (1.0, "tt", [1, 2]):
            with self.subTest(operand=type(operand).__name__):
                with self.assertRaises(TypeError):
                    self.lhs.add(operand)

    def test_add_rejects_mismatched_physical_dimensions(self):
        with self.assertRaises(tensor_train.DimensionMismatch):
            self.lhs.add(tensor_train.rand(physical_indexes=[3] * 3, ranks=2, seed=13))

    def test_dot_rejects_a_non_tensor_train(self):
        with self.assertRaises(TypeError):
            self.lhs.dot(2.0)


class TestTTInspection(unittest.TestCase):
    """Element access, conjugation, string form and truncation direction."""

    def test_get_element_rejects_the_wrong_number_of_indices(self):
        tt_rand = tensor_train.rand(physical_indexes=[2] * 3, ranks=2, seed=14)
        for index in ([0, 0], [0, 0, 0, 0]):
            with self.subTest(n_indexes=len(index)):
                with self.assertRaises(tensor_train.DimensionMismatch):
                    tt_rand.get_element(index)

    def test_get_element_accepts_numpy_integer_indices(self):
        """NumPy integers reach this method from array-valued index lists."""
        tt_rand = tensor_train.rand(physical_indexes=[2, 3, 4], ranks=3, seed=15)
        expected = tt_rand.get_element([1, 2, 3])
        for index in (np.array([1, 2, 3]), list(np.array([1, 2, 3]))):
            with self.subTest(container=type(index).__name__):
                self.assertEqual(tt_rand.get_element(index), expected)

    def test_conj_inplace_writes_into_the_receiver(self):
        """``inplace=True`` must conjugate this train, not return a new one."""
        # two rank-3 cores of shape (1, 2, 1), so the train is a 2x2 vector
        cores = [
            np.array([[[1 + 2j], [3 - 1j]]]),
            np.array([[[2 - 3j], [1 + 1j]]]),
        ]
        train = tensor_train.TensorTrain(cores)
        reference = np.conj(train.get_full_tensor())

        result = train.conj(inplace=True)

        self.assertIs(result, train)
        npt.assert_allclose(train.get_full_tensor(), reference, rtol=1e-12)

    def test_conj_without_inplace_leaves_the_receiver_alone(self):
        train = tensor_train.rand(physical_indexes=[2] * 3, ranks=2, seed=15)
        before = train.get_full_tensor()

        result = train.conj()

        self.assertIsNot(result, train)
        npt.assert_array_equal(train.get_full_tensor(), before)

    def test_str_reports_the_train_type_and_shape(self):
        tt_rand = tensor_train.rand(physical_indexes=[2] * 3, ranks=2, seed=16)
        ttm_rand = tensor_train.rand(physical_indexes=[(2, 2)] * 3, ranks=2, seed=17)

        self.assertIn(tt_rand.tt_type.value, str(tt_rand))
        self.assertIn(ttm_rand.tt_type.value, str(ttm_rand))

    def test_truncate_from_a_left_leaning_centre_preserves_the_value(self):
        """orth_center in the left half sends truncate down its other branch."""
        train = tensor_train.rand(physical_indexes=[2] * 6, ranks=4, seed=18)
        train = train.orthonormalize(canonical="right")
        self.assertLess(train.orth_center, train.n_cores / 2)
        reference = train.get_full_tensor()

        truncated = train.truncate(error=1e-12)

        npt.assert_allclose(
            truncated.get_full_tensor(), reference, rtol=1e-9, atol=1e-11
        )


class TestTTSvdValidation(unittest.TestCase):
    """``tt_svd`` needs physical indexes whenever it cannot infer them."""

    def test_rejects_a_flat_tensor_without_physical_indexes(self):
        with self.assertRaises(tensor_train.AmbiguousFormat):
            tensor_train.tt_svd(np.ones(8))

    def test_rejects_ttm_without_physical_indexes(self):
        with self.assertRaises(tensor_train.AmbiguousFormat):
            tensor_train.tt_svd(np.ones((2, 2, 2, 2)), ttm=True)


class TestTTConstructorGuards(unittest.TestCase):
    """Every constructor needs at least two cores to form a train."""

    def test_constructors_reject_fewer_than_two_physical_indexes(self):
        for name, call in (
            ("ones", lambda dims: tensor_train.ones(dims)),
            ("zeros", lambda dims: tensor_train.zeros(dims)),
            ("eye", lambda dims: tensor_train.eye(dims)),
            ("rand", lambda dims: tensor_train.rand(dims, seed=152)),
            ("canonical_tt", lambda dims: tensor_train.canonical_tt(dims, max_rank=2)),
        ):
            for dims in ([], [2]):
                with self.subTest(constructor=name, n_dims=len(dims)):
                    with self.assertRaises(tensor_train.WrongTTFormat):
                        call(dims)


class TestTTExceptions(unittest.TestCase):
    """The package exceptions carry the message they are given."""

    def test_exceptions_keep_a_custom_message(self):
        for exception in (
            tensor_train.DimensionMismatch,
            tensor_train.WrongTTFormat,
            tensor_train.WrongTTType,
            tensor_train.AmbiguousFormat,
        ):
            with self.subTest(exception=exception.__name__):
                error = exception("custom text")
                self.assertEqual(str(error), "custom text")
                self.assertEqual(error.msg, "custom text")

    def test_exceptions_have_a_default_message(self):
        for exception in (
            tensor_train.DimensionMismatch,
            tensor_train.WrongTTFormat,
            tensor_train.WrongTTType,
            tensor_train.AmbiguousFormat,
        ):
            with self.subTest(exception=exception.__name__):
                self.assertTrue(str(exception()))


class TestTTConjugateTranspose(unittest.TestCase):
    """``transpose(conjugate=True)`` is the TTM adjoint, in place or not."""

    @staticmethod
    def _complex_ttm(seed: int) -> tensor_train.TensorTrain:
        real = tensor_train.rand(physical_indexes=[(2, 3)] * 3, ranks=2, seed=seed)
        imag = tensor_train.rand(
            physical_indexes=[(2, 3)] * 3, ranks=2, seed=seed + 100
        )
        cores = [(r + 1j * i).astype(complex) for r, i in zip(real.cores, imag.cores)]
        return tensor_train.TensorTrain(cores)

    def test_conjugate_transpose_inplace_matches_the_copying_path(self):
        """The in-place branch writes through ``np.conj(..., out=...)``.

        Both branches must land on the same adjoint; only the identity of the
        returned object differs.
        """

        copying = self._complex_ttm(21)
        in_place = self._complex_ttm(21)

        expected = copying.transpose(conjugate=True)
        returned = in_place.transpose(conjugate=True, inplace=True)

        self.assertIs(returned, in_place)
        npt.assert_allclose(
            in_place.get_full_tensor(), expected.get_full_tensor(), rtol=1e-12
        )

    def test_conjugate_transpose_is_the_matrix_adjoint(self):
        """Against dense: (A^H)[j, i] == conj(A[i, j])."""

        ttm = self._complex_ttm(22)
        dense = np.asarray(ttm.get_full_tensor())

        adjoint = ttm.transpose(conjugate=True)

        # row and column modes swap, so the dense form transposes its two halves
        n_cores = ttm.n_cores
        axes = list(range(n_cores, 2 * n_cores)) + list(range(n_cores))
        npt.assert_allclose(
            np.asarray(adjoint.get_full_tensor()),
            np.conj(dense).transpose(axes),
            rtol=1e-12,
        )

    def test_transpose_without_inplace_leaves_the_receiver_alone(self):
        ttm = self._complex_ttm(23)
        before = np.asarray(ttm.get_full_tensor()).copy()

        result = ttm.transpose(conjugate=True)

        self.assertIsNot(result, ttm)
        npt.assert_array_equal(np.asarray(ttm.get_full_tensor()), before)
