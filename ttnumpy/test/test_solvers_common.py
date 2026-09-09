"""Tests for the shared solver machinery: vocabulary, environments and the
one- and two-site local algebra."""

import types
import unittest
from unittest import mock

import numpy as np
import pytest

from ttnumpy.logger_config import PACKAGE_LOGGER_NAME
import ttnumpy
from ttnumpy import solvers, tensor_train
from ttnumpy.solvers import amen as amen_module
from ttnumpy.solvers import two_site
from ttnumpy.solvers.als import als
from ttnumpy.solvers.amen import amen
from ttnumpy.solvers.common import (
    LocalSolver,
    StopReason,
    TruncationMode,
    _DEFAULT_LOCAL_RTOL,
    _SolverSettings,
    _choose_truncation_rank,
    prepare_left_environment,
    prepare_right_environment,
    relative_residual,
)
from ttnumpy.solvers.mals import mals
from ttnumpy.solvers.one_site import (
    local_explicit_matrix_one_site,
    local_operator_one_site,
    local_rhs_one_site,
)
from ttnumpy.solvers.two_site import (
    local_explicit_matrix_two_site,
    local_operator_two_site,
)
from solver_test_utils import qtt_laplacian_1d

DECIMAL = 10


class TestSolversCommon(unittest.TestCase):
    """Tests for relative_residual, the environment builders, the settings and
    the one- and two-site local operators shared by the solvers."""

    def test_relative_residual_matches_explicit(self):
        """Relative residual in TT arithmetic must match the explicit computation."""

        order = 6
        rng = np.random.default_rng(0)
        operator_mat = rng.random((2**order, 2**order))
        x_vec = rng.random(2**order)
        rhs_vec = operator_mat @ x_vec

        operator_tt = tensor_train.tt_svd(
            operator_mat.reshape([2] * 2 * order),
            physical_indexes=[(2, 2)] * order,
            ttm=True,
            error=0,
        )
        x_tt = tensor_train.tt_svd(x_vec.reshape([2] * order), error=0)
        rhs_tt = tensor_train.tt_svd(rhs_vec.reshape([2] * order), error=0)

        # exact solution: residual must vanish
        self.assertLess(relative_residual(operator_tt, rhs_tt, x_tt), 1e-10)

        # perturbed solution: residual must match the explicit value
        y_tt = tensor_train.tt_svd((2.0 * x_vec).reshape([2] * order), error=0)
        expected = np.linalg.norm(
            operator_mat @ (2.0 * x_vec) - rhs_vec
        ) / np.linalg.norm(rhs_vec)
        self.assertAlmostEqual(
            relative_residual(operator_tt, rhs_tt, y_tt),
            expected,
            places=DECIMAL,
        )

    def test_same_train_environment_matches_reference(self):
        """The frame/target environment builder, fed the same train on both
        sides, must reproduce the plain same-train recursion.

        The reference recursion is inlined here so the check does not depend
        on the production builder it is validating.
        """

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
        n_cores = len(x)

        # independent same-train recursion (the old prepare_*_contractions)
        ref_right_A = [None] * n_cores
        ref_right_b = [None] * n_cores
        ref_left_A = [None] * n_cores
        ref_left_b = [None] * n_cores
        for ind in range(n_cores - 1, -1, -1):
            if ind == n_cores - 1:
                ref_right_A[-1] = np.ones([1, 1, 1])
                ref_right_b[-1] = np.ones([1, 1])
            else:
                ref_right_A[ind] = np.einsum(
                    "idc,jdeb,kea,abc->kji",
                    x[ind + 1][:, :, 0, :].conj(),
                    operator_tt[ind + 1],
                    x[ind + 1][:, :, 0, :],
                    ref_right_A[ind + 1],
                )
                ref_right_b[ind] = np.einsum(
                    "jcb,ica,ab->ij",
                    x[ind + 1][:, :, 0, :].conj(),
                    rhs_tt[ind + 1][:, :, 0, :],
                    ref_right_b[ind + 1],
                )
        for ind in range(n_cores):
            if ind == 0:
                ref_left_A[0] = np.ones([1, 1, 1])
                ref_left_b[0] = np.ones([1, 1])
            else:
                ref_left_A[ind] = np.einsum(
                    "abc,cek,bedj,adi->ijk",
                    ref_left_A[ind - 1],
                    x[ind - 1][:, :, 0, :].conj(),
                    operator_tt[ind - 1],
                    x[ind - 1][:, :, 0, :],
                )
                ref_left_b[ind] = np.einsum(
                    "ab,bcj,aci->ij",
                    ref_left_b[ind - 1],
                    x[ind - 1][:, :, 0, :].conj(),
                    rhs_tt[ind - 1][:, :, 0, :],
                )

        right_A = [None] * n_cores
        right_b = [None] * n_cores
        left_A = [None] * n_cores
        left_b = [None] * n_cores
        for i in range(n_cores - 1, -1, -1):
            prepare_right_environment(i, right_A, x, x, operator_tt)
            prepare_right_environment(i, right_b, x, rhs_tt)
        for k in range(n_cores):
            prepare_left_environment(k, left_A, x, x, operator_tt)
            prepare_left_environment(k, left_b, x, rhs_tt)

        for k in range(n_cores):
            np.testing.assert_allclose(right_A[k], ref_right_A[k], atol=1e-12)
            np.testing.assert_allclose(right_b[k], ref_right_b[k], atol=1e-12)
            np.testing.assert_allclose(left_A[k], ref_left_A[k], atol=1e-12)
            np.testing.assert_allclose(left_b[k], ref_left_b[k], atol=1e-12)

    def test_overlap_environment_shapes(self):
        """z-onto-x overlaps, which AMEn needs to enrich, must be bond sized."""

        order = 4
        x = tensor_train.rand([2] * order, ranks=3, seed=1).orthonormalize(
            canonical="right"
        )
        z = tensor_train.rand([2] * order, ranks=2, seed=2).orthonormalize(
            canonical="right"
        )
        n_cores = len(x)

        left_env_xz = [None] * n_cores
        right_env_xz = [None] * n_cores
        for i in range(n_cores - 1, -1, -1):
            prepare_right_environment(i, right_env_xz, x, z)
        for k in range(n_cores):
            prepare_left_environment(k, left_env_xz, x, z)

        # environments are indexed (target_bond, frame_bond) = (z_bond, x_bond)
        for k in range(n_cores):
            self.assertEqual(left_env_xz[k].shape, (z.bonds[k], x.bonds[k]))
            self.assertEqual(right_env_xz[k].shape, (z.bonds[k + 1], x.bonds[k + 1]))

    def test_truncation_mode_from_residual_tol(self):
        """The truncation mode is residual-budgeted iff residual_tol is set."""
        settings_kwargs = dict(
            max_sweeps=1,
            stagnation_factor="auto",
            local_solver="auto",
            symmetric=True,
            direct_solve_size=1000,
            return_info=False,
        )
        with_tol = _SolverSettings.resolve(residual_tol=1e-8, **settings_kwargs)
        without_tol = _SolverSettings.resolve(residual_tol=None, **settings_kwargs)
        self.assertIs(with_tol.truncation_mode, TruncationMode.RESIDUAL)
        self.assertIs(without_tol.truncation_mode, TruncationMode.THRESHOLD)
        # str mixin keeps the plain value form
        self.assertEqual(TruncationMode.RESIDUAL, "residual")

    def test_one_site_local_pieces_match_explicit(self):
        """One-site local operator/rhs must be the true projected system."""

        order = 4
        rng = np.random.default_rng(11)
        operator_mat = rng.random((2**order, 2**order))
        operator_tt = tensor_train.tt_svd(
            operator_mat.reshape([2] * 2 * order),
            physical_indexes=[(2, 2)] * order,
            ttm=True,
            error=0,
        )
        rhs_tt = tensor_train.tt_svd(rng.random(2**order).reshape([2] * order), error=0)
        x = tensor_train.rand([2] * order, ranks=3, seed=7).orthonormalize(
            canonical="right"
        )

        n_cores = len(x)
        right_A = [None] * n_cores
        right_b = [None] * n_cores
        left_A = [None] * n_cores
        left_b = [None] * n_cores
        # the one-site system at k consumes the right environment at k, so
        # index 0 must be built as well
        for i in range(n_cores - 1, -1, -1):
            prepare_right_environment(i, right_A, x, x, operator_tt)
            prepare_right_environment(i, right_b, x, rhs_tt)

        for k in range(n_cores):
            prepare_left_environment(k, left_A, x, x, operator_tt)
            prepare_left_environment(k, left_b, x, rhs_tt)
            explicit = local_explicit_matrix_one_site(k, left_A, right_A, operator_tt)
            rhs = local_rhs_one_site(k, left_b, right_b, rhs_tt)
            size = x.bonds[k] * x.cores[k].shape[1] * x.bonds[k + 1]

            # the local system must be square and sized by the core it spans
            self.assertEqual(explicit.shape, (size, size))
            self.assertEqual(rhs.shape, (size,))

            # matrix-free operator must reproduce the explicit one
            operator = local_operator_one_site(
                left_A[k], operator_tt.cores[k], right_A[k], size
            )
            vector = rng.random(size)
            block = rng.random((size, 4))
            scale = np.linalg.norm(explicit)
            self.assertLess(
                np.linalg.norm(operator @ vector - explicit @ vector), 1e-10 * scale
            )
            self.assertLess(
                np.linalg.norm(operator @ block - explicit @ block), 1e-10 * scale
            )

    def test_local_operator_two_site_matches_explicit(self):
        """Matrix-free local operator must match the explicit local matrix."""

        order = 4
        rng = np.random.default_rng(3)
        operator_mat = rng.random((2**order, 2**order))
        operator_tt = tensor_train.tt_svd(
            operator_mat.reshape([2] * 2 * order),
            physical_indexes=[(2, 2)] * order,
            ttm=True,
            error=0,
        )
        rhs_tt = tensor_train.tt_svd(rng.random(2**order).reshape([2] * order), error=0)
        x = tensor_train.rand([2] * order, ranks=3, seed=5).orthonormalize(
            canonical="right"
        )

        n_cores = len(x)
        right_A = [None] * n_cores
        right_b = [None] * n_cores
        left_A = [None] * n_cores
        left_b = [None] * n_cores
        for i in range(n_cores - 1, 0, -1):
            prepare_right_environment(i, right_A, x, x, operator_tt)
            prepare_right_environment(i, right_b, x, rhs_tt)

        for ind in (0, 1):
            prepare_left_environment(ind, left_A, x, x, operator_tt)
            prepare_left_environment(ind, left_b, x, rhs_tt)
            explicit = local_explicit_matrix_two_site(ind, left_A, right_A, operator_tt)
            size = explicit.shape[0]
            operator = local_operator_two_site(
                left_A[ind],
                operator_tt.cores[ind],
                operator_tt.cores[ind + 1],
                right_A[ind + 1],
                size,
            )

            vector = rng.random(size)
            block = rng.random((size, 5))
            scale = np.linalg.norm(explicit)
            self.assertLess(
                np.linalg.norm(operator @ vector - explicit @ vector), 1e-12 * scale
            )
            self.assertLess(
                np.linalg.norm(operator @ block - explicit @ block), 1e-12 * scale
            )

    def test_solver_info_counts_local_solves(self):
        """SolverInfo must report the run's iterative local solves.

        The tallies are kept per sweep for the log lines, so the value on
        SolverInfo has to be the sum over the run rather than the last
        sweep's; a two-sweep run is checked against a one-sweep one to pin
        that down.
        """

        order = 6
        laplacian_tt, rhs_tt, _ = qtt_laplacian_1d(order)
        # rank 2 cannot represent the rank-3 solution, so ALS never reaches
        # the target and every sweep runs in full
        guess = tensor_train.ones([2] * order, ranks=2).compress_svd(canonical="right")
        shared = dict(
            residual_tol=1e-6,
            local_rtol=1e-8,
            stagnation_factor=None,
            symmetric=True,
            return_info=True,
        )

        # the direct path is deliberately not counted: it never attempted an
        # iterative solve, so it cannot have missed a tolerance
        _, direct = als(
            laplacian_tt,
            rhs_tt,
            x0=guess.copy(),
            max_sweeps=2,
            local_solver="direct",
            **shared,
        )
        self.assertEqual(direct.local_solves, 0)
        self.assertEqual(direct.local_fallbacks, 0)
        self.assertEqual(direct.local_inexact, 0)

        # direct_solve_size=0 forces every local system onto MINRES
        iterative = dict(local_solver="minres", direct_solve_size=0)
        _, one_sweep = als(
            laplacian_tt, rhs_tt, x0=guess.copy(), max_sweeps=1, **iterative, **shared
        )
        _, two_sweeps = als(
            laplacian_tt, rhs_tt, x0=guess.copy(), max_sweeps=2, **iterative, **shared
        )
        # ALS visits 2 * n_cores - 1 sites per sweep with fixed ranks
        self.assertEqual(one_sweep.local_solves, 2 * order - 1)
        self.assertEqual(two_sweeps.local_solves, 2 * one_sweep.local_solves)
        self.assertGreaterEqual(
            two_sweeps.local_solves,
            two_sweeps.local_fallbacks + two_sweeps.local_inexact,
        )

        # the two-site driver reports through the same monitor
        _, mals_direct = mals(
            laplacian_tt, rhs_tt, max_sweeps=1, local_solver="direct", **shared
        )
        _, mals_iterative = mals(
            laplacian_tt, rhs_tt, max_sweeps=1, **iterative, **shared
        )
        self.assertEqual(mals_direct.local_solves, 0)
        self.assertGreater(mals_iterative.local_solves, 0)

    def test_truncation_budget_splits_between_steps(self):
        """A sweep's truncations share the residual budget in quadrature."""
        settings_kwargs = dict(
            max_sweeps=1,
            stagnation_factor="auto",
            local_solver="auto",
            symmetric=True,
            direct_solve_size=1000,
            return_info=False,
        )
        many = _SolverSettings.resolve(
            residual_tol=1e-6, truncation_steps=25, **settings_kwargs
        )
        single = _SolverSettings.resolve(
            residual_tol=1e-6, truncation_steps=1, **settings_kwargs
        )
        # spending the full tolerance per truncation is what left many-core
        # solves a factor sqrt(steps) short of their target
        self.assertAlmostEqual(many.truncation_tol, 1e-6 / 5.0)
        self.assertAlmostEqual(single.truncation_tol, 1e-6)
        # the stopping test is unaffected
        self.assertEqual(many.residual_tol, 1e-6)
        # nothing to budget without a residual target
        self.assertIsNone(
            _SolverSettings.resolve(
                residual_tol=None, truncation_steps=25, **settings_kwargs
            ).truncation_tol
        )
        with self.assertRaises(ValueError):
            _SolverSettings.resolve(
                residual_tol=1e-6, truncation_steps=0, **settings_kwargs
            )

    def test_solver_submodule_names_are_not_shadowed(self):
        """``solvers.<name>`` must mean the module, for every submodule.

        Re-exporting the solver functions from ``solvers/__init__.py`` would
        shadow the same-named submodules, which silently changes what
        ``from ttnumpy.solvers import amen`` returns and makes dotted
        ``mock.patch`` targets resolve differently across Python versions.
        """
        for name in ("als", "amen", "mals", "common", "one_site", "two_site"):
            self.assertIsInstance(
                getattr(solvers, name), types.ModuleType, f"solvers.{name}"
            )
        # and the functions stay callable on the documented path
        for name in ("als", "amen", "mals"):
            self.assertTrue(callable(getattr(ttnumpy, name)), f"ttnumpy.{name}")

    def test_solvers_declare_their_truncations_per_sweep(self):
        """Each solver's declared count must match what a sweep really does.

        MALS decimates every neighbouring pair, AMEn truncates every solved
        core; both forward and backward. A count that drifts from the sweep
        silently rescales the budget, so it is pinned against a live solve.
        """
        order = 6
        laplacian_tt, rhs_tt, initial_tt = qtt_laplacian_1d(order)
        resolve = _SolverSettings.resolve
        declared = []

        def capturing_resolve(**kwargs):
            settings = resolve(**kwargs)
            declared.append(settings.truncation_steps)
            return settings

        for solver, module in ((mals, two_site), (amen, amen_module)):
            declared.clear()
            with (
                mock.patch.object(
                    module,
                    "_choose_truncation_rank",
                    side_effect=_choose_truncation_rank,
                ) as counter,
                mock.patch.object(_SolverSettings, "resolve", capturing_resolve),
            ):
                solver(
                    laplacian_tt,
                    rhs_tt,
                    x0=initial_tt,
                    max_sweeps=1,
                    residual_tol=1e-8,
                    stagnation_factor=None,
                    symmetric=True,
                )
            self.assertEqual(
                declared,
                [counter.call_count],
                f"{solver.__name__} declares a count its sweep does not perform",
            )

    def test_resolve_dispatch_contract(self):
        """The resolved dispatch triple, for every combination that matters.

        This is the one test that pins the local-solver redesign: which method
        runs, what tolerance it is held to, and what happens when it misses.
        Everything the sweeps do downstream reads exactly these three fields.
        """

        base = dict(
            max_sweeps=1,
            stagnation_factor=None,
            symmetric=True,
            direct_solve_size=1000,
            return_info=False,
        )
        minres, gmres = LocalSolver.MINRES, LocalSolver.GMRES
        # options -> (iterative_solver, local_rtol, allow_dense_fallback)
        cases = [
            # local_solver alone decides the method; "auto" reads `symmetric`
            (dict(local_solver="auto"), (minres, _DEFAULT_LOCAL_RTOL, False)),
            (
                dict(local_solver="auto", symmetric=False),
                (gmres, _DEFAULT_LOCAL_RTOL, False),
            ),
            (dict(local_solver="minres"), (minres, _DEFAULT_LOCAL_RTOL, False)),
            (dict(local_solver="gmres"), (gmres, _DEFAULT_LOCAL_RTOL, False)),
            # "direct" is the only value that disables the iterative path, and
            # then there is no tolerance to hold it to
            (dict(local_solver="direct"), (None, None, False)),
            (
                dict(local_solver="direct", residual_tol=1e-8, local_rtol=1e-9),
                (None, None, False),
            ),
            # residual_tol supplies a local tolerance but does not gate the path
            (dict(local_solver="auto", residual_tol=1e-8), (minres, 1e-9, False)),
            # an explicit local_rtol wins over the derivation, with or without
            # a residual_tol to derive from
            (dict(local_solver="auto", local_rtol=1e-6), (minres, 1e-6, False)),
            (
                dict(local_solver="auto", residual_tol=1e-8, local_rtol=1e-12),
                (minres, 1e-12, False),
            ),
            # the fallback flag is passed through untouched, whatever the
            # method: naming a Krylov solver no longer implies anything
            (
                dict(local_solver="auto", allow_dense_fallback=True),
                (minres, _DEFAULT_LOCAL_RTOL, True),
            ),
            (
                dict(local_solver="gmres", allow_dense_fallback=True),
                (gmres, _DEFAULT_LOCAL_RTOL, True),
            ),
            (
                dict(
                    local_solver="minres", residual_tol=1e-8, allow_dense_fallback=True
                ),
                (minres, 1e-9, True),
            ),
        ]
        for options, expected in cases:
            with self.subTest(**options):
                settings = _SolverSettings.resolve(
                    **{**base, "residual_tol": None, **options}
                )
                self.assertEqual(
                    (
                        settings.iterative_solver,
                        settings.local_rtol,
                        settings.allow_dense_fallback,
                    ),
                    expected,
                )

        # the invariant the dispatch relies on: a tolerance exists exactly when
        # there is a method to hold to it
        for options, _ in cases:
            settings = _SolverSettings.resolve(
                **{**base, "residual_tol": None, **options}
            )
            self.assertEqual(
                settings.local_rtol is None, settings.iterative_solver is None
            )

    def test_use_iterative_crossover(self):
        """The size half of the dispatch rule, asserted without a solve."""

        base = dict(
            max_sweeps=1,
            residual_tol=None,
            stagnation_factor=None,
            symmetric=True,
            return_info=False,
        )
        settings = _SolverSettings.resolve(
            local_solver="auto", direct_solve_size=100, **base
        )
        # "at most this many rows are solved directly", so the crossover is
        # strict: a system of exactly direct_solve_size still goes to LU
        self.assertFalse(settings.use_iterative(99))
        self.assertFalse(settings.use_iterative(100))
        self.assertTrue(settings.use_iterative(101))

        # 0 forces every local system onto the Krylov path
        forced = _SolverSettings.resolve(
            local_solver="auto", direct_solve_size=0, **base
        )
        self.assertTrue(forced.use_iterative(1))

        # "direct" has no method, so no size can reach the iterative path
        direct = _SolverSettings.resolve(
            local_solver="direct", direct_solve_size=0, **base
        )
        self.assertFalse(direct.use_iterative(10**6))

    def test_dense_fallback_is_opt_in(self):
        """allow_dense_fallback decides how a missed local solve is resolved.

        A local tolerance far below what MINRES can deliver on this operator
        forces the miss; the flag then chooses between keeping the inexact
        iterate and rebuilding the explicit matrix.
        """

        order = 6
        laplacian_tt, rhs_tt, initial_tt = qtt_laplacian_1d(order)
        shared = dict(
            x0=initial_tt,
            max_sweeps=1,
            residual_tol=1e-10,
            local_rtol=1e-14,  # unreachable, so every local solve misses
            symmetric=True,
            direct_solve_size=0,
            stagnation_factor=None,
            return_info=True,
        )

        _, without = mals(laplacian_tt, rhs_tt, allow_dense_fallback=False, **shared)
        self.assertGreater(without.local_inexact, 0)
        self.assertEqual(without.local_fallbacks, 0)

        _, with_fallback = mals(
            laplacian_tt, rhs_tt, allow_dense_fallback=True, **shared
        )
        self.assertGreater(with_fallback.local_fallbacks, 0)
        self.assertEqual(with_fallback.local_inexact, 0)

        # the two are disjoint subsets of the iterative solves; the remainder
        # met their tolerance outright and are counted only in local_solves
        for info in (without, with_fallback):
            self.assertLessEqual(
                info.local_fallbacks + info.local_inexact, info.local_solves
            )

    def test_loose_local_rtol_warns(self):
        """A local_rtol above the truncation budget must not pass silently."""

        settings_kwargs = dict(
            max_sweeps=1,
            residual_tol=1e-8,
            stagnation_factor=None,
            local_solver="auto",
            symmetric=True,
            direct_solve_size=1000,
            return_info=False,
        )
        with self.assertLogs(PACKAGE_LOGGER_NAME, level="WARNING") as captured:
            _SolverSettings.resolve(local_rtol=1e-3, **settings_kwargs)
        self.assertIn("local_rtol", captured.output[0])
        self.assertIn("residual_tol", captured.output[0])

        # the derived tolerance is a tenth of residual_tol, so it never warns
        with self.assertNoLogs(PACKAGE_LOGGER_NAME, level="WARNING"):
            _SolverSettings.resolve(local_rtol=None, **settings_kwargs)

    def test_local_solve_warnings_name_the_right_parameter(self):
        """The sweep warnings must point at the knob that caused them.

        The accepted-inexact message used to blame an explicit ``local_solver``,
        which is no longer what disables the fallback; the fallback message
        should recommend the cheaper option once nearly every solve rebuilds
        the explicit matrix.
        """

        order = 6
        laplacian_tt, rhs_tt, initial_tt = qtt_laplacian_1d(order)
        shared = dict(
            x0=initial_tt,
            max_sweeps=1,
            residual_tol=1e-10,
            local_rtol=1e-14,
            symmetric=True,
            direct_solve_size=0,
            stagnation_factor=None,
        )

        with self.assertLogs(PACKAGE_LOGGER_NAME, level="WARNING") as captured:
            mals(laplacian_tt, rhs_tt, allow_dense_fallback=False, **shared)
        inexact = [line for line in captured.output if "inexact" in line]
        self.assertTrue(inexact)
        self.assertIn("allow_dense_fallback", inexact[0])
        self.assertNotIn("explicit local_solver", inexact[0])

        with self.assertLogs(PACKAGE_LOGGER_NAME, level="WARNING") as captured:
            mals(laplacian_tt, rhs_tt, allow_dense_fallback=True, **shared)
        rebuilt = [line for line in captured.output if "explicit matrix" in line]
        self.assertTrue(rebuilt)
        self.assertIn("direct", rebuilt[0])


def test_solver_solves_the_same_system(solver, laplacian_system):
    """Every solver must take the same Ax = b to the same target."""

    x, info = solver(
        laplacian_system.A,
        laplacian_system.b,
        x0=laplacian_system.x0,
        residual_tol=1e-8,
        max_sweeps=10,
        symmetric=True,
        return_info=True,
    )
    assert info.converged, f"{solver.name} stopped: {info.stop_reason}"
    assert info.residuals[-1] <= 1e-8

    solution = np.asarray(x.get_full_tensor()).reshape(-1)
    exact = laplacian_system.exact
    assert np.linalg.norm(solution - exact) / np.linalg.norm(exact) < 1e-10


def test_solver_iterative_path_matches_direct(solver, laplacian_system):
    """Forcing the Krylov path must not change the answer.

    ``direct_solve_size=0`` puts every local system on MINRES, so this is the
    matrix-free path against the explicit LU one on identical input.
    """

    shared = dict(
        x0=laplacian_system.x0,
        residual_tol=1e-8,
        max_sweeps=10,
        symmetric=True,
    )
    direct = solver(
        laplacian_system.A, laplacian_system.b, local_solver="direct", **shared
    )
    iterative, info = solver(
        laplacian_system.A,
        laplacian_system.b,
        local_solver="minres",
        direct_solve_size=0,
        return_info=True,
        **shared,
    )
    # the run has to have taken the path under test
    assert info.local_solves > 0
    assert info.local_fallbacks == 0

    reference = np.asarray(direct.get_full_tensor()).reshape(-1)
    candidate = np.asarray(iterative.get_full_tensor()).reshape(-1)
    assert np.linalg.norm(candidate - reference) / np.linalg.norm(reference) < 1e-10


def test_solver_threshold_mode_iterative_matches_direct(solver, laplacian_system):
    """The same, without ``residual_tol``.

    This combination is unreachable before the dispatch redesign: without
    ``residual_tol`` every local system was solved directly whatever its size.
    """

    shared = dict(
        x0=laplacian_system.x0,
        residual_tol=None,
        max_sweeps=3,
        symmetric=True,
        stagnation_factor=None,
    )
    direct = solver(
        laplacian_system.A, laplacian_system.b, local_solver="direct", **shared
    )
    iterative, info = solver(
        laplacian_system.A,
        laplacian_system.b,
        local_solver="minres",
        direct_solve_size=0,
        local_rtol=1e-10,
        return_info=True,
        **shared,
    )
    assert info.local_solves > 0
    assert info.local_fallbacks == 0

    reference = np.asarray(direct.get_full_tensor()).reshape(-1)
    candidate = np.asarray(iterative.get_full_tensor()).reshape(-1)
    assert np.linalg.norm(candidate - reference) / np.linalg.norm(reference) < 1e-10


# --------------------------Contracts solver shares----------------------------------


_MALFORMED_OPERANDS = {
    "rhs_physical_dims": (
        lambda A, b: ((A, tensor_train.ones([3] * 3)), {}),
        tensor_train.DimensionMismatch,
    ),
    "rhs_core_count": (
        lambda A, b: ((A, tensor_train.ones([2] * 4)), {}),
        tensor_train.DimensionMismatch,
    ),
    "x0_physical_dims": (
        lambda A, b: ((A, b), {"x0": tensor_train.ones([3] * 3)}),
        tensor_train.DimensionMismatch,
    ),
    "operator_not_square": (
        lambda A, b: ((tensor_train.rand([(2, 3)] * 3, ranks=2, seed=0), b), {}),
        tensor_train.DimensionMismatch,
    ),
    "operator_is_a_tt": (lambda A, b: ((b, b), {}), tensor_train.WrongTTType),
    "rhs_is_a_ttm": (lambda A, b: ((A, A), {}), tensor_train.WrongTTType),
    "x0_is_a_ttm": (lambda A, b: ((A, b), {"x0": A}), tensor_train.WrongTTType),
}


@pytest.mark.parametrize("case", sorted(_MALFORMED_OPERANDS))
def test_solver_rejects_malformed_operands(solver, case):
    """Structurally invalid operands must fail fast, identically everywhere."""

    A, b, _ = qtt_laplacian_1d(3)
    build, expected = _MALFORMED_OPERANDS[case]
    args, kwargs = build(A, b)

    with pytest.raises(expected):
        solver(*args, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"stagnation_factor": "fast"}, id="stagnation_factor_string"),
        pytest.param({"stagnation_factor": 1.5}, id="stagnation_factor_above_one"),
        pytest.param({"stagnation_factor": -0.5}, id="stagnation_factor_negative"),
        pytest.param({"local_solver": "cg"}, id="local_solver_unknown"),
        pytest.param({"residual_tol": 0.0}, id="residual_tol_zero"),
        pytest.param({"residual_tol": -1.0}, id="residual_tol_negative"),
        pytest.param({"local_rtol": 0.0}, id="local_rtol_zero"),
        pytest.param({"local_rtol": -1e-3}, id="local_rtol_negative"),
    ],
)
def test_solver_rejects_invalid_arguments(solver, kwargs):
    """Out-of-range tuning parameters must raise before any work is done."""

    A, b, _ = qtt_laplacian_1d(3)
    with pytest.raises(ValueError):
        solver(A, b, **kwargs)


def test_solver_inplace_writes_the_solution_into_the_guess(solver):
    """``inplace=True`` must return the very object it was handed."""

    A, b, _ = qtt_laplacian_1d(3)
    # deliberately not orthonormalized: the solver has to canonicalize in place
    guess = tensor_train.rand(physical_indexes=[2] * 3, ranks=2, seed=1)
    assert guess.orth_center is None

    result = solver(
        A, b, x0=guess, inplace=True, max_sweeps=10, residual_tol=1e-8, symmetric=True
    )

    assert result is guess
    assert relative_residual(A, b, guess) <= 1e-8


def test_solver_without_inplace_leaves_the_guess_untouched(solver):
    """The default must not write through to the caller's train."""

    A, b, _ = qtt_laplacian_1d(3)
    guess = tensor_train.rand(physical_indexes=[2] * 3, ranks=2, seed=2)
    cores_before = [core.copy() for core in guess.cores]

    result = solver(A, b, x0=guess, max_sweeps=10, residual_tol=1e-8, symmetric=True)

    assert result is not guess
    for core_before, core in zip(cores_before, guess.cores):
        np.testing.assert_array_equal(core_before, core)


def test_solver_skips_the_residual_when_nothing_tracks_it(solver):
    """An untracked run must not pay for a per-sweep residual evaluation."""

    A, b, x0 = qtt_laplacian_1d(3)

    with mock.patch.object(
        solvers.common, "relative_residual", wraps=relative_residual
    ) as residual_spy:
        solver(A, b, x0=x0, max_sweeps=3, stagnation_factor=None)
        assert residual_spy.call_count == 0

        solver(A, b, x0=x0, max_sweeps=3, residual_tol=1e-8)
        assert residual_spy.call_count > 0


def test_solver_handles_a_zero_right_hand_side(solver):
    """``A x = 0`` must return zero rather than dividing by ``||b|| = 0``."""

    A, _, _ = qtt_laplacian_1d(4)
    zero_rhs = tensor_train.zeros([2] * 4)
    guess = tensor_train.ones([2] * 4, ranks=2).compress_svd(canonical="right")

    solution, info = solver(
        A, zero_rhs, x0=guess, max_sweeps=3, symmetric=True, return_info=True
    )

    assert np.linalg.norm(np.asarray(solution.get_full_tensor()).reshape(-1)) == 0.0
    # ||b|| == 0 makes relative_residual fall back to the absolute norm
    assert relative_residual(A, zero_rhs, solution) == 0.0
    assert info.stop_reason is not None


@pytest.mark.parametrize("rank_adaptive_solver", [mals, amen], ids=["mals", "amen"])
@pytest.mark.parametrize("max_rank", [0, -2])
def test_rank_adaptive_solver_rejects_non_positive_max_rank(
    rank_adaptive_solver, max_rank
):
    """A cap of zero or less has no valid interpretation."""

    A, b, _ = qtt_laplacian_1d(3)
    with pytest.raises(ValueError):
        rank_adaptive_solver(A, b, max_rank=max_rank)


@pytest.mark.parametrize(
    "enum_type",
    [StopReason, LocalSolver, TruncationMode],
    ids=["StopReason", "LocalSolver", "TruncationMode"],
)
def test_solver_enums_stringify_to_their_value(enum_type):
    """These are str-enums, so str() must give the bare value.

    SolverInfo.stop_reason reaches reports and log lines through str(), where
    the default Enum repr ("StopReason.CONVERGED") would leak the type name.
    """

    for member in enum_type:
        assert str(member) == member.value
        assert f"{member}" == member.value


def test_solver_rejects_a_ttm_right_hand_side(solver):
    """A TTM b must be named as such, not mistaken for a bad guess.

    The x0 check runs first and a TTM b makes the default guess a TTM too, so
    only an explicit TensorTrain guess reaches the b check at all.
    """

    A, _, x0 = qtt_laplacian_1d(3)
    with pytest.raises(tensor_train.WrongTTType, match="Right-hand side"):
        solver(A, A, x0=x0)


@pytest.mark.parametrize("rank_adaptive_solver", [mals, amen], ids=["mals", "amen"])
def test_rank_adaptive_solver_truncates_a_zero_local_solution(rank_adaptive_solver):
    """Residual truncation must survive an all-zero local solve.

    A zero right-hand side drives every local solution to zero, so the
    truncation rank has to be chosen from a singular-value spectrum whose
    leading entry is exactly 0 - the one input for which the residual rule
    has no ratio to work with.
    """

    A, _, _ = qtt_laplacian_1d(4)
    zero_rhs = tensor_train.zeros([2] * 4)
    guess = tensor_train.ones([2] * 4, ranks=2).compress_svd(canonical="right")

    solution, info = rank_adaptive_solver(
        A,
        zero_rhs,
        x0=guess,
        max_sweeps=2,
        residual_tol=1e-8,
        symmetric=True,
        return_info=True,
    )

    assert np.linalg.norm(np.asarray(solution.get_full_tensor()).reshape(-1)) == 0.0
    # a zero spectrum must still leave a structurally valid rank-1 train
    assert min(solution.bonds) >= 1
    assert info.stop_reason is not None


def test_solver_records_local_fallbacks_when_the_dense_path_is_armed(solver):
    """Every solver must reach and count the explicit-matrix fallback.

    A local tolerance below what MINRES can deliver on this operator forces
    the miss; ``allow_dense_fallback=True`` is what makes the explicit matrix
    reachable at all. ALS drives the one-site local solve and MALS/AMEn the
    two-site one, so the fallback needs exercising through each of them.
    """

    A, b, x0 = qtt_laplacian_1d(6)

    _, info = solver(
        A,
        b,
        x0=x0,
        max_sweeps=1,
        symmetric=True,
        residual_tol=1e-10,
        local_solver="minres",
        direct_solve_size=0,
        allow_dense_fallback=True,
        local_rtol=1e-14,  # unreachable, so every local solve misses
        stagnation_factor=None,
        return_info=True,
    )

    assert info.local_solves > 0
    assert info.local_fallbacks > 0
    # a solve that fell back is resolved, so it is not also counted as inexact
    assert info.local_inexact == 0
    assert info.local_fallbacks <= info.local_solves
