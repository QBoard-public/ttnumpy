# Copyright (c) 2026 QBoard LLC. Licensed under the AGPLv3. See LICENSE.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Benchmarks comparing the tensor-train solvers against dense baselines."""

from .poisson import (
    DEFAULT_BENCHMARK_NS,
    DEFAULT_TT_SETTINGS,
    TTSolverSettings,
    benchmark_petsc_poisson,
    benchmark_scipy_poisson,
    benchmark_tt_poisson,
    compare_poisson_solvers,
    main as run_poisson_benchmark,
    print_poisson_benchmark_suite_table,
    write_poisson_benchmark_report,
)

__all__ = [
    "DEFAULT_BENCHMARK_NS",
    "DEFAULT_TT_SETTINGS",
    "TTSolverSettings",
    "benchmark_tt_poisson",
    "benchmark_scipy_poisson",
    "benchmark_petsc_poisson",
    "compare_poisson_solvers",
    "print_poisson_benchmark_suite_table",
    "write_poisson_benchmark_report",
    "run_poisson_benchmark",
]
