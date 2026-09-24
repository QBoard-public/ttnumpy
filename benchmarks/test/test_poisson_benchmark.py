# Copyright (c) 2026 QBoard LLC. Licensed under the AGPLv3. See LICENSE.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the Poisson benchmark runner: CLI, reporting and control flow.

Two layers are tested differently.

1) The reporting layer is driven through the public entry points: ``main``,
``write_poisson_benchmark_report`` and ``print_poisson_benchmark_suite_table``.
``compare_poisson_solvers`` is replaced by a fake bound to its real signature,
and no solver runs.

2) The execution layer runs for real at ``n=3``, an 8x8 grid on which every
backend completes in milliseconds. ``pyamg`` and ``petsc4py`` are optional
dependencies. CI installs ``.[dev]``, which omits both.
"""

import dataclasses
import json
import signal
import time
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from benchmarks import poisson as poisson_benchmark
from benchmarks.poisson_problem import build_poisson_problem

_BACKEND_LABELS = [
    "tt-mals",
    "tt-amen",
    "scipy-cg",
    "scipy-cg+pyamg",
    "petsc-cg+gamg",
]

_LIVE_N = 3


def _patch_compare(monkeypatch, fake=None):
    """Replace compare_poisson_solvers with a fake bound to its real signature.

    ``create_autospec`` raises TypeError on a call whose parameter names do not
    match the real function. A change to that signature fails these tests
    rather than passing unnoticed.
    """
    spec = mock.create_autospec(
        poisson_benchmark.compare_poisson_solvers,
        side_effect=fake or _fake_comparison,
    )
    monkeypatch.setattr(poisson_benchmark, "compare_poisson_solvers", spec)
    return spec


def _fake_comparison(**kwargs):
    """A ``compare_poisson_solvers`` result with fixed numbers.

    ``petsc`` is None, the value produced on a machine without petsc4py.
    """
    n = kwargs["n"]
    settings = kwargs["tt_settings"]
    shared = {
        "elapsed_s": 1.0 + n,
        "error_vs_exact": 1e-3,
        "relative_residual": 2e-3,
        "status": "converged",
    }
    return {
        "n": n,
        "grid_size": 2**n,
        "unknowns": 4**n,
        "mode_size": settings.mode_size,
        "tolerance": settings.tolerance,
        "mals": {**shared, "sweeps": 2, "max_bond": 7},
        "amen": {**shared, "sweeps": 3, "max_bond": 9},
        "scipy": {**shared, "iterations": 11},
        "pyamg": {**shared, "iterations": 7},
        "petsc": None,
        "mals_vs_amen": 1e-9,
        "mals_vs_scipy": 2e-9,
        "mals_vs_pyamg": 3e-9,
    }


def _printed_table(tmp_path, monkeypatch, capsys, *sizes):
    """Run main over the given sizes and return the console table it printed."""
    _patch_compare(monkeypatch)
    poisson_benchmark.main(
        ["--sizes", *[str(n) for n in sizes], "--output-dir", str(tmp_path)]
    )
    lines = capsys.readouterr().out.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("n  grid"))
    # the table ends at the first blank line; main prints its file paths after
    table = lines[start:]
    end = next((i for i, line in enumerate(table) if not line.strip()), len(table))
    return table[:end]


def test_poisson_benchmark_main_writes_report(tmp_path: Path, monkeypatch):
    _patch_compare(monkeypatch)

    exit_code = poisson_benchmark.main(
        ["--sizes", "4", "5", "--tolerance", "1e-5", "--output-dir", str(tmp_path)]
    )

    assert exit_code == 0
    # the report name carries the parameters so separate runs do not collide
    json_path = tmp_path / "poisson_benchmark_mode2_tol1e-05.json"
    markdown_path = tmp_path / "poisson_benchmark_mode2_tol1e-05.md"
    assert json_path.exists()
    assert markdown_path.exists()

    markdown = markdown_path.read_text()
    assert "Poisson benchmark results" in markdown
    # one column per backend, every metric stacked inside the cell
    header = next(line for line in markdown.splitlines() if line.startswith("| n |"))
    assert header.count("|") == 8  # n, grid and backends
    for label in _BACKEND_LABELS:
        assert label in header
    assert "real time (s): 5.000e+00<br>err vs exact: 1.000e-03" in markdown

    report = json.loads(json_path.read_text())
    assert report["ns"] == [4, 5]
    assert report["tolerance"] == 1e-5
    assert report["mode_size"] == 2
    assert report["timeout_s"] == poisson_benchmark.TIMEOUT
    assert len(report["rows"]) == 10
    assert {row["backend"] for row in report["rows"]} == set(_BACKEND_LABELS)
    unavailable = [row for row in report["rows"] if row["backend"] == "petsc-cg+gamg"]
    assert all(row["status"] == "not available" for row in unavailable)
    assert all(row["elapsed_s"] is None for row in unavailable)


def test_main_reports_solver_differences_on_request(
    tmp_path: Path, monkeypatch, capsys
):
    _patch_compare(monkeypatch)

    poisson_benchmark.main(
        ["--sizes", "4", "--output-dir", str(tmp_path), "--show-solver-differences"]
    )

    out = capsys.readouterr().out
    assert "mals vs amen: 1.000e-09" in out
    assert "mals vs scipy: 2.000e-09" in out
    assert "mals vs pyamg: 3.000e-09" in out
    # petsc produced no result; its row is absent
    assert "mals vs petsc" not in out


def test_printed_table_puts_every_backend_in_its_own_column(
    tmp_path, monkeypatch, capsys
):
    """Each backend occupies one column, with metrics running down the rows."""

    lines = _printed_table(tmp_path, monkeypatch, capsys, 4)

    assert lines[0].split() == ["n", "grid", *_BACKEND_LABELS]
    metric_labels = [label for label, _ in poisson_benchmark._CELL_METRICS]
    for label, line in zip(metric_labels, lines[2:]):
        assert label in line
    # the grid size is stated once per n, on the first metric row
    assert lines[2].startswith("4  16x16")
    assert lines[3].startswith("     ")


def test_printed_table_separates_grid_sizes(tmp_path, monkeypatch, capsys):
    """Each grid size occupies its own block, separated by a rule."""

    lines = _printed_table(tmp_path, monkeypatch, capsys, 4, 5)

    metrics = len(poisson_benchmark._CELL_METRICS)
    # header, rule, metrics for n=4, rule, metrics for n=5
    assert len(lines) == 2 + metrics * 2 + 1
    assert lines[2].startswith("4  16x16")
    assert lines[2 + metrics + 1].startswith("5  32x32")


def _written_rows(tmp_path, monkeypatch, comparison):
    """Run main with a fixed comparison and read the rows back out of the JSON."""
    _patch_compare(monkeypatch, lambda **kwargs: comparison)
    poisson_benchmark.main(["--sizes", "4", "--output-dir", str(tmp_path)])
    report = json.loads(next(tmp_path.glob("*.json")).read_text())
    return {row["backend"]: row for row in report["rows"]}


def test_report_marks_missing_backends_as_unavailable(tmp_path, monkeypatch):
    """A backend that produced nothing keeps its column.

    Every grid size renders the same five columns. A reader compares rows
    down a fixed set of headings.
    """

    comparison = _fake_comparison(
        n=4, tt_settings=poisson_benchmark.DEFAULT_TT_SETTINGS
    )
    comparison["pyamg"] = None

    by_backend = _written_rows(tmp_path, monkeypatch, comparison)

    assert by_backend["scipy-cg+pyamg"]["status"] == "not available"
    assert by_backend["scipy-cg+pyamg"]["relative_residual"] is None
    assert list(by_backend) == _BACKEND_LABELS
    assert all(row["grid_shape"] == "16x16" for row in by_backend.values())
    assert all(row["unknowns"] == 4**4 for row in by_backend.values())


def test_report_notes_carry_sweeps_bonds_and_iterations(tmp_path, monkeypatch):
    """The notes column records sweeps and bond dimension, or iterations."""

    by_backend = _written_rows(
        tmp_path,
        monkeypatch,
        _fake_comparison(n=4, tt_settings=poisson_benchmark.DEFAULT_TT_SETTINGS),
    )

    assert by_backend["tt-mals"]["notes"] == "sweeps=2 bond=7"
    assert by_backend["tt-amen"]["notes"] == "sweeps=3 bond=9"
    assert by_backend["scipy-cg"]["notes"] == "iters=11"
    assert by_backend["scipy-cg+pyamg"]["notes"] == "iters=7"
    # a backend that never ran has no note at all
    assert by_backend["petsc-cg+gamg"]["notes"] == ""


def test_console_table_renders_an_empty_report():
    """A report with no rows renders its header.

    ``print_poisson_benchmark_suite_table`` is exported, and a caller may pass
    a report in which no backend produced a row. Column widths are then taken
    from an empty set of rows.
    """
    table = poisson_benchmark._render_poisson_benchmark_rows([])

    lines = table.splitlines()
    assert lines[0].split() == ["n", "grid", *_BACKEND_LABELS]


def test_suite_table_prints_an_empty_report(capsys):
    """The public entry point accepts the same input."""
    poisson_benchmark.print_poisson_benchmark_suite_table(
        {"ns": [], "mode_size": 2, "tolerance": 1e-5, "rows": []}
    )

    assert "Poisson benchmark suite" in capsys.readouterr().out


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("exceeded the 600s benchmark time limit"),
        MemoryError(),
        ValueError("rhs must be a scalar, a callable f(x, y), or an array"),
        RuntimeError("local solve did not converge"),
        np.linalg.LinAlgError("matrix is singular to machine precision"),
    ],
    ids=["timeout", "memory", "value", "runtime", "linalg"],
)
def test_run_backend_turns_an_expected_failure_into_a_row(error):
    """A backend that fails in a foreseeable way must not abort the suite.

    A suite covers several grid sizes and runs for hours. Each of these
    failures is recorded as a status row, and the remaining backends run.
    """

    def failing():
        raise error

    row = poisson_benchmark._run_backend("probe", failing)

    assert row["backend"] == "probe"
    assert row["elapsed_s"] >= 0.0
    if isinstance(error, TimeoutError):
        assert row["status"].startswith("timeout>")
    else:
        assert row["status"] == f"failed: {type(error).__name__}"


@pytest.mark.parametrize(
    "error",
    [
        TypeError("problem must be a PoissonProblem"),
        KeyError("elapsed_s"),
        AttributeError("'NoneType' object has no attribute 'laplacian'"),
    ],
    ids=["type", "key", "attribute"],
)
def test_run_backend_lets_an_unexpected_failure_propagate(error):
    """An exception outside the expected set propagates.

    The caught set is deliberately narrow. Widening it would record a defect
    as a "failed" row instead of surfacing it. Any addition to the set is a
    decision to be reviewed.
    """

    def failing():
        raise error

    with pytest.raises(type(error)):
        poisson_benchmark._run_backend("probe", failing)


def test_run_backend_returns_the_result_untouched_on_success():
    """A successful backend's own result object reaches the report."""

    expected = {"backend": "probe", "elapsed_s": 1.5, "status": "converged"}
    assert poisson_benchmark._run_backend("probe", lambda: expected) is expected


def test_time_limit_raises_and_restores_the_previous_handler():
    """The alarm fires, and the previous SIGALRM handler is restored."""

    previous = signal.getsignal(signal.SIGALRM)

    with pytest.raises(TimeoutError):
        with poisson_benchmark._time_limit(0.05):
            deadline = time.perf_counter() + 5.0
            while time.perf_counter() < deadline:
                pass

    assert signal.getsignal(signal.SIGALRM) is previous
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_time_limit_disarms_the_timer_when_the_block_finishes_early():
    """A block that finishes early leaves no armed timer behind."""

    with poisson_benchmark._time_limit(30.0):
        pass

    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_pyamg_column_is_unavailable_when_the_run_fell_back_to_plain_cg(
    tmp_path, monkeypatch
):
    """A column reports only numbers the named backend produced.

    ``benchmark_scipy_poisson(preconditioned=True)`` falls back to plain CG
    when pyamg is absent and reports ``backend="scipy-cg"``. An install
    without pyamg therefore places a plain CG result in the pyamg slot. The
    column is marked unavailable, as the petsc column already is; otherwise
    the table presents unpreconditioned timings under a preconditioned
    heading.
    """

    comparison = _fake_comparison(
        n=4, tt_settings=poisson_benchmark.DEFAULT_TT_SETTINGS
    )
    # the value the fallback returns: a plain CG result in the pyamg slot
    comparison["pyamg"] = {**comparison["scipy"], "backend": "scipy-cg"}

    by_backend = _written_rows(tmp_path, monkeypatch, comparison)

    assert by_backend["scipy-cg+pyamg"]["status"] == "not available"
    assert by_backend["scipy-cg+pyamg"]["elapsed_s"] is None
    # the plain CG column is unchanged
    assert by_backend["scipy-cg"]["status"] == "converged"


def test_backend_rows_keep_results_that_match_their_column(tmp_path, monkeypatch):
    """A result whose backend matches its column is reported unchanged."""

    comparison = _fake_comparison(
        n=4, tt_settings=poisson_benchmark.DEFAULT_TT_SETTINGS
    )
    comparison["pyamg"] = {**comparison["pyamg"], "backend": "scipy-cg+pyamg"}

    by_backend = _written_rows(tmp_path, monkeypatch, comparison)

    assert by_backend["scipy-cg+pyamg"]["status"] == "converged"
    assert by_backend["scipy-cg+pyamg"]["elapsed_s"] is not None


def _report(**settings_overrides):
    settings = dataclasses.replace(
        poisson_benchmark.DEFAULT_TT_SETTINGS, **settings_overrides
    )
    return {
        "ns": [4],
        "mode_size": settings.mode_size,
        "tolerance": settings.tolerance,
        "tt_settings": settings,
        "runs": [],
        "rows": [],
    }


def test_default_run_keeps_its_established_report_name():
    """A default run keeps the name the CI job publishes."""

    assert (
        poisson_benchmark._report_stem(_report()) == "poisson_benchmark_mode2_tol1e-05"
    )


@pytest.mark.parametrize(
    ("overrides", "expected_suffix"),
    [
        ({"max_rank": 20}, "_rank20"),
        ({"max_sweeps": 50}, "_sweeps50"),
        ({"kickrank": 8}, "_kick8"),
        ({"seed": 3}, "_seed3"),
    ],
)
def test_a_non_default_parameter_appears_in_the_report_name(overrides, expected_suffix):
    """A parameter that changes the numbers changes the file name."""

    stem = poisson_benchmark._report_stem(_report(**overrides))

    assert stem == "poisson_benchmark_mode2_tol1e-05" + expected_suffix


def test_runs_differing_in_any_parameter_get_distinct_names():
    """Each of these parameter sets produces a distinct file name."""

    stems = {
        poisson_benchmark._report_stem(_report(**overrides))
        for overrides in (
            {},
            {"max_rank": 20},
            {"max_sweeps": 50},
            {"kickrank": 8},
            {"seed": 3},
            {"tolerance": 1e-8},
            {"mode_size": 4},
            {"max_rank": 20, "seed": 3},
        )
    }

    assert len(stems) == 8


def test_report_without_settings_still_names_a_file():
    """A report carrying no settings is still named.

    ``write_poisson_benchmark_report`` is exported and accepts reports written
    before ``tt_settings`` was recorded on them.
    """

    stem = poisson_benchmark._report_stem(
        {"ns": [4], "mode_size": 2, "tolerance": 1e-5, "rows": []}
    )

    assert stem == "poisson_benchmark_mode2_tol1e-05"


@pytest.mark.parametrize(
    ("argv", "field", "expected"),
    [
        (["--max-rank", "20"], "max_rank", 20),
        (["--max-sweeps", "7"], "max_sweeps", 7),
        (["--kickrank", "9"], "kickrank", 9),
        (["--seed", "5"], "seed", 5),
        (["--mode-size", "4"], "mode_size", 4),
        (["--tolerance", "1e-9"], "tolerance", 1e-9),
    ],
)
def test_cli_flag_reaches_the_solver_settings(
    tmp_path, monkeypatch, argv, field, expected
):
    spec = _patch_compare(monkeypatch)

    poisson_benchmark.main(["--sizes", "4", "--output-dir", str(tmp_path), *argv])

    settings = spec.call_args.kwargs["tt_settings"]
    assert getattr(settings, field) == expected


def test_cli_defaults_match_the_published_defaults(tmp_path, monkeypatch):
    """With no flags, the run uses the documented default settings."""

    spec = _patch_compare(monkeypatch)

    poisson_benchmark.main(["--sizes", "4", "--output-dir", str(tmp_path)])

    assert spec.call_args.kwargs["tt_settings"] == poisson_benchmark.DEFAULT_TT_SETTINGS
    assert spec.call_args.kwargs["compute_reference"] is True


def test_no_reference_flag_turns_off_the_direct_solve(tmp_path, monkeypatch):
    """--no-reference skips the sparse direct solve.

    On large grids ``spsolve`` dominates the run time of the suite.
    """

    spec = _patch_compare(monkeypatch)

    poisson_benchmark.main(
        ["--sizes", "4", "--output-dir", str(tmp_path), "--no-reference"]
    )

    assert spec.call_args.kwargs["compute_reference"] is False


def test_single_n_flag_runs_one_size(tmp_path, monkeypatch):
    """-n runs the suite at a single grid size."""

    spec = _patch_compare(monkeypatch)

    poisson_benchmark.main(["-n", "6", "--output-dir", str(tmp_path)])

    assert [call.kwargs["n"] for call in spec.call_args_list] == [6]


def test_default_sizes_are_used_when_neither_flag_is_given(tmp_path, monkeypatch):
    spec = _patch_compare(monkeypatch)

    poisson_benchmark.main(["--output-dir", str(tmp_path)])

    assert [call.kwargs["n"] for call in spec.call_args_list] == list(
        poisson_benchmark.DEFAULT_BENCHMARK_NS
    )


def test_non_default_flags_change_the_report_file(tmp_path, monkeypatch):
    """The naming rule holds end to end, not only in _report_stem."""

    _patch_compare(monkeypatch)

    poisson_benchmark.main(["--sizes", "4", "--output-dir", str(tmp_path)])
    poisson_benchmark.main(
        ["--sizes", "4", "--output-dir", str(tmp_path), "--max-rank", "20"]
    )

    written = sorted(path.name for path in tmp_path.glob("*.json"))
    assert written == [
        "poisson_benchmark_mode2_tol1e-05.json",
        "poisson_benchmark_mode2_tol1e-05_rank20.json",
    ]


def _live_problem(**kwargs):
    return build_poisson_problem(N=_LIVE_N, tt_format=True, **kwargs)


@pytest.mark.parametrize("solver", ["mals", "amen"])
def test_tt_backend_solves_the_problem_and_reports_its_run(solver):
    """The TT backends return a solution and the metrics describing the run."""

    problem = _live_problem()

    result = poisson_benchmark.benchmark_tt_poisson(
        n=_LIVE_N,
        problem=problem,
        tt_settings=poisson_benchmark.TTSolverSettings(tolerance=1e-8, max_sweeps=10),
        solver=solver,
    )

    assert result["backend"] == f"tt-{solver}"
    assert result["n"] == _LIVE_N
    assert result["elapsed_s"] > 0.0
    assert result["sweeps"] >= 1
    assert result["max_bond"] >= 1
    # accuracy is measured against the sparse direct solution, not against the
    # solver's own residual
    assert result["error_vs_exact"] < 1e-6
    assert result["relative_residual"] < 1e-6
    assert result["solution"].shape == (problem.K**2,)


def test_tt_backend_builds_its_own_problem_when_given_none():
    """With no problem supplied, the backend assembles one in TT format."""

    result = poisson_benchmark.benchmark_tt_poisson(
        n=_LIVE_N,
        tt_settings=poisson_benchmark.TTSolverSettings(tolerance=1e-8, max_sweeps=10),
    )

    assert result["backend"] == "tt-mals"
    assert result["error_vs_exact"] < 1e-6


def test_scipy_backend_solves_the_same_problem():
    problem = _live_problem()

    result = poisson_benchmark.benchmark_scipy_poisson(
        n=_LIVE_N, problem=problem, tolerance=1e-10
    )

    assert result["backend"] == "scipy-cg"
    assert result["status"] == "converged"
    assert result["iterations"] > 0
    assert result["error_vs_exact"] < 1e-6


def test_preconditioned_scipy_backend_names_the_path_it_took():
    """The reported label names the path taken.

    With pyamg installed the label is ``scipy-cg+pyamg``; without it, the run
    falls back and the label is ``scipy-cg``. The report keys its columns on
    this label. CI runs without pyamg.
    """

    problem = _live_problem()

    result = poisson_benchmark.benchmark_scipy_poisson(
        n=_LIVE_N, problem=problem, preconditioned=True, tolerance=1e-10
    )

    assert result["backend"] in {"scipy-cg", "scipy-cg+pyamg"}
    assert result["status"] == "converged"
    assert result["error_vs_exact"] < 1e-6


def test_petsc_backend_reports_absence_rather_than_raising(monkeypatch):
    """Without petsc4py the backend returns None rather than raising."""

    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("petsc4py"):
            raise ImportError("petsc4py is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)

    assert poisson_benchmark.benchmark_petsc_poisson(n=_LIVE_N) is None


def test_compare_runs_every_backend_and_cross_checks_them():
    """The suite entry point produces one comparable result per backend."""

    comparisons = poisson_benchmark.compare_poisson_solvers(
        n=_LIVE_N,
        tt_settings=poisson_benchmark.TTSolverSettings(tolerance=1e-8, max_sweeps=10),
    )

    assert comparisons["n"] == _LIVE_N
    assert comparisons["grid_size"] == 2**_LIVE_N
    assert comparisons["unknowns"] == 4**_LIVE_N

    # the always-available backends must all have solved the same system
    for key in ("mals", "amen", "scipy"):
        assert comparisons[key]["error_vs_exact"] < 1e-6, key
    # the cross-check reports how far the backends are from one another
    for key in ("amen", "scipy"):
        assert comparisons[f"mals_vs_{key}"] < 1e-6, key


def test_compare_without_a_reference_reports_no_error_column():
    """Without a reference solution the error column is None.

    ``compute_reference=False`` skips the sparse direct solve that
    ``error_vs_exact`` is measured against.
    """

    comparisons = poisson_benchmark.compare_poisson_solvers(
        n=_LIVE_N,
        tt_settings=poisson_benchmark.TTSolverSettings(tolerance=1e-8, max_sweeps=10),
        compute_reference=False,
    )

    for key in ("mals", "amen", "scipy"):
        assert comparisons[key]["error_vs_exact"] is None, key
        # the residual is measured against the right-hand side, not the reference
        assert comparisons[key]["relative_residual"] < 1e-6, key


def test_relative_difference_falls_back_to_the_absolute_norm():
    """A reference of norm zero leaves no denominator."""

    zero = np.zeros(4)
    candidate = np.array([3.0, 4.0, 0.0, 0.0])

    assert poisson_benchmark._relative_difference(zero, candidate) == 5.0
    assert poisson_benchmark._relative_difference(zero, zero) == 0.0


def test_relative_residual_falls_back_to_the_absolute_norm():
    """A right-hand side of norm zero leaves no denominator."""
    from scipy import sparse

    identity = sparse.eye(3, format="csr")
    solution = np.array([3.0, 4.0, 0.0])

    absolute = poisson_benchmark._relative_residual(identity, solution, np.zeros(3))
    assert absolute == 5.0


def test_scipy_backend_builds_its_own_problem_when_given_none():
    result = poisson_benchmark.benchmark_scipy_poisson(n=_LIVE_N, tolerance=1e-10)

    assert result["backend"] == "scipy-cg"
    assert result["error_vs_exact"] < 1e-6


def test_preconditioned_backend_falls_back_when_pyamg_is_missing(monkeypatch):
    """Without pyamg the run falls back to plain CG and reports that label."""

    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("pyamg"):
            raise ImportError("pyamg is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)

    result = poisson_benchmark.benchmark_scipy_poisson(
        n=_LIVE_N, preconditioned=True, tolerance=1e-10
    )

    # this label is what the report uses to mark the column unavailable
    assert result["backend"] == "scipy-cg"
    assert result["status"] == "converged"


def test_note_is_empty_for_a_result_that_never_ran():
    """A row from a timed-out backend carries neither sweeps nor iterations."""

    assert (
        poisson_benchmark._benchmark_note({"backend": "x", "status": "timeout"}) == ""
    )


def test_cross_check_skips_backends_that_produced_no_solution(tmp_path, monkeypatch):
    """A backend with no solution is skipped by the MALS cross-check."""

    comparison = _fake_comparison(
        n=4, tt_settings=poisson_benchmark.DEFAULT_TT_SETTINGS
    )
    # a timed-out backend: a row carrying a status and no solution
    comparison["scipy"] = {"backend": "scipy-cg", "elapsed_s": 1.0, "status": "timeout"}

    by_backend = _written_rows(tmp_path, monkeypatch, comparison)

    assert by_backend["scipy-cg"]["status"] == "timeout"
    assert by_backend["scipy-cg"]["notes"] == ""
    assert by_backend["tt-mals"]["status"] == "converged"


def test_markdown_marks_a_backend_with_no_row_as_unavailable(tmp_path):
    """A backend absent from the rows still gets a column in the Markdown.

    ``write_poisson_benchmark_report`` is exported and accepts a report whose
    rows cover only some of the backends.
    """

    shared = {
        "n": 4,
        "grid_size": 16,
        "grid_shape": "16x16",
        "unknowns": 256,
        "mode_size": 2,
        "tolerance": 1e-5,
    }
    report = {
        "ns": [4],
        "mode_size": 2,
        "tolerance": 1e-5,
        "rows": [
            {
                **shared,
                "backend": "tt-mals",
                "elapsed_s": 1.0,
                "error_vs_exact": 1e-3,
                "relative_residual": 2e-3,
                "max_bond": 7,
                "status": "converged",
                "notes": "sweeps=2",
            }
        ],
    }

    paths = poisson_benchmark.write_poisson_benchmark_report(report, tmp_path)
    markdown = paths["markdown"].read_text()

    # every backend keeps a column even though only one produced a row
    for label in _BACKEND_LABELS:
        assert label in markdown
    assert "n/a" in markdown


def test_compare_runs_the_way_ci_does_without_the_optional_backends(monkeypatch):
    """The suite runs in CI's environment, without the optional backends.

    CI installs ``.[dev]``, which omits pyamg and petsc4py. In that
    environment petsc produces no result at all, and the cross-check skips
    entries that carry no solution.
    """

    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.split(".")[0] in {"pyamg", "petsc4py"}:
            raise ImportError(f"{name} is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)

    comparisons = poisson_benchmark.compare_poisson_solvers(
        n=_LIVE_N,
        tt_settings=poisson_benchmark.TTSolverSettings(tolerance=1e-8, max_sweeps=10),
    )

    # petsc produced nothing at all; pyamg degraded to plain CG and said so
    assert comparisons["petsc"] is None
    assert comparisons["pyamg"]["backend"] == "scipy-cg"
    # the backends that do exist still solved the system and were cross-checked
    assert comparisons["mals"]["error_vs_exact"] < 1e-6
    assert comparisons["mals_vs_amen"] < 1e-6
    assert "mals_vs_petsc" not in comparisons

    # and the report built from it blanks both unavailable columns
    by_backend = {
        row["backend"]: row
        for row in poisson_benchmark._flatten_poisson_comparison(comparisons)
    }
    assert by_backend["petsc-cg+gamg"]["status"] == "not available"
    assert by_backend["scipy-cg+pyamg"]["status"] == "not available"


def test_petsc_backend_builds_its_own_problem_when_given_none():
    """Applies only where petsc4py is installed; CI skips it.

    The skip is keyed on the backend's own return value rather than on a
    separate import probe. Returning ``None`` when petsc4py is absent is the
    documented contract.
    """

    result = poisson_benchmark.benchmark_petsc_poisson(n=_LIVE_N, tolerance=1e-10)
    if result is None:
        pytest.skip("petsc4py is not installed")

    assert result["backend"] == "petsc-cg+gamg"
    assert result["error_vs_exact"] < 1e-6
