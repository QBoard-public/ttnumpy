"""Benchmarks for the Poisson equation solvers."""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import signal
import time
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.sparse.linalg import cg
import scipy as sp

from benchmarks.poisson_problem import (
    PoissonProblem,
    build_poisson_problem,
    solve_poisson_tt,
)

from ttnumpy.logger_config import (
    LOG_LEVEL_ENV_VAR,
    PACKAGE_LOGGER_NAME,
    setup_logging,
)

logger = logging.getLogger(PACKAGE_LOGGER_NAME)

# --------------------------- benchmark configuration ------------------------

# QTT depths per axis to benchmark by default; the grid is 2**n x 2**n.
DEFAULT_BENCHMARK_NS: tuple[int, ...] = (4, 5, 6, 7, 8)

# Where JSON and Markdown reports are written by default.
DEFAULT_OUTPUT_DIR = Path("benchmark-results")

# Wall-clock budget for a single solve. A backend that exceeds it is reported
# as a timed-out row instead of holding up the suite.
TIMEOUT: float = 600.0

# Target relative residual shared by every iterative solver, TT and classical
# alike, so that the times compare solves of equal accuracy.
DEFAULT_TOLERANCE: float = 1e-5

# A benchmark suite runs for a long time, so per-run progress is on by default.
DEFAULT_LOG_LEVEL = "INFO"


@dataclass(frozen=True)
class TTSolverSettings:
    """TT solver parameters used by the benchmarks."""

    # target relative residual; drives both the stopping test and the
    # residual-based truncation of the TT solvers
    tolerance: float = DEFAULT_TOLERANCE
    # bond dimension cap; None lets the tolerance alone decide the ranks
    max_rank: int | None = None
    # maximum number of sweeps; the solvers stop earlier when the residual
    # converges or stagnates, so this only caps pathological runs
    max_sweeps: int = 20
    # AMEn enrichment rank
    kickrank: int = 4
    # mode size of the TT operator and right-hand side: 2 is QTT, 2**k fuses
    # k QTT cores, 2**n leaves one core per grid axis
    mode_size: int = 2
    # seed for AMEn's random residual train
    seed: int | None = 0


DEFAULT_TT_SETTINGS = TTSolverSettings()

# -----------------------------------------------------------------------------

_BENCHMARK_BACKENDS: tuple[tuple[str, str], ...] = (
    ("mals", "tt-mals"),
    ("amen", "tt-amen"),
    ("scipy", "scipy-cg"),
    ("pyamg", "scipy-cg+pyamg"),
    ("petsc", "petsc-cg+gamg"),
)

# TT solvers that get a row of their own in every report.
_TT_SOLVERS: tuple[str, ...] = ("mals", "amen")


@contextlib.contextmanager
def _time_limit(seconds: float):
    """Raise TimeoutError if the wrapped block runs longer than ``seconds``.

    Relies on SIGALRM, so the limit is checked between Python operations: a
    single long BLAS call runs to completion before the alarm is delivered.
    """

    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def _expire(_signum, _frame):
        raise TimeoutError(f"exceeded the {seconds:.0f}s benchmark time limit")

    previous = signal.signal(signal.SIGALRM, _expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def _run_backend(label: str, runner: Callable[[], Any]) -> dict[str, Any] | None:
    """Run one backend, turning an expected failure into a result row.

    A backend that exceeds :data:`TIMEOUT`, runs out of memory, or raises
    ``ValueError``, ``RuntimeError`` or ``LinAlgError`` is reported as a row
    carrying its status, so one failure does not cost the suite the results it
    has already gathered. Anything else propagates: an unexpected exception is
    a defect, and hiding it behind a "failed" row would only delay finding it.
    """

    status = "failed"
    start = time.perf_counter()
    try:
        with _time_limit(TIMEOUT):
            return runner()
    except TimeoutError:
        status = f"timeout>{TIMEOUT:.0f}s"
        logger.warning("%s exceeded the %.0fs time limit", label, TIMEOUT)
    except (MemoryError, ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        status = f"failed: {type(exc).__name__}"
        logger.warning("%s failed: %s", label, exc)
    return {
        "backend": label,
        "elapsed_s": time.perf_counter() - start,
        "status": status,
    }


def _tt_to_vector(solution) -> np.ndarray:
    return np.asarray(solution.get_full_tensor()).reshape(-1)


def _relative_difference(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference_norm = np.linalg.norm(reference)
    if reference_norm == 0.0:
        return np.linalg.norm(reference - candidate)
    return np.linalg.norm(reference - candidate) / reference_norm


def _relative_residual(
    laplacian: sp.sparse.csr_matrix, solution: np.ndarray, rhs: np.ndarray
) -> float:
    rhs_norm = np.linalg.norm(rhs)
    if rhs_norm == 0.0:
        return np.linalg.norm(laplacian @ solution - rhs)
    return np.linalg.norm(laplacian @ solution - rhs) / rhs_norm


def benchmark_tt_poisson(
    n: int = 3,
    problem: PoissonProblem | None = None,
    tt_settings: TTSolverSettings = DEFAULT_TT_SETTINGS,
    solver: str = "mals",
) -> dict[str, Any]:
    """Solve the manufactured Poisson problem with one of the TT solvers.

    Problem construction (matrix assembly, reference solve, rhs compression)
    happens outside the timed region; only the TT solve is timed.
    """

    if problem is None:
        problem = build_poisson_problem(
            N=n, tt_format=True, mode_size=tt_settings.mode_size
        )

    logger.info(
        "Starting TT %s Poisson benchmark for n=%s (grid=%sx%s, mode size %s)",
        solver.upper(),
        n,
        problem.K,
        problem.K,
        problem.mode_size,
    )

    start = time.perf_counter()
    tt_solution, info = solve_poisson_tt(
        problem,
        solver=solver,
        max_rank=tt_settings.max_rank,
        max_sweeps=tt_settings.max_sweeps,
        residual_tol=tt_settings.tolerance,
        kickrank=tt_settings.kickrank,
        seed=tt_settings.seed,
        return_info=True,
    )
    elapsed = time.perf_counter() - start

    logger.info(
        "Finished TT %s benchmark for n=%s in %.3fs", solver.upper(), n, elapsed
    )

    solution = _tt_to_vector(tt_solution)

    return {
        "backend": f"tt-{solver}",
        "n": n,
        "elapsed_s": elapsed,
        "solution": solution,
        "tt_solution": tt_solution,
        "sweeps": info.num_sweeps_run,
        "status": str(info.stop_reason),
        "residuals": info.residuals,
        "max_bond": max(tt_solution.bonds),
        "error_vs_exact": (
            None
            if problem.exact is None
            else _relative_difference(problem.exact, solution)
        ),
        "relative_residual": _relative_residual(
            problem.laplacian, solution, -problem.rhs
        ),
    }


def benchmark_scipy_poisson(
    n: int = 3,
    problem: PoissonProblem | None = None,
    preconditioned: bool = False,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    """Solve the same Poisson system with SciPy sparse CG.

    With ``preconditioned=True``, PyAMG smoothed aggregation is used and the
    AMG hierarchy setup is part of the timed region, consistent with the
    PETSc backend where GAMG setup happens inside ``ksp.solve``. Falls back
    to plain CG when pyamg is not installed.
    """

    if problem is None:
        problem = build_poisson_problem(N=n)

    pyamg = None
    if preconditioned:
        try:
            import pyamg
        except ImportError:
            pass
    backend = "scipy-cg" if pyamg is None else "scipy-cg+pyamg"

    logger.info(
        "Starting %s benchmark for n=%s (grid=%sx%s)",
        backend,
        n,
        problem.K,
        problem.K,
    )

    iterations = 0

    def _count_iterations(_: np.ndarray) -> None:
        nonlocal iterations
        iterations += 1

    start = time.perf_counter()
    preconditioner = None
    if pyamg is not None:
        multigrid = pyamg.smoothed_aggregation_solver(problem.laplacian)
        preconditioner = multigrid.aspreconditioner()
    solution, info = cg(
        problem.laplacian,
        -problem.rhs,
        M=preconditioner,
        rtol=tolerance,
        atol=0.0,
        callback=_count_iterations,
    )
    elapsed = time.perf_counter() - start

    logger.info("Finished %s benchmark for n=%s in %.3fs", backend, n, elapsed)

    return {
        "backend": backend,
        "n": n,
        "elapsed_s": elapsed,
        "cg_info": info,
        "iterations": iterations,
        "solution": solution,
        "status": "converged" if info == 0 else f"cg info={info}",
        "error_vs_exact": (
            None
            if problem.exact is None
            else _relative_difference(problem.exact, solution)
        ),
        "relative_residual": _relative_residual(
            problem.laplacian, solution, -problem.rhs
        ),
    }


def benchmark_petsc_poisson(
    n: int = 3,
    problem: PoissonProblem | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, Any] | None:
    """Solve the same Poisson system with PETSc CG and GAMG if petsc4py is installed."""

    try:
        from petsc4py import PETSc
    except ImportError:
        return None

    if problem is None:
        problem = build_poisson_problem(N=n)
    size = problem.laplacian.shape[0]

    logger.info(
        "Starting PETSc CG+GAMG benchmark for n=%s (grid=%sx%s)",
        n,
        problem.K,
        problem.K,
    )

    matrix = PETSc.Mat().createAIJ(size=(size, size))
    matrix.setUp()
    matrix.setValuesCSR(
        problem.laplacian.indptr.astype(np.int32),
        problem.laplacian.indices.astype(np.int32),
        problem.laplacian.data,
    )
    matrix.assemble()

    rhs_vec = PETSc.Vec().createSeq(size)
    rhs_vec.setArray((-problem.rhs).copy())
    solution_vec = PETSc.Vec().createSeq(size)

    ksp = PETSc.KSP().create()
    ksp.setOperators(matrix)
    ksp.setType("cg")
    ksp.getPC().setType("gamg")
    ksp.setTolerances(rtol=tolerance, atol=0.0)
    # GAMG makes CG measure the preconditioned residual by default, which is a
    # different quantity from the one SciPy and the TT solvers test against
    ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
    ksp.setFromOptions()

    start = time.perf_counter()
    ksp.solve(rhs_vec, solution_vec)
    elapsed = time.perf_counter() - start

    logger.info("Finished PETSc CG+GAMG benchmark for n=%s in %.3fs", n, elapsed)

    solution = np.array(solution_vec.getArray(readonly=True), copy=True)
    residual_vec = rhs_vec.duplicate()
    matrix.mult(solution_vec, residual_vec)
    residual_vec.aypx(-1.0, rhs_vec)

    return {
        "backend": "petsc-cg+gamg",
        "n": n,
        "elapsed_s": elapsed,
        "iterations": ksp.getIterationNumber(),
        "solution": solution,
        "status": "converged" if ksp.getConvergedReason() > 0 else "not converged",
        "error_vs_exact": (
            None
            if problem.exact is None
            else _relative_difference(problem.exact, solution)
        ),
        "relative_residual": residual_vec.norm() / rhs_vec.norm(),
    }


def compare_poisson_solvers(
    n: int = 3,
    tt_settings: TTSolverSettings = DEFAULT_TT_SETTINGS,
    compute_reference: bool = True,
) -> dict[str, Any]:
    """Run every Poisson solver and collect timing and accuracy metrics.

    The problem is built once and shared, so every backend solves the exact
    same system and none of them is charged for problem construction. Each
    TT solver gets its own entry; all iterative solvers are held to
    ``tt_settings.tolerance``.
    """

    problem = build_poisson_problem(
        N=n,
        tt_format=True,
        compute_reference=compute_reference,
        mode_size=tt_settings.mode_size,
    )
    tolerance = tt_settings.tolerance

    comparisons: dict[str, Any] = {
        "n": n,
        "grid_size": problem.K,
        "unknowns": problem.K**2,
        "mode_size": problem.mode_size,
        "tolerance": tolerance,
    }

    for solver in _TT_SOLVERS:
        comparisons[solver] = _run_backend(
            f"tt-{solver}",
            lambda solver=solver: benchmark_tt_poisson(
                n=n, problem=problem, tt_settings=tt_settings, solver=solver
            ),
        )
    comparisons["scipy"] = _run_backend(
        "scipy-cg",
        lambda: benchmark_scipy_poisson(n=n, problem=problem, tolerance=tolerance),
    )
    comparisons["pyamg"] = _run_backend(
        "scipy-cg+pyamg",
        lambda: benchmark_scipy_poisson(
            n=n, problem=problem, preconditioned=True, tolerance=tolerance
        ),
    )
    comparisons["petsc"] = _run_backend(
        "petsc-cg+gamg",
        lambda: benchmark_petsc_poisson(n=n, problem=problem, tolerance=tolerance),
    )

    reference = comparisons["mals"].get("solution")
    if reference is not None:
        for name in ("amen", "scipy", "pyamg", "petsc"):
            result = comparisons.get(name)
            if result is None or result.get("solution") is None:
                continue
            comparisons[f"mals_vs_{name}"] = _relative_difference(
                reference, result["solution"]
            )

    return comparisons


def _benchmark_note(result: dict[str, Any]) -> str:
    if "sweeps" in result:
        note = f"sweeps={result['sweeps']}"
        if result.get("max_bond") is not None:
            note += f" bond={result['max_bond']}"
        return note
    if "iterations" in result:
        return f"iters={result['iterations']}"
    return ""


def _flatten_poisson_comparison(comparisons: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    shared = {
        "n": comparisons["n"],
        "grid_size": comparisons["grid_size"],
        "grid_shape": f"{comparisons['grid_size']}x{comparisons['grid_size']}",
        "unknowns": comparisons["unknowns"],
        "mode_size": comparisons["mode_size"],
        "tolerance": comparisons["tolerance"],
    }
    for key, label in _BENCHMARK_BACKENDS:
        result = comparisons.get(key)
        if result is not None and result.get("backend", label) != label:
            logger.info(
                "%s is unavailable: the run fell back to %s",
                label,
                result.get("backend"),
            )
            result = None
        if result is None:
            rows.append(
                {
                    **shared,
                    "backend": label,
                    "elapsed_s": None,
                    "error_vs_exact": None,
                    "relative_residual": None,
                    "max_bond": None,
                    "status": "not available",
                    "notes": "",
                }
            )
            continue

        rows.append(
            {
                **shared,
                "backend": label,
                "elapsed_s": result.get("elapsed_s"),
                "error_vs_exact": result.get("error_vs_exact"),
                "relative_residual": result.get("relative_residual"),
                "max_bond": result.get("max_bond"),
                "status": result.get("status", ""),
                "notes": _benchmark_note(result),
            }
        )

    return rows


def _format_float(value: float) -> str:
    return f"{value:.3e}"


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return _format_float(value)


def _format_optional_text(value: str | None) -> str:
    if not value:
        return "n/a"
    return value


# Metrics shown inside each backend cell: (label, row field).
_CELL_METRICS: tuple[tuple[str, str], ...] = (
    ("real time (s)", "elapsed_s"),
    ("err vs exact", "error_vs_exact"),
    ("rel residual", "relative_residual"),
    ("status", "status"),
    ("notes", "notes"),
)

_TEXT_FIELDS = frozenset({"status", "notes"})


def _format_cell_metric(row: dict[str, Any], field: str) -> str:
    if field in _TEXT_FIELDS:
        return _format_optional_text(row[field])
    return _format_optional_float(row[field])


def _group_poisson_rows_by_n(
    rows: list[dict[str, Any]],
) -> dict[int, dict[str, dict[str, Any]]]:
    grouped: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["n"], {})[row["backend"]] = row
    return grouped


def _render_poisson_benchmark_rows(rows: list[dict[str, Any]]) -> str:
    backend_labels = tuple(label for _, label in _BENCHMARK_BACKENDS)
    headers = ("n", "grid", "") + backend_labels

    rendered_rows: list[tuple[str, ...]] = []
    for n, backends in _group_poisson_rows_by_n(rows).items():
        grid_shape = next(iter(backends.values()))["grid_shape"]
        for index, (metric_label, field) in enumerate(_CELL_METRICS):
            rendered_rows.append(
                (
                    str(n) if index == 0 else "",
                    grid_shape if index == 0 else "",
                    metric_label,
                    *(
                        (
                            _format_cell_metric(backends[label], field)
                            if label in backends
                            else "n/a"
                        )
                        for label in backend_labels
                    ),
                )
            )

    widths = [
        max([len(headers[i])] + [len(row[i]) for row in rendered_rows])
        for i in range(len(headers))
    ]

    def _row(values: tuple[str, ...]) -> str:
        return "  ".join(value.ljust(widths[i]) for i, value in enumerate(values))

    separator = _row(tuple("-" * width for width in widths))
    lines = [_row(headers), separator]
    for index, row in enumerate(rendered_rows):
        if index and index % len(_CELL_METRICS) == 0:
            lines.append(separator)
        lines.append(_row(row))
    return "\n".join(lines)


def _render_poisson_benchmark_markdown(report: dict[str, Any]) -> str:
    backend_labels = tuple(label for _, label in _BENCHMARK_BACKENDS)
    lines = [
        "# Poisson benchmark results",
        "",
        f"Mode size {report['mode_size']}, tolerance {report['tolerance']:.1e} "
        f"for every iterative solver, {TIMEOUT:.0f}s time limit per solve.",
        "",
        "| n | grid | " + " | ".join(backend_labels) + " |",
        "| --- | --- | " + " | ".join("---" for _ in backend_labels) + " |",
    ]
    for n, backends in _group_poisson_rows_by_n(report["rows"]).items():
        grid_shape = next(iter(backends.values()))["grid_shape"]
        cells = []
        for label in backend_labels:
            if label not in backends:
                cells.append("n/a")
                continue
            cells.append(
                "<br>".join(
                    f"{metric_label}: {_format_cell_metric(backends[label], field)}"
                    for metric_label, field in _CELL_METRICS
                )
            )
        lines.append(f"| {n} | {grid_shape} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


_REPORT_NAME_PARTS: tuple[tuple[str, str], ...] = (
    ("max_rank", "rank"),
    ("max_sweeps", "sweeps"),
    ("kickrank", "kick"),
    ("seed", "seed"),
)


def _report_stem(report: dict[str, Any]) -> str:
    """Name reports after their parameters so separate runs do not collide.

    Only parameters that differ from :data:`DEFAULT_TT_SETTINGS` are appended,
    so a default run keeps the short name its artifacts have always had while
    two runs that differ in anything at all land in separate files.
    """

    stem = f"poisson_benchmark_mode{report['mode_size']}_tol{report['tolerance']:.0e}"

    settings = report.get("tt_settings")
    if settings is None:
        return stem
    for field, label in _REPORT_NAME_PARTS:
        value = getattr(settings, field)
        if value != getattr(DEFAULT_TT_SETTINGS, field):
            stem += f"_{label}{value}"
    return stem


def write_poisson_benchmark_report(
    report: dict[str, Any], output_dir: Path
) -> dict[str, Path]:
    """Write the benchmark report to JSON and Markdown files."""

    output_dir.mkdir(parents=True, exist_ok=True)

    stem = _report_stem(report)
    json_path = output_dir / f"{stem}.json"
    markdown_path = output_dir / f"{stem}.md"
    json_report = {
        "ns": report["ns"],
        "mode_size": report["mode_size"],
        "tolerance": report["tolerance"],
        "timeout_s": TIMEOUT,
        "rows": report["rows"],
    }
    settings = report.get("tt_settings")
    if settings is not None:
        json_report["tt_settings"] = dataclasses.asdict(settings)

    json_path.write_text(
        json.dumps(json_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        _render_poisson_benchmark_markdown(report), encoding="utf-8"
    )

    return {"json": json_path, "markdown": markdown_path}


def print_poisson_benchmark_suite_table(report: dict[str, Any]) -> None:
    """Print the benchmark suite table for multiple grid sizes."""

    print()
    print(
        "Poisson benchmark suite (n={}, mode size {}, tolerance {:.1e}):".format(
            ", ".join(str(n) for n in report["ns"]),
            report["mode_size"],
            report["tolerance"],
        )
    )
    print(_render_poisson_benchmark_rows(report["rows"]))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark Poisson solvers (TT MALS, TT AMEn, SciPy, PyAMG, PETSc)."
    )
    parser.add_argument(
        "-n",
        type=int,
        default=None,
        help="Run a single benchmark at this QTT depth per axis (grid size 2**n)",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=None,
        help="Run the benchmark suite for these QTT depths per axis (default: 4 5 6 7 8)",
    )
    parser.add_argument(
        "--mode-size",
        type=int,
        default=DEFAULT_TT_SETTINGS.mode_size,
        help="Mode size of the TT operator and right-hand side: 2 is QTT, 2**k "
        "fuses k QTT cores on the same grid (log2 must divide n)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TT_SETTINGS.tolerance,
        help="Target relative residual for every iterative solver (TT and classical)",
    )
    parser.add_argument(
        "--max-rank",
        type=int,
        default=DEFAULT_TT_SETTINGS.max_rank,
        help="Bond dimension cap for the TT solvers (default: unbounded)",
    )
    parser.add_argument(
        "--max-sweeps",
        type=int,
        default=DEFAULT_TT_SETTINGS.max_sweeps,
        help="Maximum number of TT sweeps (the solvers stop earlier when the "
        "residual converges or stagnates)",
    )
    parser.add_argument(
        "--kickrank",
        type=int,
        default=DEFAULT_TT_SETTINGS.kickrank,
        help="AMEn enrichment rank",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_TT_SETTINGS.seed,
        help="Seed for AMEn's random residual train",
    )
    parser.add_argument(
        "--no-reference",
        action="store_true",
        help="Skip the sparse direct reference solve; the error columns become n/a",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where benchmark JSON and Markdown reports are written",
    )
    parser.add_argument(
        "--show-solver-differences",
        action="store_true",
        help="Also print MALS-vs-other-backend solution differences",
    )
    parser.add_argument(
        "--log-level",
        default=DEFAULT_LOG_LEVEL,
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Verbosity of ttnumpy log output; INFO reports per-backend progress "
        f"(default: {DEFAULT_LOG_LEVEL}). Takes precedence over {LOG_LEVEL_ENV_VAR}.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run Poisson benchmarks from the command line."""

    args = _build_parser().parse_args(argv)
    setup_logging(args.log_level)

    sizes = (
        args.sizes
        if args.sizes is not None
        else ([args.n] if args.n is not None else list(DEFAULT_BENCHMARK_NS))
    )
    tt_settings = TTSolverSettings(
        tolerance=args.tolerance,
        max_rank=args.max_rank,
        max_sweeps=args.max_sweeps,
        kickrank=args.kickrank,
        mode_size=args.mode_size,
        seed=args.seed,
    )

    runs: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for index, n in enumerate(sizes, start=1):
        logger.info("Running benchmark set %s/%s for n=%s", index, len(sizes), n)
        comparisons = compare_poisson_solvers(
            n=n,
            tt_settings=tt_settings,
            compute_reference=not args.no_reference,
        )
        runs.append(comparisons)
        rows.extend(_flatten_poisson_comparison(comparisons))
    report = {
        "ns": list(sizes),
        "mode_size": args.mode_size,
        "tolerance": args.tolerance,
        "tt_settings": tt_settings,
        "runs": runs,
        "rows": rows,
    }

    print_poisson_benchmark_suite_table(report)

    if args.show_solver_differences:
        for run in runs:
            print()
            print(f"MALS solution differences vs other backends (n={run['n']}):")
            for name in ("amen", "scipy", "pyamg", "petsc"):
                key = f"mals_vs_{name}"
                if key in run:
                    print(f"  mals vs {name}: {_format_float(run[key])}")

    output_paths = write_poisson_benchmark_report(report, args.output_dir)
    print()
    print(
        f"Wrote benchmark results to {output_paths['markdown']} and {output_paths['json']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
