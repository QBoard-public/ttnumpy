# Copyright (c) 2026 QBoard LLC. Licensed under the AGPLv3. See LICENSE.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Benchmarks of the arithmetic of tensor trains.

Two tables:

- **accuracy**: products, sums and an operator applied to QTT vectors of
  ``2**16`` to ``2**40`` entries, rounded with the direct method, against the
  same operations in NumPy. It reports the memory of both forms and the
  error of the train.
- **speed**: the direct and the randomized rounding methods on the same
  workloads. It reports the time and the error of each.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import platform
import statistics
import subprocess
import textwrap
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import scipy

import ttnumpy as tt
from ttnumpy.logger_config import (
    LOG_LEVEL_ENV_VAR,
    PACKAGE_LOGGER_NAME,
    setup_logging,
)

logger = logging.getLogger(PACKAGE_LOGGER_NAME)

# --------------------------- benchmark configuration ------------------------

TABLES: tuple[str, ...] = ("accuracy", "speed")

# Numbers of binary cores of the accuracy table; the vectors have 2**cores entries.
DEFAULT_ACCURACY_CORES: tuple[int, ...] = (16, 20, 24, 30, 40)

# Up to this number of cores the NumPy reference covers every entry. Above
# it the reference is evaluated at DEFAULT_SAMPLES random entries.
DEFAULT_DENSE_MAX_CORES = 24
DEFAULT_SAMPLES = 10_000

# Relative error requested from every rounding of the accuracy table.
ACCURACY_ERROR = 1e-10

# Relative error requested from every rounding of the speed table, together
# with the max_rank of each workload.
SPEED_ERROR = 1e-5

# Timing rounds of the speed table and seeds of its randomized method.
DEFAULT_ROUNDS = 10
DEFAULT_SEEDS = 10

# Each timed call is repeated until a batch lasts at least this long.
BATCH_SECONDS = 0.02

DEFAULT_OUTPUT_DIR = Path("benchmark-results")
REPORT_STEM = "arithmetic_benchmark"
DEFAULT_LOG_LEVEL = "INFO"

# Frequencies of the four sums of sines of the accuracy table.
SINE_FREQUENCIES: tuple[tuple[int, ...], ...] = (
    (1, 3, 7, 13),
    (2, 5, 11, 17),
    (4, 6, 9, 19),
    (8, 10, 15, 23),
)

# Taylor coefficients of exp(t / 2) up to degree 5, highest degree first.
TAYLOR_COEFFICIENTS: tuple[float, ...] = (
    1 / 3840,
    1 / 384,
    1 / 48,
    1 / 8,
    1 / 2,
    1.0,
)

# ------------------------------- inputs -------------------------------------

Wave = tuple[float, float, float]


def sine_waves(frequencies: tuple[int, ...], seed: int) -> list[Wave]:
    """Angular frequency, phase and amplitude of each sine of a sum.

    Phases are uniform in ``[0, 2 pi)`` and amplitudes uniform in
    ``[0.5, 1.5)``.
    """
    rng = np.random.default_rng(seed)
    return [
        (2 * np.pi * frequency, rng.uniform(0, 2 * np.pi), rng.uniform(0.5, 1.5))
        for frequency in frequencies
    ]


def sine_qtt(n_cores: int, omega: float, phase: float = 0.0) -> tt.TensorTrain:
    """``sin(omega * x + phase)`` at ``x = i / 2**n_cores`` as a QTT train of bond 2.

    The first core holds the most significant bit of ``i``. Each core rotates
    the pair ``(sin, cos)`` by the angle of its bit, so the train is exact up
    to rounding for any number of cores.
    """
    cores = []
    for k in range(n_cores):
        angle = omega * 2.0 ** (-1 - k)
        if k == 0:
            core = np.array(
                [
                    [np.sin(phase), np.cos(phase)],
                    [np.sin(phase + angle), np.cos(phase + angle)],
                ]
            )
            cores.append(core.reshape((1, 2, 1, 2)))
            continue
        # rotation of the row vector (sin, cos) by 0 or by the angle of bit k
        core = np.array(
            [[[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]] for t in (0.0, angle)]
        ).transpose(1, 0, 2)
        if k == n_cores - 1:
            core = core[:, :, :1]
        cores.append(core.reshape((core.shape[0], 2, 1, core.shape[2])))
    return tt.TensorTrain(cores)


def sine_sum_qtt(n_cores: int, waves: list[Wave]) -> tt.TensorTrain:
    """Sum of sines as a QTT train, rounded at the default error."""
    total = None
    for omega, phase, amplitude in waves:
        term = sine_qtt(n_cores, omega, phase).multiply(float(amplitude))
        total = term if total is None else total.add(term)
    return total.truncate()


def sine_sum_values(n_cores: int, waves: list[Wave], indices: np.ndarray) -> np.ndarray:
    """Sum of sines at the entries ``indices`` of a grid of ``2**n_cores`` points."""
    x = indices / 2.0**n_cores
    return sum(
        amplitude * np.sin(omega * x + phase) for omega, phase, amplitude in waves
    )


def tridiagonal_qtt(
    n_cores: int, diagonal: float, off_diagonal: float
) -> tt.TensorTrain:
    """Tridiagonal Toeplitz matrix of size ``2**n_cores`` as a QTT operator of bond 3."""
    eye, up = np.eye(2), np.eye(2, k=1)
    down, zero = up.T, np.zeros((2, 2))
    first = np.stack([eye, down, up])[None]
    middle = np.array([[eye, down, up], [zero, up, zero], [zero, zero, down]])
    last = np.stack(
        [
            diagonal * eye + off_diagonal * (up + down),
            off_diagonal * up,
            off_diagonal * down,
        ]
    )[:, None]
    blocks = [first] + [middle] * (n_cores - 2) + [last]
    return tt.TensorTrain([block.transpose(0, 2, 3, 1) for block in blocks])


def sample_qtt(train: tt.TensorTrain, indices: np.ndarray) -> np.ndarray:
    """Entries of a QTT vector at the integer ``indices``, first core most significant."""
    n_cores = train.n_cores
    bits = (indices[:, None] >> (n_cores - 1 - np.arange(n_cores))) & 1
    vectors = train.cores[0][0, bits[:, 0], 0, :]
    for k in range(1, n_cores):
        core = train.cores[k][:, :, 0, :]
        out = np.empty(
            (len(indices), core.shape[2]), dtype=np.result_type(vectors, core)
        )
        for bit in (0, 1):
            rows = bits[:, k] == bit
            out[rows] = vectors[rows] @ core[:, bit, :]
        vectors = out
    return vectors[:, 0]


# ---------------------------- accuracy table --------------------------------


def _accuracy_operations(
    n_cores: int, indices: np.ndarray
) -> list[tuple[str, Callable[[], tt.TensorTrain], Callable[[], np.ndarray]]]:
    """Name, TT computation and NumPy reference at ``indices`` of each operation."""
    waves = [
        sine_waves(frequencies, seed)
        for seed, frequencies in enumerate(SINE_FREQUENCIES)
    ]
    trains = [sine_sum_qtt(n_cores, wave) for wave in waves]
    values = [sine_sum_values(n_cores, wave, indices) for wave in waves]
    mass = tridiagonal_qtt(n_cores, 4 / 6, 1 / 6)
    size = 2**n_cores

    def mass_reference() -> np.ndarray:
        left = np.where(
            indices > 0, sine_sum_values(n_cores, waves[0], indices - 1), 0.0
        )
        right = np.where(
            indices < size - 1, sine_sum_values(n_cores, waves[0], indices + 1), 0.0
        )
        return (4 * values[0] + left + right) / 6

    def taylor_chain() -> tt.TensorTrain:
        ones = tt.ones([2] * n_cores)
        result = ones.multiply(TAYLOR_COEFFICIENTS[0])
        for coefficient in TAYLOR_COEFFICIENTS[1:]:
            result = result.multiply(trains[0], truncate=True, error=ACCURACY_ERROR)
            result = result.add(
                ones.multiply(coefficient), truncate=True, error=ACCURACY_ERROR
            )
        return result

    return [
        (
            "f * g",
            lambda: trains[0].multiply(trains[1], truncate=True, error=ACCURACY_ERROR),
            lambda: values[0] * values[1],
        ),
        (
            "f1 + f2 + f3 + f4",
            lambda: trains[0]
            .add(trains[1])
            .add(trains[2])
            .add(trains[3], truncate=True, error=ACCURACY_ERROR),
            lambda: values[0] + values[1] + values[2] + values[3],
        ),
        (
            "M @ f",
            lambda: mass.dot(trains[0], truncate=True, error=ACCURACY_ERROR),
            mass_reference,
        ),
        (
            "exp(f / 2), degree 5",
            taylor_chain,
            lambda: np.polyval(TAYLOR_COEFFICIENTS, values[0]),
        ),
    ]


def accuracy_rows(
    cores: tuple[int, ...] = DEFAULT_ACCURACY_CORES,
    dense_max_cores: int = DEFAULT_DENSE_MAX_CORES,
    samples: int = DEFAULT_SAMPLES,
) -> list[dict[str, Any]]:
    """Rows of the accuracy table, one per operation and number of cores.

    The error is the relative root-mean-square difference between the train
    and NumPy over every entry up to ``dense_max_cores`` cores, over
    ``samples`` random entries above.
    """
    rows = []
    for n_cores in cores:
        logger.info("Accuracy table: %s cores", n_cores)
        size = 2**n_cores
        everywhere = n_cores <= dense_max_cores
        if everywhere:
            indices = np.arange(size)
        else:
            indices = np.random.default_rng(n_cores).integers(0, size, samples)
        for name, compute, reference in _accuracy_operations(n_cores, indices):
            train = compute()
            computed = train.matricize() if everywhere else sample_qtt(train, indices)
            expected = reference()
            tt_bytes = sum(core.nbytes for core in train.cores)
            rows.append(
                {
                    "operation": name,
                    "cores": n_cores,
                    "bond": train.max_rank,
                    "tt_bytes": tt_bytes,
                    "dense_bytes": size * np.dtype(np.float64).itemsize,
                    "compression": size * np.dtype(np.float64).itemsize / tt_bytes,
                    "error": float(
                        np.linalg.norm(computed - expected) / np.linalg.norm(expected)
                    ),
                    "points": "all" if everywhere else samples,
                }
            )
    return rows


# ------------------------------ speed table ---------------------------------


def multiscale_field(n_cores: int, seed: int, rank: int) -> tt.TensorTrain:
    """Random Fourier series with 4000 modes and amplitudes ``k**-1.2``, rounded to ``rank``."""
    rng = np.random.default_rng(seed)
    modes = np.arange(1, 4001)
    spectrum = np.zeros(2 ** (n_cores - 1) + 1, dtype=complex)
    spectrum[1:4001] = (
        rng.standard_normal(modes.size)
        * modes**-1.2
        * np.exp(1j * rng.uniform(0, 2 * np.pi, modes.size))
    )
    values = np.fft.irfft(spectrum, n=2**n_cores) * 2 ** (n_cores - 1)
    return tt.tt_svd(values.reshape([2] * n_cores), error=1e-10, max_rank=rank)


def two_core_fields(size: int) -> tuple[tt.TensorTrain, tt.TensorTrain]:
    """Two smooth functions on a ``size`` x ``size`` grid as trains of two cores.

    The functions are ``1 / (1 + 25 ((x - 0.3)**2 + (y - 0.6)**2))`` and
    ``exp(-sin(2 pi x + 1) cos(2 pi y))`` at the cell centres of the unit
    square, rounded at a relative error of ``1e-10``.
    """
    x = (np.arange(size) + 0.5) / size
    x, y = x[:, None], x[None, :]
    runge = 1 / (1 + 25 * ((x - 0.3) ** 2 + (y - 0.6) ** 2))
    periodic = np.exp(-np.sin(2 * np.pi * x + 1) * np.cos(2 * np.pi * y))
    return tt.tt_svd(runge, error=1e-10), tt.tt_svd(periodic, error=1e-10)


def rounding_workload(
    group: str,
    name: str,
    built: tt.TensorTrain,
    max_rank: int,
    reference: Any,
) -> dict[str, Any]:
    """Workload that rounds an already built train."""
    return {
        "group": group,
        "name": name,
        "input_bond": built.max_rank,
        "max_rank": max_rank,
        "direct": lambda: built.truncate(SPEED_ERROR, max_rank=max_rank),
        "randomized": lambda seed: built.truncate(
            SPEED_ERROR, max_rank=max_rank, method="randomized", seed=seed
        ),
        "reference": reference,
    }


def product_workload(
    name: str, left: tt.TensorTrain, right: tt.TensorTrain, max_rank: int
) -> dict[str, Any]:
    """Workload that multiplies two trains element-wise and rounds the product."""
    return {
        "group": "element-wise product",
        "name": name,
        "input_bond": left.max_rank * right.max_rank,
        "max_rank": max_rank,
        "direct": lambda: left.multiply(
            right, truncate=True, error=SPEED_ERROR, max_rank=max_rank
        ),
        "randomized": lambda seed: left.multiply(
            right,
            truncate=True,
            error=SPEED_ERROR,
            max_rank=max_rank,
            method="randomized",
            seed=seed,
        ),
        "reference": left.matricize() * right.matricize(),
    }


def _transport_rate_workloads(
    n_cores: int, laplacian: tt.TensorTrain
) -> list[dict[str, Any]]:
    """Sums of operator products and fluxes, as in the rate of a transport step."""
    workloads = []
    for rank, caps in ((16, (48,)), (32, (64, 32)), (64, (80,))):
        f, u, v, source = (
            multiscale_field(n_cores, seed, rank) for seed in (3, 4, 5, 6)
        )
        flux_x = f.multiply(u).truncate(SPEED_ERROR)
        flux_y = f.multiply(v).truncate(SPEED_ERROR)
        rate = (laplacian @ f).multiply(0.1)
        rate = rate.add((laplacian @ flux_x).multiply(-1))
        rate = rate.add((laplacian @ flux_y).multiply(-1)).add(source)
        workloads += [
            rounding_workload(
                "rounding of a sum",
                f"4 terms, fields of bond {rank}",
                rate,
                cap,
                rate.matricize(),
            )
            for cap in caps
        ]
    return workloads


def _sum_workloads(n_cores: int) -> list[dict[str, Any]]:
    """A sum of seven fields and a sum that cancels."""
    f, u = multiscale_field(n_cores, 3, 16), multiscale_field(n_cores, 4, 16)
    total = f
    for j in range(1, 7):
        total = total.add((u if j % 2 else f).multiply(0.3 * j))
    near_zero = f.add(f.multiply(-1)).add(u.multiply(1e-6))
    return [
        rounding_workload(
            "rounding of a sum", "7 fields of bond 16", total, 32, total.matricize()
        ),
        rounding_workload(
            "rounding of a sum",
            "f - f + 1e-6 u, bond 16",
            near_zero,
            32,
            1e-6 * u.matricize(),
        ),
    ]


def _operator_workloads(
    n_cores: int, laplacian: tt.TensorTrain
) -> list[dict[str, Any]]:
    """The Laplacian applied to fields."""
    workloads = []
    for rank in (32, 64):
        applied = laplacian @ multiscale_field(n_cores, 3, rank)
        workloads.append(
            rounding_workload(
                "operator times vector",
                f"Laplacian @ field of bond {rank}",
                applied,
                rank,
                applied.matricize(),
            )
        )
    return workloads


def _product_workloads(n_cores: int) -> list[dict[str, Any]]:
    """Element-wise products of fields, of random trains and of two-core trains."""
    workloads = [
        product_workload(
            f"fields of bond {rank}",
            multiscale_field(n_cores, 3, rank),
            multiscale_field(n_cores, 4, rank),
            cap,
        )
        for rank, cap in ((16, 32), (32, 32), (32, 64), (64, 64))
    ]
    workloads += [
        product_workload(
            f"random trains, 13 cores, bond {rank}",
            tt.rand([2] * 13, ranks=rank, seed=2),
            tt.rand([2] * 13, ranks=rank, seed=3),
            rank,
        )
        for rank in (8, 16)
    ]
    workloads.append(
        product_workload("smooth fields, 2 cores of 1024", *two_core_fields(1024), 16)
    )
    return workloads


def _long_train_workloads() -> list[dict[str, Any]]:
    """A true rank 8 stored with bond 24, close to max_rank plus the oversampling."""
    workloads = []
    for n_cores in (13, 20, 40, 60):
        x = tt.rand([2] * n_cores, ranks=8, seed=7)
        workloads.append(
            rounding_workload(
                "long trains",
                f"x + x + x, {n_cores} cores of bond 8",
                x.add(x).add(x),
                12,
                x.multiply(3),
            )
        )
    return workloads


def speed_workloads() -> list[dict[str, Any]]:
    """Workloads of the speed table.

    Fields are random Fourier series on 18 binary cores, except in the rows
    that name other inputs. Each workload has a direct and a randomized
    callable, the ``max_rank`` both use and the exact result as a dense array
    or as an unrounded train.
    """
    n_cores = 18
    laplacian = tridiagonal_qtt(n_cores, 2.0, -1.0)
    return (
        _transport_rate_workloads(n_cores, laplacian)
        + _sum_workloads(n_cores)
        + _operator_workloads(n_cores, laplacian)
        + _product_workloads(n_cores)
        + _long_train_workloads()
    )


def _relative_error(result: tt.TensorTrain, reference: Any) -> float:
    if isinstance(reference, tt.TensorTrain):
        return float((result - reference).tt_norm() / reference.tt_norm())
    return float(
        np.linalg.norm(result.matricize() - reference) / np.linalg.norm(reference)
    )


def _median_times(tasks: dict[str, Callable[[], Any]], rounds: int) -> dict[str, float]:
    """Median time of each task over ``rounds`` rounds that visit every task in turn."""
    batches = {}
    for key, task in tasks.items():
        start = time.perf_counter()
        task()
        batches[key] = max(1, math.ceil(BATCH_SECONDS / (time.perf_counter() - start)))
    samples: dict[str, list[float]] = {key: [] for key in tasks}
    # rounds outermost, so a drift of the machine reaches every task alike
    gc.disable()
    try:
        for _ in range(rounds):
            for key, task in tasks.items():
                start = time.perf_counter()
                for _ in range(batches[key]):
                    task()
                samples[key].append((time.perf_counter() - start) / batches[key])
            gc.collect()
    finally:
        gc.enable()
    return {key: statistics.median(values) for key, values in samples.items()}


def speed_rows(
    workloads: Optional[list[dict[str, Any]]] = None,
    rounds: int = DEFAULT_ROUNDS,
    seeds: int = DEFAULT_SEEDS,
) -> list[dict[str, Any]]:
    """Rows of the speed table, one per workload.

    The randomized error is taken over the seeds ``0, ..., seeds - 1``. Its
    time is taken with seed 0.
    """
    workloads = speed_workloads() if workloads is None else workloads
    rows, tasks = [], {}
    for index, workload in enumerate(workloads):
        logger.info("Speed table: %s, %s", workload["group"], workload["name"])
        direct = workload["direct"]()
        randomized = [workload["randomized"](seed) for seed in range(seeds)]
        errors = [
            _relative_error(result, workload["reference"]) for result in randomized
        ]
        rows.append(
            {
                "group": workload["group"],
                "workload": workload["name"],
                "input_bond": workload["input_bond"],
                "max_rank": workload["max_rank"],
                "error_direct": _relative_error(direct, workload["reference"]),
                "error_randomized_median": statistics.median(errors),
                "error_randomized_max": max(errors),
                "bond_direct": direct.max_rank,
                "bond_randomized_max": max(result.max_rank for result in randomized),
            }
        )
        tasks[f"{index} direct"] = workload["direct"]
        tasks[f"{index} randomized"] = lambda task=workload["randomized"]: task(0)
    logger.info("Speed table: timing %s rounds", rounds)
    times = _median_times(tasks, rounds)
    for index, row in enumerate(rows):
        row["direct_s"] = times[f"{index} direct"]
        row["randomized_s"] = times[f"{index} randomized"]
        row["speedup"] = row["direct_s"] / row["randomized_s"]
    return rows


# ------------------------------- reporting ----------------------------------


def _environment() -> dict[str, str]:
    """Versions and machine the report was produced with."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    config = np.show_config(mode="dicts")
    return {
        "commit": commit,
        "ttnumpy": tt.__version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "blas": config["Build Dependencies"]["blas"]["name"],
        "machine": f"{platform.system()} {platform.machine()}",
    }


def _bytes(value: float) -> str:
    units = ("B", "KB", "MB", "GB", "TB", "PB")
    power = 0
    while value >= 1024 and power < len(units) - 1:
        value /= 1024
        power += 1
    return f"{value:.3g} {units[power]}"


def _scientific(value: float, digits: int) -> str:
    """``value`` in scientific notation with an exponent of no leading zeros."""
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def _compression(value: float) -> str:
    return f"{value:.3g}" if value < 1e3 else _scientific(value, 2)


def _milliseconds(seconds: float) -> str:
    value = 1e3 * seconds
    decimals = 0 if value >= 100 else 1 if value >= 10 else 2
    return f"{value:.{decimals}f}"


def _column_widths(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> list[int]:
    return [max(len(cell) for cell in column) for column in zip(headers, *rows)]


def _markdown_table(
    headers: tuple[str, ...],
    rows: list[tuple[str, ...]],
    right: tuple[bool, ...],
    widths: Optional[list[int]] = None,
) -> list[str]:
    """Markdown table with every column padded to one width.

    Columns flagged in ``right`` are aligned to the right, the others to the
    left, so the table reads the same as plain text and as Markdown. The
    widths default to those of ``headers`` and ``rows``.
    """
    widths = _column_widths(headers, rows) if widths is None else widths

    def line(cells: tuple[str, ...]) -> str:
        padded = (
            cell.rjust(width) if flush else cell.ljust(width)
            for cell, width, flush in zip(cells, widths, right)
        )
        return "| " + " | ".join(padded) + " |"

    rule = (
        "-" * (width - 1) + ":" if flush else ":" + "-" * (width - 1)
        for width, flush in zip(widths, right)
    )
    return [line(headers), "| " + " | ".join(rule) + " |"] + [line(row) for row in rows]


def _paragraph(text: str) -> list[str]:
    return textwrap.wrap(text, width=88)


def _render_accuracy_markdown(rows: list[dict[str, Any]]) -> list[str]:
    frequencies = ", ".join(
        "(" + ", ".join(map(str, sines)) + ")" for sines in SINE_FREQUENCIES
    )
    lines = ["## Accuracy and memory: direct rounding against NumPy", ""]
    lines += _paragraph(
        "f, g, f1, ..., f4 are sums of four sines on a grid of 2^cores points "
        f"(frequencies {frequencies}), built as QTT trains. M is the finite "
        "element mass matrix tridiag(1, 4, 1) / 6. The row exp(f / 2) is the "
        "Taylor polynomial of degree 5 of exp(f / 2) by Horner's rule, rounded "
        "after each of its 10 operations. Every rounding requests a relative "
        f"error of {ACCURACY_ERROR:.0e}. The error is the relative root-mean-square "
        "difference from NumPy on all points or on a random sample of them."
    )
    operations = list(dict.fromkeys(row["operation"] for row in rows))
    table, previous = [], None
    for row in sorted(
        rows, key=lambda row: (operations.index(row["operation"]), row["cores"])
    ):
        points = row["points"]
        table.append(
            (
                row["operation"] if row["operation"] != previous else "",
                f"2^{row['cores']}",
                str(row["bond"]),
                _bytes(row["tt_bytes"]),
                _bytes(row["dense_bytes"]),
                _compression(row["compression"]),
                _scientific(row["error"], 1),
                points if points == "all" else f"{points} random",
            )
        )
        previous = row["operation"]
    headers = (
        "operation",
        "grid points",
        "bond",
        "TT memory",
        "dense memory",
        "compression",
        "relative error",
        "points",
    )
    right = (False, True, True, True, True, True, True, False)
    return lines + [""] + _markdown_table(headers, table, right)


def _render_speed_markdown(rows: list[dict[str, Any]], seeds: int) -> list[str]:
    lines = ["## Speed: direct against randomized rounding", ""]
    lines += _paragraph(
        f"Every rounding requests a relative error of {SPEED_ERROR:.0e} with the "
        "max_rank of its row. Times are medians over the rounds. The randomized "
        f"error is the median over {seeds} seeds, with the largest in parentheses. "
        "Bonds are those of the direct result and the largest over the seeds of "
        "the randomized one. The Laplacian is tridiag(-1, 2, -1), of bond 3."
    )
    headers = (
        "workload",
        "input bond",
        "max_rank",
        "direct ms",
        "randomized ms",
        "speed-up",
        "direct error",
        "randomized error",
        "bonds",
    )
    right = (False, True, True, True, True, True, True, True, True)
    tables: dict[str, list[tuple[str, ...]]] = {}
    for row in rows:
        tables.setdefault(row["group"], []).append(
            (
                row["workload"],
                str(row["input_bond"]),
                str(row["max_rank"]),
                _milliseconds(row["direct_s"]),
                _milliseconds(row["randomized_s"]),
                f"{row['speedup']:.2f}",
                _scientific(row["error_direct"], 1),
                f"{_scientific(row['error_randomized_median'], 1)} "
                f"({_scientific(row['error_randomized_max'], 1)})",
                f"{row['bond_direct']} / {row['bond_randomized_max']}",
            )
        )
    # one width per column across the groups, so their tables line up
    widths = _column_widths(
        headers, [row for table in tables.values() for row in table]
    )
    for group, table in tables.items():
        lines += ["", f"### {group}", ""]
        lines += _markdown_table(headers, table, right, widths)
    return lines


def render_arithmetic_markdown(report: dict[str, Any]) -> str:
    """Markdown report with the tables the run produced."""
    environment = report["environment"]
    lines = ["# Arithmetic benchmark results", ""]
    lines += _paragraph(
        f"ttnumpy {environment['ttnumpy']} (commit {environment['commit']}), "
        f"Python {environment['python']}, NumPy {environment['numpy']}, "
        f"SciPy {environment['scipy']}, BLAS {environment['blas']}, "
        f"{environment['machine']}."
    )
    if "accuracy" in report:
        lines += [""] + _render_accuracy_markdown(report["accuracy"])
    if "speed" in report:
        lines += [""] + _render_speed_markdown(
            report["speed"], report["settings"]["seeds"]
        )
    return "\n".join(lines) + "\n"


def write_arithmetic_report(
    report: dict[str, Any], output_dir: Path
) -> dict[str, Path]:
    """Write the report to JSON and Markdown files in ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{REPORT_STEM}.json"
    markdown_path = output_dir / f"{REPORT_STEM}.md"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(render_arithmetic_markdown(report), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}


# ---------------------------------- CLI -------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark the arithmetic of tensor trains: accuracy and "
        "memory against NumPy, speed of the direct and randomized rounding."
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        choices=TABLES,
        default=list(TABLES),
        help="Tables to produce (default: both)",
    )
    parser.add_argument(
        "--cores",
        nargs="+",
        type=int,
        default=list(DEFAULT_ACCURACY_CORES),
        help="Accuracy table: numbers of binary cores, the grids have 2**cores "
        "points (default: 16 20 24 30 40)",
    )
    parser.add_argument(
        "--dense-max-cores",
        type=int,
        default=DEFAULT_DENSE_MAX_CORES,
        help="Accuracy table: largest number of cores at which NumPy computes "
        "every entry. Above it the error is taken on random entries "
        f"(default: {DEFAULT_DENSE_MAX_CORES})",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help=f"Accuracy table: number of random entries (default: {DEFAULT_SAMPLES})",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=DEFAULT_ROUNDS,
        help=f"Speed table: timing rounds (default: {DEFAULT_ROUNDS})",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=DEFAULT_SEEDS,
        help="Speed table: seeds of the randomized method for its error "
        f"(default: {DEFAULT_SEEDS})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the JSON and Markdown reports are written",
    )
    parser.add_argument(
        "--log-level",
        default=DEFAULT_LOG_LEVEL,
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Verbosity of ttnumpy log output. INFO reports progress "
        f"(default: {DEFAULT_LOG_LEVEL}). Takes precedence over {LOG_LEVEL_ENV_VAR}.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Run the arithmetic benchmarks from the command line."""
    args = _build_parser().parse_args(argv)
    setup_logging(args.log_level)
    report: dict[str, Any] = {
        "environment": _environment(),
        "settings": {
            "tables": args.tables,
            "cores": args.cores,
            "dense_max_cores": args.dense_max_cores,
            "samples": args.samples,
            "rounds": args.rounds,
            "seeds": args.seeds,
            "accuracy_error": ACCURACY_ERROR,
            "speed_error": SPEED_ERROR,
        },
    }
    if "accuracy" in args.tables:
        report["accuracy"] = accuracy_rows(
            tuple(args.cores), args.dense_max_cores, args.samples
        )
    if "speed" in args.tables:
        report["speed"] = speed_rows(rounds=args.rounds, seeds=args.seeds)

    print(render_arithmetic_markdown(report))
    paths = write_arithmetic_report(report, args.output_dir)
    print(f"Wrote benchmark results to {paths['markdown']} and {paths['json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
