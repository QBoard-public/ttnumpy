# Copyright (c) 2026 QBoard LLC. Licensed under the AGPLv3. See LICENSE.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the arithmetic benchmark: its exact inputs and one small run."""

import json
from unittest import mock

import numpy as np

import ttnumpy as tt
from benchmarks import arithmetic


def test_inputs_match_numpy():
    """The QTT sines, the tridiagonal operator and the sampler agree with NumPy."""
    n_cores = 10
    waves = arithmetic.sine_waves((1, 3, 7, 13), seed=0)
    train = arithmetic.sine_sum_qtt(n_cores, waves)
    values = arithmetic.sine_sum_values(n_cores, waves, np.arange(2**n_cores))
    indices = np.random.default_rng(0).integers(0, 2**n_cores, 50)
    size = 2**6
    operator = 4 * np.eye(size) + np.eye(size, k=1) + np.eye(size, k=-1)

    np.testing.assert_allclose(train.matricize(), values, atol=1e-12)
    np.testing.assert_allclose(
        arithmetic.sample_qtt(train, indices), values[indices], atol=1e-12
    )
    np.testing.assert_allclose(
        arithmetic.tridiagonal_qtt(6, 4.0, 1.0).matricize(), operator, atol=1e-14
    )


def test_main_writes_both_tables(tmp_path):
    """One small run writes both tables to JSON and Markdown."""

    def small_workloads():
        x = tt.rand([2] * 6, ranks=3, seed=0)
        y = tt.rand([2] * 6, ranks=3, seed=1)
        built = x.add(y)
        return [
            arithmetic.rounding_workload("sum", "x + y", built, 4, built.matricize()),
            arithmetic.product_workload("x * y", x, y, 4),
        ]

    with mock.patch.object(arithmetic, "speed_workloads", small_workloads):
        status = arithmetic.main(
            ["--cores", "8", "10", "--dense-max-cores", "8", "--samples", "100"]
            + ["--rounds", "1", "--seeds", "2", "--log-level", "ERROR"]
            + ["--output-dir", str(tmp_path)]
        )

    report = json.loads((tmp_path / "arithmetic_benchmark.json").read_text())
    markdown = (tmp_path / "arithmetic_benchmark.md").read_text()
    products = [row for row in report["accuracy"] if row["operation"] == "f * g"]
    assert status == 0
    assert len(report["accuracy"]) == 8
    assert [row["points"] for row in products] == ["all", 100]
    assert max(row["error"] for row in report["accuracy"]) < 1e-9
    assert [row["workload"] for row in report["speed"]] == ["x + y", "x * y"]
    assert "## Accuracy and memory" in markdown
    assert "## Speed" in markdown
