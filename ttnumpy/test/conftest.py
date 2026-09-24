# Copyright (c) 2026 QBoard LLC. Licensed under the AGPLv3. See LICENSE.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pytest configuration shared by all tests under this package."""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import numpy as np
import pytest

from ttnumpy import tensor_train
from ttnumpy.solvers.als import als
from ttnumpy.solvers.amen import amen
from ttnumpy.solvers.mals import mals
from solver_test_utils import qtt_laplacian_1d


def pytest_report_header(config: pytest.Config) -> str:
    """Extra line at the start of the pytest run (visible in CI logs)."""
    return f"ttnumpy tests | rootdir: {config.rootpath}"


_SOLVER_EXTRAS: dict[str, dict[str, Any]] = {
    "als": {},
    "mals": {"error_threshold": 1e-10},
    "amen": {"error_threshold": 1e-10, "seed": 3},
}

_SOLVER_FUNCTIONS: dict[str, Callable] = {"als": als, "mals": mals, "amen": amen}


class Solver:
    """One TT linear solver, callable with only the shared parameters."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._func = _SOLVER_FUNCTIONS[name]
        self._extras = _SOLVER_EXTRAS[name]

    def __call__(self, A, b, **kwargs):
        """Solve ``A x = b``; ``kwargs`` override the solver-specific extras."""
        return self._func(A, b, **{**self._extras, **kwargs})

    def __repr__(self) -> str:
        return f"<solver {self.name}>"


@pytest.fixture(params=sorted(_SOLVER_FUNCTIONS))
def solver(request: pytest.FixtureRequest) -> Solver:
    """Each TT linear solver in turn, so one test body runs three times.

    Note this cannot be requested from a ``unittest.TestCase`` method — pytest
    does not inject fixtures there. Use it from plain test functions.
    """
    return Solver(request.param)


class LaplacianSystem(NamedTuple):
    """A QTT linear system together with its exact discrete solution."""

    #: Operator in TTM format.
    A: tensor_train.TensorTrain
    #: Right-hand side in TT format.
    b: tensor_train.TensorTrain
    #: Initial guess carrying enough rank for every solver (see the fixture).
    x0: tensor_train.TensorTrain
    #: Dense solution of the same system, as a flat vector.
    exact: np.ndarray


@pytest.fixture
def laplacian_system() -> LaplacianSystem:
    """One ill-conditioned system every solver is held to."""
    order = 6
    A, b, _ = qtt_laplacian_1d(order)
    dense = np.asarray(A.get_full_tensor()).reshape(2**order, 2**order)
    exact = np.linalg.solve(dense, np.asarray(b.get_full_tensor()).reshape(-1))
    x0 = tensor_train.ones([2] * order, ranks=4).orthonormalize(canonical="right")
    return LaplacianSystem(A=A, b=b, x0=x0, exact=exact)
