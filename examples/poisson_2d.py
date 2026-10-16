# Copyright (c) 2026 QBoard LLC. Licensed under the AGPLv3. See LICENSE.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Poisson equation on the unit square in quantized tensor train format.

Solves -(u_xx + u_yy) = 2 pi^2 sin(pi x) sin(pi y) with zero boundary values
by five-point finite differences on grids of 2^N x 2^N interior points, with
the operator written down from its cores. Compares the error against the exact
solution with the closed-form discretization error on every grid, then
compares the three solvers on one grid.
"""

import logging

import numpy as np

import ttnumpy as tt

# the tables report when a solver stagnates
logging.getLogger("ttnumpy").setLevel(logging.ERROR)

I = np.eye(2)
J = np.array([[0.0, 1.0], [0.0, 0.0]])  # one bit of i -> i + 1, no carry
O = np.zeros((2, 2))


def ttm_core(blocks):
    """Tensor train matrix core from rows of 2 x 2 blocks, one per pair of bonds."""
    core = np.zeros((len(blocks), 2, 2, len(blocks[0])))
    for left, row in enumerate(blocks):
        for right, block in enumerate(row):
            core[left, :, :, right] = block
    return core


def laplacian_1d(n_bits, h):
    """-d^2/dx^2 on 2^n_bits interior points with zero ends, most significant bit first.

    The bond index is 0 without a carry, 1 while the shift i -> i + 1 carries
    and 2 while the shift i -> i - 1 carries.
    """
    first = ttm_core([[I, J, J.T]])
    middle = ttm_core([[I, J, J.T], [O, J.T, O], [O, O, J]])
    last = ttm_core([[2 * I - J - J.T], [-J.T], [-J]]) / h**2
    return tt.TensorTrain([first] + [middle] * (n_bits - 2) + [last])


def kron(a, b):
    """Kronecker product of two trains: the cores of a followed by those of b."""
    return tt.TensorTrain(a.cores + b.cores)


def poisson(n_bits):
    """Operator, right-hand side, exact solution and closed-form error."""
    size = 2**n_bits
    h = 1.0 / (size + 1)
    x = h * np.arange(1, size + 1)
    laplacian = laplacian_1d(n_bits, h)
    identity = tt.TensorTrain([I.reshape((1, 2, 2, 1))] * n_bits)
    A = kron(laplacian, identity) + kron(identity, laplacian)
    sine = tt.tt_svd(np.sin(np.pi * x).reshape([2] * n_bits), error=1e-14)
    exact = kron(sine, sine)
    # sin(pi x) sin(pi y) is an eigenvector of A with this eigenvalue
    eigenvalue = 8 / h**2 * np.sin(np.pi * h / 2) ** 2
    return A, exact * (2 * np.pi**2), exact, abs(2 * np.pi**2 / eigenvalue - 1)


def relative_error(x, exact):
    """||x - exact|| / ||exact||, in tensor train arithmetic."""
    return (x - exact).tt_norm() / exact.tt_norm()


print("AMEn on grids of 2^N x 2^N points")
print(
    f"{'N':>3} {'unknowns':>10} {'rank':>5} {'sweeps':>7} {'residual':>9} "
    f"{'error':>10} {'predicted':>10}  stop"
)
for N in (4, 6, 8, 10, 12, 14, 15):
    operator, rhs, u_exact, predicted = poisson(N)
    u, info = tt.amen(
        operator,
        rhs,
        residual_tol=1e-10,
        error_threshold=1e-12,
        max_sweeps=20,
        symmetric=True,
        return_info=True,
        seed=0,
    )
    error = relative_error(u, u_exact)
    rank = max(u.truncate(error=1e-10).bonds)
    print(
        f"{N:3d} {4.0**N:10.2e} {rank:5d} {info.num_sweeps_run:7d} "
        f"{info.residuals[-1]:9.1e} {error:10.3e} {predicted:10.3e}  {info.stop_reason}"
    )
    assert rank == 2
    if N <= 12:
        assert abs(error - predicted) < 0.01 * predicted

print()
print("The three solvers on 2^10 x 2^10 points")
operator, rhs, u_exact, predicted = poisson(10)
solvers = {
    "amen": (tt.amen, {"error_threshold": 1e-12, "seed": 0}),
    "mals": (tt.mals, {}),
    "als, x0 with the ranks of f": (tt.als, {}),
    "als, x0 of rank 1": (tt.als, {"x0": tt.ones(rhs.physical_dims())}),
}
print(f"{'solver':28} {'sweeps':>7} {'error':>10}  stop")
for name, (solver, options) in solvers.items():
    u, info = solver(
        operator,
        rhs,
        residual_tol=1e-10,
        max_sweeps=20,
        symmetric=True,
        return_info=True,
        **options,
    )
    error = relative_error(u, u_exact)
    print(f"{name:28} {info.num_sweeps_run:7d} {error:10.3e}  {info.stop_reason}")
    if name == "als, x0 of rank 1":
        assert error > 0.5
    else:
        assert abs(error - predicted) < 0.01 * predicted
