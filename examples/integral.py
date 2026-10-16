# Copyright (c) 2026 QBoard LLC. Licensed under the AGPLv3. See LICENSE.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A high-dimensional integral by tensor cross interpolation.

Integrates f(x) = 2^d / (1 + 2 (x_1 + ... + x_d)) over the unit cube in 10
and 20 dimensions, interpolating f on Gauss-Legendre nodes and contracting the
train with the quadrature weights. Compares the result with the closed form
and with Monte Carlo at the same number of evaluations of f.
"""

from decimal import Decimal, localcontext
from math import comb, factorial

import numpy as np

import ttnumpy as tt

nodes, weights = np.polynomial.legendre.leggauss(8)
nodes, weights = (nodes + 1) / 2, weights / 2  # moved from [-1, 1] to [0, 1]


def exact_integral(dim):
    """Closed form; the alternating sum cancels heavily, so it runs in 50 digits."""
    with localcontext() as context:
        context.prec = 50
        total = sum(
            (-1) ** (dim - k)
            * comb(dim, k)
            * Decimal(1 + 2 * k) ** (dim - 1)
            * Decimal(1 + 2 * k).ln()
            for k in range(dim + 1)
        )
        return float(total / factorial(dim - 1))


def integrand(dim):
    """f at the Gauss-Legendre nodes, one index per variable."""
    return lambda indices: 2.0**dim / (1.0 + 2.0 * sum(nodes[i] for i in indices))


def integrate(train):
    """Quadrature of a train: each core contracted with the weights, then multiplied."""
    vector = np.ones(1)
    for core in train.cores:
        vector = vector @ np.einsum("aib,i->ab", core[:, :, 0, :], weights)
    return vector[0]


rng = np.random.default_rng(0)
print(
    f"{'d':>3} {'threshold':>9} {'evaluations':>12} {'rank':>5} {'TCI error':>10} {'MC error':>10} {'MC sigma':>10}"
)
for d in (10, 20):
    exact = exact_integral(d)
    for threshold in (1e-4, 1e-7, 1e-10):
        tensor_ci = tt.tci(
            integrand(d), [8] * d, initial_pivots=[[0] * d], error_threshold=threshold
        )
        tci_error = abs(integrate(tensor_ci.to_tensor_train()) - exact) / exact

        # Monte Carlo with as many evaluations as TCI used
        calls = tensor_ci.box.n_evals
        samples = 2.0**d / (1.0 + 2.0 * rng.random((calls, d)).sum(axis=1))
        mc_error = abs(samples.mean() - exact) / exact
        mc_sigma = samples.std(ddof=1) / np.sqrt(calls) / exact

        rank = max(tensor_ci.pivots.ranks)
        print(
            f"{d:3d} {threshold:9.0e} {calls:12d} {rank:5d} {tci_error:10.1e} {mc_error:10.1e} {mc_sigma:10.1e}"
        )
        assert tci_error < 10 * threshold
