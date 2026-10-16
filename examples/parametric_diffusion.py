# Copyright (c) 2026 QBoard LLC. Licensed under the AGPLv3. See LICENSE.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Diffusion through a layered medium whose layer conductivities are uncertain.

Solves -(a(x; y) u')' = 1 on (0, 1) with zero boundary values, where the
conductivity is exp(y_k) in layer k and each y_k takes one of eight values,
for all 8^6 parameter combinations at once. Checks the solution against
direct solves at random combinations, then compares the mean and standard
deviation of the average of u, computed exactly from the tensor train, with
a Monte Carlo estimate from direct solves.
"""

import logging
from functools import reduce

import numpy as np

import ttnumpy as tt

# the solver output below reports how the run stopped
logging.getLogger("ttnumpy").setLevel(logging.ERROR)

n, n_layers, n_values = 127, 6, 8  # grid points, layers, values per parameter
h = 1.0 / (n + 1)
values = np.linspace(-1.0, 1.0, n_values)  # y_k; the conductivity is exp(y_k)

# conductivities live on the n + 1 cell edges, the layer of each edge by its midpoint
layer_of_edge = np.minimum(
    ((np.arange(n + 1) + 0.5) * h * n_layers).astype(int), n_layers - 1
)
D = np.eye(n + 1, n) - np.eye(n + 1, n, k=-1)  # differences u_j - u_{j-1}


def stiffness(conductivity_of_edge):
    """Finite-difference matrix of -(a u')' for the given edge conductivities."""
    return D.T @ np.diag(conductivity_of_edge) @ D / h**2


def direct_solve(config):
    """u for one parameter combination, one index into values per layer."""
    return np.linalg.solve(stiffness(np.exp(values[config])[layer_of_edge]), np.ones(n))


# A(y) = sum_k exp(y_k) A_k, and over every combination at once
# A = sum_k A_k x I x ... x E x ... x I with E = diag(exp(values)) at site k
E, I = np.diag(np.exp(values)), np.eye(n_values)
terms = (
    tt.TensorTrain(
        [stiffness((layer_of_edge == k).astype(float)).reshape((1, n, n, 1))]
        + [
            (E if j == k else I).reshape((1, n_values, n_values, 1))
            for j in range(n_layers)
        ]
    )
    for k in range(n_layers)
)
A = reduce(lambda a, b: a + b, terms)
f = tt.TensorTrain([np.ones((1, n, 1))] + [np.ones((1, n_values, 1))] * n_layers)

u, info = tt.amen(
    A,
    f,
    residual_tol=1e-6,
    error_threshold=1e-8,
    max_sweeps=20,
    symmetric=True,
    return_info=True,
    seed=0,
)
print(f"combinations {n_values**n_layers}, unknowns {n * n_values**n_layers}")
print(f"operator bonds   {A.bonds}")
print(f"solution bonds   {u.bonds}")
print(
    f"AMEn: {info.stop_reason} after {info.num_sweeps_run} sweeps, relative residual {info.residuals[-1]:.1e}"
)
print(
    f"numbers stored   {sum(core.size for core in u.cores)} instead of {n * n_values**n_layers}"
)


def solution_at(config):
    """u for one combination, read from the train by fixing every parameter site."""
    vector = u.cores[0][0, :, 0, :]
    for site, index in enumerate(config, start=1):
        vector = vector @ u.cores[site][:, index, 0, :]
    return vector[:, 0]


rng = np.random.default_rng(0)
errors = []
for combination in rng.integers(0, n_values, size=(50, n_layers)):
    reference = direct_solve(combination)
    errors.append(
        np.abs(solution_at(combination) - reference).max() / np.abs(reference).max()
    )
print(f"largest relative error at 50 random combinations: {max(errors):.1e}")


def inner(x, y):
    """<x|y>: x as a one-row tensor train matrix applied to y."""
    return x.transpose(conjugate=True).dot(y).get_full_tensor()


# Q(y) = h * sum_i u_i(y), the average of u, as a train over the parameters only
weighted = np.einsum("i,aib->ab", np.full(n, h), u.cores[0][:, :, 0, :])
cores = [core[:, :, 0, :] for core in u.cores[1:]]
cores[0] = np.einsum("ab,bjc->ajc", weighted, cores[0])
Q = tt.TensorTrain(cores)
ones = tt.TensorTrain([np.ones((1, n_values, 1))] * n_layers)
count = n_values**n_layers
mean = inner(Q, ones) / count
std = np.sqrt(inner(Q, Q) / count - mean**2)

samples = np.array(
    [
        h * direct_solve(combination).sum()
        for combination in rng.integers(0, n_values, size=(2000, n_layers))
    ]
)
mc_mean, mc_std = samples.mean(), samples.std(ddof=1)
mc_error = mc_std / np.sqrt(samples.size)
print()
print(f"{'average of u':28} {'mean':>12} {'std':>12}")
print(f"{'tensor train, all ' + str(count):28} {mean:12.8f} {std:12.8f}")
print(
    f"{'Monte Carlo, 2000 samples':28} {mc_mean:12.8f} {mc_std:12.8f}   +- {mc_error:.1e} on the mean"
)

assert max(errors) < 1e-5
assert abs(mean - mc_mean) < 4 * mc_error
