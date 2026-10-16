# Copyright (c) 2026 QBoard LLC. Licensed under the AGPLv3. See LICENSE.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Surrogate of a black-box simulator by tensor cross interpolation.

The black box solves the layered diffusion problem of parametric_diffusion.py,
here with ten layers, for one combination of the layer conductivities at a
time and returns the average of the solution. Tensor cross interpolation
learns this output over all 8^10 combinations from a small fraction of them.
The surrogate is checked against the simulator at random combinations, and its
mean and standard deviation against Monte Carlo.
"""

import numpy as np
import scipy.linalg

import ttnumpy as tt

n, n_layers, n_values = 127, 10, 8  # grid points, layers, values per parameter
h = 1.0 / (n + 1)
values = np.linspace(-1.0, 1.0, n_values)  # y_k; the conductivity is exp(y_k)
layer_of_edge = np.minimum(
    ((np.arange(n + 1) + 0.5) * h * n_layers).astype(int), n_layers - 1
)


def simulator(config):
    """Average of u for one combination of the layer conductivities."""
    a = np.exp(values[list(config)])[layer_of_edge] / h**2  # edge conductivities
    banded = np.zeros((3, n))
    banded[0, 1:] = -a[1:n]  # superdiagonal of the tridiagonal matrix
    banded[1] = a[:n] + a[1:]  # diagonal
    banded[2, :-1] = -a[1:n]  # subdiagonal
    return h * scipy.linalg.solve_banded((1, 1), banded, np.ones(n)).sum()


box = tt.BlackBox(simulator)  # counts the simulator calls
surrogate, info = tt.tci(
    box,
    [n_values] * n_layers,
    initial_pivots=[[0] * n_layers],
    error_threshold=1e-6,
    return_info=True,
)
train = surrogate.to_tensor_train()
count = n_values**n_layers
print(f"combinations     {count}")
print(f"simulator calls  {box.n_evals} ({100 * box.n_evals / count:.4f} %)")
print(f"largest rank     {max(surrogate.pivots.ranks)}")
print(f"half-sweeps      {info.num_half_sweeps_run}, converged: {info.converged}")

# the simulator at random combinations, called directly rather than through the box
rng = np.random.default_rng(0)
configs = rng.integers(0, n_values, size=(2000, n_layers))
reference = np.array([simulator(config) for config in configs])
predicted = np.array([train.get_element(list(config)) for config in configs])
error = np.max(np.abs(predicted - reference) / reference)
print(f"largest relative error at 2000 random combinations: {error:.1e}")


def inner(x, y):
    """<x|y>: x as a one-row tensor train matrix applied to y."""
    return x.transpose(conjugate=True).dot(y).get_full_tensor()


ones = tt.TensorTrain([np.ones((1, n_values, 1))] * n_layers)
mean = inner(train, ones) / count
std = np.sqrt(inner(train, train) / count - mean**2)
mc_mean, mc_std = reference.mean(), reference.std(ddof=1)
mc_error = mc_std / np.sqrt(reference.size)
print()
print(f"{'average of u':26} {'mean':>12} {'std':>12}")
print(f"{'surrogate, all 8^10':26} {mean:12.8f} {std:12.8f}")
print(
    f"{'Monte Carlo, 2000 calls':26} {mc_mean:12.8f} {mc_std:12.8f}   +- {mc_error:.1e} on the mean"
)

assert error < 1e-5
assert abs(mean - mc_mean) < 4 * mc_error
