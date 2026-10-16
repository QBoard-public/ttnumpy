# Copyright (c) 2026 QBoard LLC. Licensed under the AGPLv3. See LICENSE.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""GHZ and W states as tensor trains, and two operators acting on them.

Prints the cores of the GHZ state on two qubits. On ten qubits it builds the
GHZ state from its cores and both states from sums of product states, the
Ising chain by TT-SVD of its dense matrix and the global spin flip from its
cores, then checks the results against their closed forms.
"""

from functools import reduce

import numpy as np

import ttnumpy as tt

ket0, ket1 = np.array([1.0, 0.0]), np.array([0.0, 1.0])


def product_state(bits):
    """Rank-1 tensor train of the basis state |bits>."""
    return tt.TensorTrain([(ket1 if bit else ket0).reshape((1, 2, 1)) for bit in bits])


def state_sum(terms):
    """Sum of the basis states listed in terms, one tuple of bits each."""
    return reduce(lambda x, y: x + y, (product_state(bits) for bits in terms))


def inner(x, y):
    """<x|y>: x as a one-row tensor train matrix applied to y."""
    return x.transpose(conjugate=True).dot(y).get_full_tensor()


def ghz_from_cores(n_qubits):
    """GHZ state; the bond index records whether the qubits so far are 0 or 1."""
    first = np.eye(2).reshape((1, 2, 2))
    middle = np.zeros((2, 2, 2))
    middle[0, 0, 0] = middle[1, 1, 1] = 1.0
    last = np.eye(2).reshape((2, 2, 1)) / np.sqrt(2)
    return tt.TensorTrain([first] + [middle] * (n_qubits - 2) + [last])


# GHZ on two qubits: one matrix G_s[i] per core s and index value i
for s, core in enumerate(ghz_from_cores(2).cores):
    for i in (0, 1):
        print(f"G_{s}[{i}] = {core[:, i, 0, :].round(4).tolist()}")
print()

d = 10  # number of qubits from here on
ghz = ghz_from_cores(d)

# GHZ = (|00...0> + |11...1>) / sqrt(2): ranks add, 1 + 1 = 2
ghz_sum = state_sum([(0,) * d, (1,) * d]) * (1 / np.sqrt(2))

# W = (|10...0> + |01...0> + ... + |00...1>) / sqrt(d): ranks add up to d
w = state_sum([tuple(int(j == k) for j in range(d)) for k in range(d)])
w = w * (1 / np.sqrt(d))
w_bonds = w.bonds
w = w.truncate(error=1e-12)

# Ising chain H = -sum_s Z_s Z_{s+1}: TT-SVD of its dense 2^d x 2^d matrix
Z, I = np.diag([1.0, -1.0]), np.eye(2)
dense = -sum(
    reduce(np.kron, [Z if j in (s, s + 1) else I for j in range(d)])
    for s in range(d - 1)
)
H = tt.tt_svd(dense.reshape([2] * (2 * d)), physical_indexes=[(2, 2)] * d, ttm=True)

# global spin flip X x X x ... x X from its cores, a rank-1 tensor train matrix
X = np.array([[0.0, 1.0], [1.0, 0.0]])
flip = tt.TensorTrain([X.reshape((1, 2, 2, 1))] * d)

energy = -(d - 1)
H_ghz = H @ ghz
checks = {
    "|| GHZ from cores - GHZ as a sum ||": ((ghz - ghz_sum).tt_norm(), 0.0),
    "<GHZ|GHZ>": (inner(ghz, ghz), 1.0),
    "<W|W>": (inner(w, w), 1.0),
    "<GHZ|W>": (inner(ghz, w), 0.0),
    "|| H GHZ - E GHZ ||": ((H_ghz - ghz * energy).tt_norm(), 0.0),
    "<GHZ|H|GHZ>": (inner(ghz, H_ghz), energy),
    "<W|H|W>": (inner(w, H @ w), -(d - 1) * (d - 4) / d),
    "|| flip GHZ - GHZ ||": ((flip @ ghz - ghz).tt_norm(), 0.0),
}

bonds = {
    "GHZ from cores": ghz.bonds,
    "W as a sum": w_bonds,
    "W after rounding": w.bonds,
    "H from TT-SVD": H.bonds,
    "H GHZ": H_ghz.bonds,
    "H GHZ after rounding": H_ghz.truncate(error=1e-12).bonds,
}
for name, value in bonds.items():
    print(f"bonds of {name:22} {value}")
print()
print(f"{'quantity':36} {'computed':>18} {'exact':>10}")
for name, (computed, exact) in checks.items():
    print(f"{name:36} {computed:18.12f} {exact:10.4f}")
    assert abs(computed - exact) < 1e-12, name
