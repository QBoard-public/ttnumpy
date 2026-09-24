# Copyright (c) 2026 QBoard LLC. Licensed under the AGPLv3. See LICENSE.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared fixtures for the tensor-train solver tests."""

import numpy as np

from ttnumpy import tensor_train


def qtt_laplacian_1d(order):
    """Exact QTT of the 1D Dirichlet Laplacian with all-ones rhs.

    Ill-conditioned (cond ~ 0.4 * 4**order), but the solution is a parabola
    with exact QTT rank 3, so it exercises accuracy control cheaply. Returns
    the operator TTM, the right-hand side TT and a rank-3 initial guess.
    """
    size = 2**order
    h = 1.0 / (size + 1)
    laplacian = (
        np.diag(-2.0 * np.ones(size))
        + np.diag(np.ones(size - 1), 1)
        + np.diag(np.ones(size - 1), -1)
    ) / h**2
    laplacian_tt = tensor_train.tt_svd(
        laplacian.reshape([2] * 2 * order),
        physical_indexes=[(2, 2)] * order,
        ttm=True,
        error=0,
    )
    rhs_tt = tensor_train.ones([2] * order)
    initial_tt = tensor_train.ones([2] * order, ranks=3).orthonormalize(
        canonical="right"
    )
    return laplacian_tt, rhs_tt, initial_tt
