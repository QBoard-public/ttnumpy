# Copyright (c) 2026 QBoard LLC. Licensed under the AGPLv3. See LICENSE.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared fixtures for the tensor cross interpolation tests."""

import numpy as np

from ttnumpy.interpolation import black_box

#: One dimension per site of :func:`rational`.
RATIONAL_DIMS = (6, 6, 6, 6, 6)


def rational(indices) -> float:
    """1 / (1 + 10 (x1 + ... + x5)) on six nodes in [0, 1] per site."""
    nodes = np.linspace(0.0, 1.0, 6)
    return 1.0 / (1.0 + 10.0 * sum(nodes[i] for i in indices))


def dense(function, dims) -> np.ndarray:
    """Every element of the tensor behind ``function``."""
    return np.array([function(index) for index in np.ndindex(*dims)]).reshape(dims)


def dense_box(tensor: np.ndarray) -> black_box.BlackBox:
    """Sample a dense tensor by index."""
    return black_box.BlackBox(lambda indices: tensor[tuple(indices)])
