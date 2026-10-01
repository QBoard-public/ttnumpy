# Copyright (c) 2026 QBoard LLC. Licensed under the AGPLv3. See LICENSE.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Interpolation of black-box functions and tensors into tensor trains.

Tensor cross interpolation builds a tensor train from a small number of
sampled entries, in contrast to the SVD-based constructors in
:mod:`ttnumpy.tensor_train`, which consume the whole tensor.
"""
