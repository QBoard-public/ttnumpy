# Copyright (c) 2026 QBoard LLC. Licensed under the AGPLv3. See LICENSE.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Sampling wrapper for the function that tensor cross interpolation reads.

Neighbouring bonds share pivots, so a sweep requests the same configuration
several times; the wrapper evaluates each one once and counts the distinct
evaluations.

What is sampled is addressed by indices, not by coordinates. A function of
real arguments reaches TCI through a grid supplied by the caller::

    nodes = np.linspace(-1.0, 1.0, 64)
    box = BlackBox(lambda indices: f(*(nodes[i] for i in indices)))
"""

import operator
from typing import Callable, Dict, Sequence, Tuple

import numpy as np

from ttnumpy import tensor_train

#: One index per tensor axis, addressing a single element.
Configuration = Sequence[int]

#: Shape of the function TCI samples, which takes one configuration and
#: returns the value there.
BlackBoxFunction = Callable[[Tuple[int, ...]], complex]


class BlackBox:
    """A function of discrete indices, memoized and counted.

    The wrapped function is called with a tuple of Python ints, whatever
    sequence type the caller passed.
    """

    def __init__(self, element_function: BlackBoxFunction):
        """Wrap a function of indices, called at most once per configuration.

        Args:
            element_function: Returns the tensor element at a configuration
                of indices.
        """
        self._element_function = element_function
        self._cache: Dict[Tuple[int, ...], complex] = {}
        self._n_requests = 0

    def __call__(self, indices: Configuration) -> complex:
        """Value of the wrapped function at ``indices``.

        Args:
            indices: One index per axis, as any sequence of integers.

        Raises:
            TypeError: If an index is not an integer.
            ValueError: If the wrapped function returns a value that is not
                finite. The value is not cached.

        Returns:
            complex: The tensor element at ``indices``.
        """
        key = tuple(map(operator.index, indices))
        self._n_requests += 1
        if key not in self._cache:
            value = self._element_function(key)
            if not np.isfinite(value):
                raise ValueError(
                    f"element_function returned {value} at {key}; "
                    f"values must be finite."
                )
            self._cache[key] = value
        return self._cache[key]

    def sample(
        self, rows: Sequence[Tuple[int, ...]], columns: Sequence[Tuple[int, ...]]
    ) -> np.ndarray:
        """Values at every row multi-index followed by every column multi-index.

        Costs ``len(rows) * len(columns)`` requests.

        Args:
            rows: Multi-indices covering the leading sites.
            columns: Multi-indices covering the trailing sites.

        Raises:
            TypeError: If an index is not an integer.
            ValueError: If the wrapped function returns a value that is not
                finite.

        Returns:
            np.ndarray: Matrix of shape ``(len(rows), len(columns))`` whose
            entry ``[r, c]`` is the value at ``rows[r] + columns[c]``.
        """
        return np.array([[self(row + column) for column in columns] for row in rows])

    @property
    def max_modulus(self) -> float:
        """Largest modulus among the values evaluated so far, 0.0 before any."""
        return float(max(map(abs, self._cache.values()), default=0.0))

    @property
    def n_evals(self) -> int:
        """Configurations passed to the wrapped function."""
        return len(self._cache)

    @property
    def n_requests(self) -> int:
        """Configurations asked for, whether evaluated or cached."""
        return self._n_requests

    @property
    def n_hits(self) -> int:
        """Requests served from the cache."""
        return self._n_requests - len(self._cache)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._element_function!r}, "
            f"n_evals={self.n_evals}, n_requests={self.n_requests})"
        )


def tt_as_black_box(train: tensor_train.TensorTrain) -> BlackBox:
    """Wrap a tensor train so that TCI can sample it like any other function.

    Args:
        train: Tensor train in TT format.

    Raises:
        WrongTTType: If the train is in tensor train matrix format, whose
            elements are addressed by index pairs rather than single indices.

    Returns:
        BlackBox: Sampler over the elements of ``train``.
    """
    if train.is_ttm():
        raise tensor_train.WrongTTType(
            "Tensor cross interpolation samples a tensor train, "
            "not a tensor train matrix."
        )
    return BlackBox(train.get_element)
