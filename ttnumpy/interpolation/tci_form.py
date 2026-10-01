# Copyright (c) 2026 QBoard LLC. Licensed under the AGPLv3. See LICENSE.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""TCI form of a tensor train: pivot lists and the tensor they index.

A TCI form is fixed by two kinds of multi-indices: the row pivots, each
naming a configuration of the sites left of a bond, and the column pivots,
each naming a configuration of the sites right of it.

Pivots are addressed by bond rather than by the subscripts of SciPost Phys.
18, 104 (2025). Bond ``b`` lies between site ``b - 1`` and site ``b``, so
``left(b)`` covers sites ``0 .. b-1`` and ``right(b)`` covers sites
``b .. n_sites-1``. In the notation of the paper, ``left(b)`` is ``I_b`` and
``right(b)`` is ``J_{b+1}``.

The form writes the tensor as ``T_0 P_1^-1 T_1 P_2^-1 ... P_{n-1}^-1 T_{n-1}``.
``T_s`` samples the tensor with site ``s`` free between ``left(s)`` and
``right(s + 1)``, and ``P_b`` samples it at the pivots of bond ``b``.

Sites are numbered from 0 and site indices run ``0 .. d-1``, where the paper
counts both from 1. Indices are positions rather than values. ``T_s`` is the
paper's ``T_{s+1}``, and ``P_b`` is its ``P_b``.

Reference:
    Y. Núñez Fernández et al., Learning tensor networks with tensor cross
    interpolation: new algorithms and libraries, SciPost Phys. 18, 104
    (2025), Secs. 4.1 and 4.2.
"""

import operator
from typing import List, Sequence, Tuple

import numpy as np

from ttnumpy import tensor_train
from ttnumpy.interpolation.black_box import BlackBox

#: A configuration of some contiguous run of sites, one index per site.
MultiIndex = Tuple[int, ...]


def split_configuration(
    configuration: Sequence[int],
) -> Tuple[List[MultiIndex], List[MultiIndex]]:
    """Cut a complete configuration at every bond.

    Args:
        configuration: One index per site.

    Raises:
        TypeError: If an index is not an integer.

    Returns:
        tuple: The left halves and the right halves, both of length
        ``len(configuration) + 1``, such that ``left[b] + right[b]`` is the
        configuration for every bond ``b``.
    """
    indices = tuple(map(operator.index, configuration))
    bonds = range(len(indices) + 1)
    return [indices[:bond] for bond in bonds], [indices[bond:] for bond in bonds]


def append_site(indices: Sequence[MultiIndex], dim: int) -> List[MultiIndex]:
    """Extend each multi-index by every value of one further site, at the end.

    Args:
        indices: Multi-indices covering the sites to the left.
        dim: Number of values the appended site takes.

    Returns:
        list: ``len(indices) * dim`` multi-indices, each one longer.
    """
    return [index + (value,) for index in indices for value in range(dim)]


def prepend_site(indices: Sequence[MultiIndex], dim: int) -> List[MultiIndex]:
    """Extend each multi-index by every value of one further site, at the front.

    Args:
        indices: Multi-indices covering the sites to the right.
        dim: Number of values the prepended site takes.

    Returns:
        list: ``dim * len(indices)`` multi-indices, each one longer.
    """
    return [(value,) + index for value in range(dim) for index in indices]


class PivotLists:
    """Row and column pivot multi-indices of a TCI form, addressed by bond.

    ``left(b)`` and ``right(b)`` are the two halves of the cut at bond ``b``.
    The internal bonds ``1 .. n_sites-1`` hold equal numbers of each and are
    changed only through :meth:`replace`. The outer bonds are the dummy bonds
    of the train: ``left(0)`` and ``right(n_sites)`` are the single empty
    multi-index, and ``right(0)`` and ``left(n_sites)`` are not defined.
    """

    def __init__(
        self, bonds: Sequence[Tuple[Sequence[MultiIndex], Sequence[MultiIndex]]]
    ):
        """Build pivot lists from the pivots of each internal bond.

        Args:
            bonds: One ``(left, right)`` pair per internal bond, in bond order,
                so the first pair belongs to bond 1.

        Raises:
            TypeError: If an index is not an integer.
            ValueError: If a bond fails the checks of :meth:`replace`.
        """
        self._n_sites = len(bonds) + 1
        self._left: List[Tuple[MultiIndex, ...]] = []
        self._right: List[Tuple[MultiIndex, ...]] = []
        for bond, (left, right) in enumerate(bonds, start=1):
            left, right = self._checked(bond, left, right)
            self._left.append(left)
            self._right.append(right)

    def _checked(
        self, bond: int, left: Sequence[MultiIndex], right: Sequence[MultiIndex]
    ) -> Tuple[Tuple[MultiIndex, ...], Tuple[MultiIndex, ...]]:
        """Normalize the pivots of one bond to tuples of ints and check them.

        Raises:
            TypeError: If an index is not an integer.
            ValueError: If the bond is empty, holds different numbers of row
                and column pivots, has a multi-index that does not span the
                sites its bond implies, lists a pivot twice, or has a negative
                index.
        """
        left = tuple(tuple(map(operator.index, index)) for index in left)
        right = tuple(tuple(map(operator.index, index)) for index in right)
        if not left or not right:
            raise ValueError(f"bond {bond} needs at least one pivot on each side.")
        if len(left) != len(right):
            raise ValueError(
                f"bond {bond} has {len(left)} row pivots and {len(right)} "
                f"column pivots; the pivot matrix must be square."
            )
        if any(len(index) != bond for index in left):
            raise ValueError(f"row pivots at bond {bond} must span {bond} sites.")
        if any(len(index) != self._n_sites - bond for index in right):
            raise ValueError(
                f"column pivots at bond {bond} must span "
                f"{self._n_sites - bond} sites."
            )
        for pivots in (left, right):
            if len(set(pivots)) != len(pivots):
                repeated = next(index for index in pivots if pivots.count(index) > 1)
                raise ValueError(f"bond {bond} lists the pivot {repeated} twice.")
        for index in left + right:
            if any(value < 0 for value in index):
                raise ValueError(f"pivot {index} at bond {bond} has a negative index.")
        return left, right

    def _require_internal(self, bond: int) -> None:
        """Raise unless ``bond`` is one of ``1 .. n_sites-1``."""
        if not 1 <= bond <= self._n_sites - 1:
            raise ValueError(
                f"bond {bond} is not an internal bond of a "
                f"{self._n_sites}-site train."
            )

    @property
    def n_sites(self) -> int:
        """Number of sites the train covers."""
        return self._n_sites

    @property
    def ranks(self) -> List[int]:
        """Number of pivots at each bond ``0 .. n_sites``: the bond dimensions."""
        return [1] + [len(left) for left in self._left] + [1]

    def left(self, bond: int) -> Tuple[MultiIndex, ...]:
        """Row pivots at ``bond``, each covering sites ``0 .. bond-1``.

        Raises:
            ValueError: If ``bond`` is outside ``0 .. n_sites-1``.
        """
        if bond == 0:
            return ((),)
        if not 1 <= bond <= self._n_sites - 1:
            raise ValueError(
                f"left pivots are defined at bonds 0..{self._n_sites - 1}; "
                f"got {bond}."
            )
        return self._left[bond - 1]

    def right(self, bond: int) -> Tuple[MultiIndex, ...]:
        """Column pivots at ``bond``, each covering sites ``bond .. n_sites-1``.

        Raises:
            ValueError: If ``bond`` is outside ``1 .. n_sites``.
        """
        if bond == self._n_sites:
            return ((),)
        if not 1 <= bond <= self._n_sites - 1:
            raise ValueError(
                f"right pivots are defined at bonds 1..{self._n_sites}; got {bond}."
            )
        return self._right[bond - 1]

    def replace(
        self, bond: int, left: Sequence[MultiIndex], right: Sequence[MultiIndex]
    ) -> None:
        """Install new pivots at an internal bond.

        Args:
            bond: Internal bond, in ``1 .. n_sites-1``.
            left: Row pivots, each spanning sites ``0 .. bond-1``.
            right: Column pivots, each spanning sites ``bond .. n_sites-1``.

        Raises:
            TypeError: If an index is not an integer.
            ValueError: If ``bond`` is not internal or the pivots are
                malformed. The stored pivots are unchanged on failure.
        """
        self._require_internal(bond)
        left, right = self._checked(bond, left, right)
        self._left[bond - 1] = left
        self._right[bond - 1] = right

    def is_left_nested_at(self, bond: int) -> bool:
        """Whether every row pivot at ``bond`` extends one at ``bond - 1``.

        Raises:
            ValueError: If ``bond`` is not internal.
        """
        self._require_internal(bond)
        known = set(self.left(bond - 1))
        return all(index[:-1] in known for index in self.left(bond))

    def is_right_nested_at(self, bond: int) -> bool:
        """Whether every column pivot at ``bond`` extends one at ``bond + 1``.

        Raises:
            ValueError: If ``bond`` is not internal.
        """
        self._require_internal(bond)
        known = set(self.right(bond + 1))
        return all(index[1:] in known for index in self.right(bond))

    def is_fully_nested(self) -> bool:
        """Whether every internal bond is nested with both its neighbours."""
        return all(
            self.is_left_nested_at(bond) and self.is_right_nested_at(bond)
            for bond in range(1, self._n_sites)
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n_sites={self._n_sites}, ranks={self.ranks})"

    @classmethod
    def from_configuration(cls, configuration: Sequence[int]) -> "PivotLists":
        """Build rank-one pivot lists from a single configuration.

        Fully nested by construction: the halves at neighbouring bonds differ
        by exactly one site.

        Args:
            configuration: One index per site.

        Raises:
            TypeError: If an index is not an integer.
            ValueError: If the configuration is empty or has a negative index.

        Returns:
            PivotLists: One row pivot and one column pivot at every bond.
        """
        if len(configuration) == 0:
            raise ValueError("a configuration must cover at least one site.")
        lefts, rights = split_configuration(configuration)
        return cls(
            [([lefts[bond]], [rights[bond]]) for bond in range(1, len(configuration))]
        )


def _outside(indices: Sequence[MultiIndex], dims: Sequence[int]) -> bool:
    """Whether any multi-index has an entry outside ``0 .. dim-1`` at its site."""
    return any(
        not 0 <= value < dim for index in indices for value, dim in zip(index, dims)
    )


def _validate(dims: Tuple[int, ...], pivots: PivotLists) -> None:
    """Check that ``pivots`` index a tensor train of shape ``dims``.

    Raises:
        ValueError: If dims does not give one dimension per site, if there are
            fewer than two sites, or if a pivot lies outside dims.
    """
    if len(dims) != pivots.n_sites:
        raise ValueError(
            f"dims must give one dimension per site; got {len(dims)} "
            f"for {pivots.n_sites} sites."
        )
    if pivots.n_sites < 2:
        raise ValueError(
            f"a tensor train needs at least two sites; got {pivots.n_sites}."
        )
    for bond in range(1, pivots.n_sites):
        if _outside(pivots.left(bond), dims[:bond]) or _outside(
            pivots.right(bond), dims[bond:]
        ):
            raise ValueError(f"pivots at bond {bond} lie outside dims {dims}.")


class TensorCI:
    """A tensor in TCI form: pivot lists over a sampled tensor.

    No cores are stored. :meth:`to_tensor_train` samples them from ``box`` at
    the current pivots, so it reflects every update made to :attr:`pivots`.
    """

    def __init__(self, box: BlackBox, dims: Sequence[int], pivots: PivotLists):
        """Pair pivot lists with the tensor they index.

        Args:
            box: Sampler over the tensor.
            dims: One dimension per site.
            pivots: Pivot lists, kept by reference rather than copied.

        Raises:
            TypeError: If a dimension is not an integer.
            ValueError: If dims does not give one dimension per site, if there
                are fewer than two sites, or if a pivot lies outside dims.
        """
        self._dims = tuple(map(operator.index, dims))
        _validate(self._dims, pivots)
        self._box = box
        self._pivots = pivots

    @property
    def box(self) -> BlackBox:
        """Sampler over the tensor."""
        return self._box

    @property
    def dims(self) -> Tuple[int, ...]:
        """One dimension per site."""
        return self._dims

    @property
    def pivots(self) -> PivotLists:
        """Pivot lists at every bond."""
        return self._pivots

    def to_tensor_train(self) -> tensor_train.TensorTrain:
        """Contract the TCI form into a tensor train.

        Core ``s`` is ``T_s P_{s+1}^-1``, and the last core is ``T_{n-1}``.
        When the pivots are fully nested, the train equals the tensor on every
        entry of every ``T_s``, and hence of every ``P_b`` (Eq. (38)).

        Costs one request to ``box`` per entry of every ``T_s`` and ``P_b``,
        of which the ones already sampled are served from its cache.

        Raises:
            numpy.linalg.LinAlgError: If a pivot matrix is exactly singular.

        Returns:
            TensorTrain: Train whose bond dimensions are the ranks of
            :attr:`pivots`. Integer and boolean values give float64 cores.
        """
        last = self._pivots.n_sites - 1
        cores = []
        for site in range(last + 1):
            left = self._pivots.left(site)
            right = self._pivots.right(site + 1)
            core = self._box.sample(append_site(left, self._dims[site]), right)
            if not np.issubdtype(core.dtype, np.inexact):
                core = core.astype(np.float64)
            if site < last:
                pivot = self._box.sample(self._pivots.left(site + 1), right)
                core = np.linalg.solve(pivot.T, core.T).T  # core @ inv(pivot)
            cores.append(core.reshape(len(left), self._dims[site], len(right)))
        return tensor_train.TensorTrain(cores)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(dims={self._dims}, ranks={self._pivots.ranks})"
