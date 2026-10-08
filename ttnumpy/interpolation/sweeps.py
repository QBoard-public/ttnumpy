# Copyright (c) 2026 QBoard LLC. Licensed under the AGPLv3. See LICENSE.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""2-site, 1-site and 0-site updates of a TCI form, and sweeps of them.

An update replaces the pivots at one bond by the rows and columns that a
factorization selects from a matrix of tensor values. The three kinds differ
in where the candidate rows and columns come from:

- 2-site update at bond ``b``: rows extend the row pivots of bond ``b - 1``
  by site ``b - 1``, and columns extend the column pivots of bond ``b + 1``
  by site ``b``. The matrix is the two-site slice.
- 1-site update at site ``s``, forward: rows extend the row pivots of bond
  ``s`` by site ``s``, columns are the column pivots of bond ``s + 1``, and
  bond ``s + 1`` is replaced. Backward: rows are the row pivots of bond
  ``s``, columns extend those of bond ``s + 1`` by site ``s``, and bond ``s``
  is replaced. The matrix is the site tensor.
- 0-site update at bond ``b``: rows and columns are the pivots of bond ``b``.
  The matrix is the pivot matrix.

New pivots that extend a neighbour's are nested with it. Only the 2-site
update can raise a rank. A 0-site sweep updates every bond once. A 1-site
sweep updates every bond forward and then backward, repeated until the
pivots are fully nested. Both take a threshold relative to the largest
modulus sampled.

Cost, for ``n`` sites of dimension ``d`` and largest rank ``chi``:

- 2-site update: up to ``chi^2 d^2`` requests to the sampler,
  ``O(chi^3 d^2)`` for the factorization, ``O(chi^2 d^2)`` memory.
- 1-site update: ``chi^2 d`` requests, ``O(chi^3 d)`` time, ``O(chi^2 d)``
  memory.
- 0-site update: ``chi^2`` requests, ``O(chi^3)`` time, ``O(chi^2)`` memory.
- Each request builds and hashes a configuration of ``n`` indices: ``O(n)``.
- A 0-site sweep makes ``n - 1`` updates, and each forward and backward
  pass of a 1-site sweep ``2 (n - 1)``.

Reference:
    Y. Núñez Fernández et al., Learning tensor networks with tensor cross
    interpolation: new algorithms and libraries, SciPost Phys. 18, 104
    (2025): 2-site Sec. 4.3.1, 1-site Sec. 4.4.1, 0-site Sec. 4.4.2.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ttnumpy.interpolation import prrlu, tci_form
from ttnumpy.logger_config import PACKAGE_LOGGER_NAME

logger = logging.getLogger(PACKAGE_LOGGER_NAME)


@dataclass(frozen=True)
class BondUpdate:
    """Outcome of an update of the pivots at one bond."""

    #: Bond whose pivots were considered.
    bond: int
    #: Number of pivots the factorization kept.
    rank: int
    #: Largest modulus the factorization left in the Schur complement.
    pivot_error: float


def _bond_update(
    tensor_ci: tci_form.TensorCI,
    bond: int,
    matrix: np.ndarray,
    rows: Sequence[tci_form.MultiIndex],
    columns: Sequence[tci_form.MultiIndex],
    tolerance: float,
    max_rank: Optional[int],
) -> BondUpdate:
    """Install at ``bond`` the rows and columns a factorization selects.

    A factorization that selects none leaves the pivots untouched.
    """
    factorization = prrlu.prrlu(matrix, tolerance=tolerance, max_rank=max_rank)
    if factorization.rank:
        tensor_ci.pivots.replace(
            bond,
            [rows[index] for index in factorization.row_pivots],
            [columns[index] for index in factorization.col_pivots],
        )
    return BondUpdate(
        bond=bond, rank=factorization.rank, pivot_error=factorization.pivot_error
    )


def two_site_update(
    tensor_ci: tci_form.TensorCI,
    bond: int,
    tolerance: float = 0.0,
    max_rank: Optional[int] = None,
) -> BondUpdate:
    """2-site update of the pivots at ``bond``, in place.

    The new pivots extend those of bonds ``bond - 1`` and ``bond + 1``. A
    factorization that keeps no pivots leaves them untouched.

    Args:
        tensor_ci: Tensor in TCI form to update.
        bond: Bond to update, in ``1 .. n_sites - 1``.
        tolerance: Absolute threshold passed to the factorization.
        max_rank: Upper bound on the number of pivots kept.

    Raises:
        ValueError: If bond is outside ``1 .. n_sites - 1``.

    Returns:
        BondUpdate: The rank kept and the pivot error left on the two-site
        slice. While bonds ``bond - 1`` and ``bond + 1`` are nested with their
        own neighbours, it is the error of the tensor train on the slice.
    """
    pivots, dims = tensor_ci.pivots, tensor_ci.dims
    if not 1 <= bond <= pivots.n_sites - 1:
        raise ValueError(
            f"a 2-site update needs a bond in 1..{pivots.n_sites - 1}; got {bond}."
        )
    rows = tci_form.append_site(pivots.left(bond - 1), dims[bond - 1])
    columns = tci_form.prepend_site(pivots.right(bond + 1), dims[bond])
    matrix = tensor_ci.box.sample(rows, columns)
    return _bond_update(tensor_ci, bond, matrix, rows, columns, tolerance, max_rank)


def one_site_update(
    tensor_ci: tci_form.TensorCI,
    site: int,
    forward: bool,
    tolerance: float = 0.0,
    max_rank: Optional[int] = None,
) -> BondUpdate:
    """1-site update of the pivots at a bond of ``site``, in place.

    Forward, bond ``site + 1`` is replaced: its new row pivots extend those of
    bond ``site``, and its column pivots are a subset of the old. Backward,
    bond ``site`` is replaced: its new column pivots extend those of bond
    ``site + 1``, and its row pivots are a subset of the old. The rank never
    grows. A factorization that keeps no pivots leaves them untouched.

    Args:
        tensor_ci: Tensor in TCI form to update.
        site: Site whose tensor is factorized, in ``0 .. n_sites - 2``
            forward and in ``1 .. n_sites - 1`` backward.
        forward: Whether to replace bond ``site + 1`` rather than bond
            ``site``.
        tolerance: Absolute threshold passed to the factorization.
        max_rank: Upper bound on the number of pivots kept.

    Raises:
        ValueError: If site is outside the range for its direction.

    Returns:
        BondUpdate: The rank kept and the pivot error left on the site tensor.
    """
    pivots, dims = tensor_ci.pivots, tensor_ci.dims
    first, last = (0, pivots.n_sites - 2) if forward else (1, pivots.n_sites - 1)
    if not first <= site <= last:
        raise ValueError(
            f"a {'forward' if forward else 'backward'} 1-site update needs a "
            f"site in {first}..{last}; got {site}."
        )
    if forward:
        bond = site + 1
        rows = tci_form.append_site(pivots.left(site), dims[site])
        columns = pivots.right(bond)
    else:
        bond = site
        rows = pivots.left(bond)
        columns = tci_form.prepend_site(pivots.right(site + 1), dims[site])
    matrix = tensor_ci.box.sample(rows, columns)
    return _bond_update(tensor_ci, bond, matrix, rows, columns, tolerance, max_rank)


def zero_site_update(
    tensor_ci: tci_form.TensorCI,
    bond: int,
    tolerance: float = 0.0,
    max_rank: Optional[int] = None,
) -> BondUpdate:
    """0-site update of the pivots at ``bond``, in place.

    The new pivots are a subset of the old. A factorization that keeps no
    pivots leaves them untouched.

    Args:
        tensor_ci: Tensor in TCI form to update.
        bond: Bond to update, in ``1 .. n_sites - 1``.
        tolerance: Absolute threshold passed to the factorization.
        max_rank: Upper bound on the number of pivots kept.

    Raises:
        ValueError: If bond is outside ``1 .. n_sites - 1``.

    Returns:
        BondUpdate: The rank kept and the pivot error left on the pivot matrix.
    """
    pivots = tensor_ci.pivots
    if not 1 <= bond <= pivots.n_sites - 1:
        raise ValueError(
            f"a 0-site update needs a bond in 1..{pivots.n_sites - 1}; got {bond}."
        )
    rows, columns = pivots.left(bond), pivots.right(bond)
    matrix = tensor_ci.box.sample(rows, columns)
    return _bond_update(tensor_ci, bond, matrix, rows, columns, tolerance, max_rank)


def _insert_pivots(
    tensor_ci: tci_form.TensorCI,
    configurations: Sequence[tci_form.MultiIndex],
    tolerance: float,
    max_rank: Optional[int],
) -> None:
    """Offer the halves of ``configurations`` as pivots at every bond, in place.

    At each bond, a 0-site update factorizes the matrix whose rows are the row
    pivots followed by the new left halves and whose columns are the column
    pivots followed by the new right halves, and installs what it selects.
    """
    pivots = tensor_ci.pivots
    for bond in range(1, pivots.n_sites):
        rows = list(
            dict.fromkeys(
                pivots.left(bond)
                + tuple(configuration[:bond] for configuration in configurations)
            )
        )
        columns = list(
            dict.fromkeys(
                pivots.right(bond)
                + tuple(configuration[bond:] for configuration in configurations)
            )
        )
        matrix = tensor_ci.box.sample(rows, columns)
        _bond_update(tensor_ci, bond, matrix, rows, columns, tolerance, max_rank)


def _check_threshold(error_threshold: float, max_rank: Optional[int]) -> None:
    """Reject an error threshold or rank bound that no sweep can use.

    Raises:
        ValueError: If error_threshold is negative or NaN, or max_rank is not
            positive.
    """
    if error_threshold < 0.0 or np.isnan(error_threshold):
        raise ValueError(
            f"error_threshold must be non-negative; got {error_threshold}."
        )
    if max_rank is not None and max_rank < 1:
        raise ValueError(f"max_rank must be positive; got {max_rank}.")


def _sampled_scale(tensor_ci: tci_form.TensorCI) -> float:
    """Largest modulus sampled so far, which a relative threshold scales by.

    Raises:
        ValueError: If every value sampled so far is zero.
    """
    scale = tensor_ci.box.max_modulus
    if scale == 0.0:
        raise ValueError(
            "error_threshold is relative to the largest modulus sampled, and "
            "no nonzero value has been sampled."
        )
    return scale


def _one_site_half_sweep(
    tensor_ci: tci_form.TensorCI,
    forward: bool,
    error_threshold: float,
    max_rank: Optional[int],
) -> float:
    """1-site update of every bond in one direction, in place.

    Returns:
        float: The largest pivot error, relative to the largest modulus
        sampled before the half-sweep.
    """
    scale = _sampled_scale(tensor_ci)
    n_sites = len(tensor_ci.dims)
    sites = range(n_sites - 1) if forward else range(n_sites - 1, 0, -1)
    updates = [
        one_site_update(tensor_ci, site, forward, error_threshold * scale, max_rank)
        for site in sites
    ]
    return max(update.pivot_error for update in updates) / scale


def one_site_sweep(
    tensor_ci: tci_form.TensorCI,
    error_threshold: float = 0.0,
    max_rank: Optional[int] = None,
    max_sweeps: int = 10,
) -> float:
    """1-site sweeps until the pivots are fully nested, in place.

    Each sweep updates sites ``0 .. n_sites - 2`` forward and then sites
    ``n_sites - 1 .. 1`` backward. The forward half leaves every bond nested
    with the one before it, and the backward half with the one after it.
    Sweeps stop after the first whose backward half keeps every rank, which
    leaves the pivots fully nested; each earlier backward half lowers a rank.
    A site tensor lying entirely within the threshold leaves its bond
    untouched. Ranks never grow.

    Args:
        tensor_ci: Tensor in TCI form to update.
        error_threshold: Threshold on the pivot error, relative to the largest
            modulus sampled before each half of a sweep.
        max_rank: Upper bound on the number of pivots at each bond.
        max_sweeps: Largest number of sweeps to run.

    Raises:
        ValueError: If error_threshold is negative or NaN, max_rank or
            max_sweeps is not positive, or no nonzero value has been sampled.

    Returns:
        float: The largest pivot error of the sweeps, each half relative to
        the largest modulus sampled before it.
    """
    _check_threshold(error_threshold, max_rank)
    if max_sweeps < 1:
        raise ValueError(f"max_sweeps must be positive; got {max_sweeps}.")
    pivot_errors = []
    for _ in range(max_sweeps):
        pivot_errors.append(
            _one_site_half_sweep(tensor_ci, True, error_threshold, max_rank)
        )
        ranks = tensor_ci.pivots.ranks
        pivot_errors.append(
            _one_site_half_sweep(tensor_ci, False, error_threshold, max_rank)
        )
        if tensor_ci.pivots.ranks == ranks:
            return max(pivot_errors)
    logger.warning(
        "one_site_sweep reached %d sweep(s) with the ranks still falling; "
        "the pivots may not be fully nested",
        max_sweeps,
    )
    return max(pivot_errors)


def zero_site_sweep(
    tensor_ci: tci_form.TensorCI,
    error_threshold: float = 0.0,
    max_rank: Optional[int] = None,
) -> float:
    """0-site update of every bond, in place.

    Each bond keeps a subset of its pivots, chosen from its pivot matrix
    alone, so the nesting between bonds may not hold afterwards.

    Args:
        tensor_ci: Tensor in TCI form to update.
        error_threshold: Threshold on the pivot error, relative to the largest
            modulus sampled before the sweep.
        max_rank: Upper bound on the number of pivots at each bond.

    Raises:
        ValueError: If error_threshold is negative or NaN, max_rank is not
            positive, or no nonzero value has been sampled.

    Returns:
        float: The largest pivot error of the sweep, relative to the largest
        modulus sampled before it.
    """
    _check_threshold(error_threshold, max_rank)
    scale = _sampled_scale(tensor_ci)
    updates = [
        zero_site_update(tensor_ci, bond, error_threshold * scale, max_rank)
        for bond in range(1, len(tensor_ci.dims))
    ]
    return max(update.pivot_error for update in updates) / scale


def recompress(
    tensor_ci: tci_form.TensorCI,
    error_threshold: float,
    max_rank: Optional[int] = None,
    max_sweeps: int = 10,
) -> float:
    """Lower the ranks of a TCI form to ``error_threshold``, in place.

    Runs a 0-site sweep and then a 1-site sweep, which leaves the pivots fully
    nested, both at ``error_threshold`` and ``max_rank``.

    Args:
        tensor_ci: Tensor in TCI form to recompress.
        error_threshold: Threshold on the pivot error, relative to the largest
            modulus sampled before the 0-site sweep and before each half of a
            1-site sweep.
        max_rank: Upper bound on the number of pivots at each bond.
        max_sweeps: Largest number of 1-site sweeps to run.

    Raises:
        ValueError: If error_threshold is negative or NaN, max_rank or
            max_sweeps is not positive, or no nonzero value has been sampled.

    Returns:
        float: The larger pivot error of the two sweeps.
    """
    pivot_error = zero_site_sweep(tensor_ci, error_threshold, max_rank)
    return max(
        pivot_error,
        one_site_sweep(tensor_ci, error_threshold, max_rank, max_sweeps),
    )
