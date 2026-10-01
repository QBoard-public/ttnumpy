# Copyright (c) 2026 QBoard LLC. Licensed under the AGPLv3. See LICENSE.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tensor cross interpolation algorithms, which update the pivots of a TCI form.

The two-site slice at bond ``b`` leaves sites ``b - 1`` and ``b`` free and
pins the rest to the pivots of the neighbouring bonds. Factorizing it selects
fresh pivots for bond ``b``, which are drawn from extensions of those
neighbours and are therefore nested with them. Bonds ``b - 1`` and ``b + 1``
are left as they were, so their nesting against the new lists may not hold.

:func:`tci` alternates forward and backward half-sweeps, each updating every
bond once, until the ranks settle with every pivot error within
``error_threshold``.

Cost, for ``n`` sites of dimension ``d`` and largest rank ``chi``:

- Update of one bond: up to ``chi^2 d^2`` requests to the sampler for the
  slice, ``O(chi^3 d^2)`` for its factorization with complete pivoting,
  ``O(chi^2 d^2)`` memory.
- Each request builds and hashes a configuration of ``n`` indices: ``O(n)``.
- Half-sweep: ``O(n chi^2 d^2)`` requests and ``O(n chi^2 d^2 (chi + n))``
  time.
- The sampler caches every distinct value it evaluates, each keyed by its
  ``n`` indices; requests repeated across half-sweeps are served from the
  cache.
- Pivot lists: ``O(n^2 chi)`` integers.
- Contraction into a tensor train: ``O(n chi^2 d)`` requests and
  ``O(n d chi^3)`` time.

Overall, a run of ``h`` half-sweeps takes ``O(h n chi^2 d^2 (chi + n))`` time
plus ``e`` calls to the function, one per distinct configuration sampled,
with ``e = O(h n chi^2 d^2)``, and ``O(e n + n^2 chi + chi^2 d^2)`` memory.

Reference:
    Y. Núñez Fernández et al., Learning tensor networks with tensor cross
    interpolation: new algorithms and libraries, SciPost Phys. 18, 104
    (2025), Sec. 4.3.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from ttnumpy.interpolation import prrlu, tci_form
from ttnumpy.interpolation.black_box import BlackBox, BlackBoxFunction
from ttnumpy.logger_config import PACKAGE_LOGGER_NAME

logger = logging.getLogger(PACKAGE_LOGGER_NAME)


@dataclass(frozen=True)
class TwoSiteUpdate:
    """Outcome of one two-site update."""

    #: Bond whose pivot lists were considered.
    bond: int
    #: Number of pivots the factorization kept.
    rank: int
    #: Largest modulus the factorization left in the Schur complement, which
    #: is the error of the tensor train on this slice.
    pivot_error: float


@dataclass(frozen=True)
class TCIInfo:
    """How a run of :func:`tci` ended."""

    #: Number of half-sweeps performed.
    num_half_sweeps_run: int
    #: Whether the last half-sweep kept every pivot error within the threshold
    #: without changing a rank or the largest sampled modulus.
    converged: bool
    #: Largest pivot error of the last half-sweep, relative to the largest
    #: modulus sampled before it.
    pivot_error: float


def two_site_slice(
    tensor_ci: tci_form.TensorCI, bond: int
) -> Tuple[np.ndarray, List[tci_form.MultiIndex], List[tci_form.MultiIndex]]:
    """Sample the two-site slice at ``bond``.

    Costs ``len(rows) * len(columns)`` requests to the sampler of
    ``tensor_ci``, of which the ones already sampled are served from its cache.

    Args:
        tensor_ci: Tensor in TCI form supplying the halves either side of the
            free sites.
        bond: Bond to slice at, in ``1 .. n_sites - 1``.

    Raises:
        ValueError: If bond is outside ``1 .. n_sites - 1``.

    Returns:
        tuple: The slice as a matrix, the multi-indices of its rows, and the
        multi-indices of its columns. Row ``r`` and column ``c`` hold the
        tensor element at ``rows[r] + columns[c]``.
    """
    pivots, dims = tensor_ci.pivots, tensor_ci.dims
    if not 1 <= bond <= pivots.n_sites - 1:
        raise ValueError(
            f"a two-site update needs a bond in 1..{pivots.n_sites - 1}; got {bond}."
        )
    rows = tci_form.append_site(pivots.left(bond - 1), dims[bond - 1])
    columns = tci_form.prepend_site(pivots.right(bond + 1), dims[bond])
    return tensor_ci.box.sample(rows, columns), rows, columns


def update_bond(
    tensor_ci: tci_form.TensorCI,
    bond: int,
    tolerance: float = 0.0,
    max_rank: Optional[int] = None,
) -> TwoSiteUpdate:
    """Replace the pivot lists at ``bond`` by the pivots of its two-site slice.

    The pivots of ``tensor_ci`` are modified in place. A factorization that
    keeps no pivots, reported as a rank of zero, leaves the lists untouched,
    since an empty bond would not describe a tensor train.

    Args:
        tensor_ci: Tensor in TCI form to update.
        bond: Bond to update, in ``1 .. n_sites - 1``.
        tolerance: Absolute threshold passed to the factorization.
        max_rank: Upper bound on the number of pivots kept.

    Raises:
        ValueError: If bond is outside ``1 .. n_sites - 1``.

    Returns:
        TwoSiteUpdate: The rank kept and the pivot error left on the slice.
    """
    matrix, rows, columns = two_site_slice(tensor_ci, bond)
    factorization = prrlu.prrlu(matrix, tolerance=tolerance, max_rank=max_rank)
    if factorization.rank:
        tensor_ci.pivots.replace(
            bond,
            [rows[index] for index in factorization.row_pivots],
            [columns[index] for index in factorization.col_pivots],
        )
    return TwoSiteUpdate(
        bond=bond, rank=factorization.rank, pivot_error=factorization.pivot_error
    )


def tci(
    element_function: Union[BlackBoxFunction, BlackBox],
    dims: Sequence[int],
    initial_pivot: Sequence[int],
    error_threshold: float = 1e-7,
    max_rank: Optional[int] = None,
    max_half_sweeps: int = 20,
    return_info: bool = False,
) -> Union[tci_form.TensorCI, Tuple[tci_form.TensorCI, TCIInfo]]:
    """Interpolate a tensor by two-site tensor cross interpolation.

    Starts from the pivots of ``initial_pivot`` and alternates forward and
    backward half-sweeps, forward first, each updating every bond once. An
    update samples the tensor on the slice through its bond, with the two
    sites at the bond free and the others pinned to neighbouring pivots, and
    its pivot error is the largest deviation of the train from the tensor on
    that slice. Each half-sweep refuses pivots at or below ``error_threshold``
    times the largest modulus sampled before it. The run stops once a
    half-sweep keeps every pivot error within that threshold without changing a
    rank or the largest sampled modulus. A run that reaches ``max_half_sweeps``
    first is logged as a warning to the ``ttnumpy`` logger.

    Only the slices are sampled, and the pivot errors cover no entry outside
    them: a feature that no slice reaches, such as an isolated spike, is
    absent from the result without affecting the pivot errors.

    Args:
        element_function: Tensor element at a configuration, called with a
            tuple of Python ints, one index per site. Either a plain function
            or a :class:`BlackBox`; a BlackBox is used as it is, so its cache
            and counters carry over, and a function is wrapped in a new one.
        dims: One dimension per site.
        initial_pivot: Configuration the first pivots are taken from, one
            index per site. The tensor must not vanish there.
        error_threshold: Threshold on the pivot error, relative to the largest
            modulus sampled before each half-sweep.
        max_rank: Upper bound on the number of pivots at each bond.
        max_half_sweeps: Largest number of half-sweeps to run.
        return_info: If True, also return a :class:`TCIInfo`.

    Raises:
        TypeError: If dims or initial_pivot has an entry that is not an
            integer.
        ValueError: If error_threshold is negative or NaN, max_rank or
            max_half_sweeps is not positive, initial_pivot does not give one
            index per site, dims has fewer than two sites or a dimension below
            2, initial_pivot lies outside dims or the tensor vanishes there, or
            element_function returns a value that is not finite.

    Returns:
        TensorCI: The interpolation, or a tuple (TensorCI, TCIInfo) if
        return_info is True. It holds the sampler with every value sampled,
        and its :meth:`~ttnumpy.interpolation.tci_form.TensorCI.to_tensor_train`
        builds a :class:`~ttnumpy.tensor_train.TensorTrain` through the same
        cache.

    Reference:
        Y. Núñez Fernández et al., Learning tensor networks with tensor cross
        interpolation: new algorithms and libraries, SciPost Phys. 18, 104
        (2025), Sec. 4.3.
    """
    if error_threshold < 0.0 or np.isnan(error_threshold):
        raise ValueError(
            f"error_threshold must be non-negative; got {error_threshold}."
        )
    if max_rank is not None and max_rank < 1:
        raise ValueError(f"max_rank must be positive; got {max_rank}.")
    if max_half_sweeps < 1:
        raise ValueError(f"max_half_sweeps must be positive; got {max_half_sweeps}.")
    if len(initial_pivot) != len(dims):
        raise ValueError(
            f"initial_pivot must give one index per site; got "
            f"{len(initial_pivot)} for {len(dims)} sites."
        )
    if any(dim < 2 for dim in dims):
        raise ValueError(f"every dimension must be at least 2; got dims {dims}.")
    tensor_ci = tci_form.TensorCI(
        (
            element_function
            if isinstance(element_function, BlackBox)
            else BlackBox(element_function)
        ),
        dims,
        tci_form.PivotLists.from_configuration(initial_pivot),
    )
    if tensor_ci.box(initial_pivot) == 0:
        raise ValueError(f"the tensor vanishes at initial_pivot {initial_pivot}.")

    bonds = range(1, len(tensor_ci.dims))
    num_half_sweeps_run = 0
    converged = False
    while not converged and num_half_sweeps_run < max_half_sweeps:
        scale = tensor_ci.box.max_modulus
        ranks_before = tensor_ci.pivots.ranks
        pivot_error = 0.0
        for bond in bonds if num_half_sweeps_run % 2 == 0 else reversed(bonds):
            update = update_bond(tensor_ci, bond, error_threshold * scale, max_rank)
            pivot_error = max(pivot_error, update.pivot_error / scale)
        num_half_sweeps_run += 1
        # a larger modulus raises the threshold of the next half-sweep, which
        # can then drop pivots this one kept
        converged = (
            pivot_error <= error_threshold
            and tensor_ci.pivots.ranks == ranks_before
            and tensor_ci.box.max_modulus == scale
        )
        logger.debug(
            "tci half-sweep %d/%d: ranks %s, relative pivot error %.2e, "
            "%d evaluations",
            num_half_sweeps_run,
            max_half_sweeps,
            tensor_ci.pivots.ranks,
            pivot_error,
            tensor_ci.box.n_evals,
        )

    logger.info(
        "tci stopped after %d half-sweep(s) (%s): ranks %s, relative pivot "
        "error %.2e, %d evaluations",
        num_half_sweeps_run,
        "converged" if converged else "max_half_sweeps",
        tensor_ci.pivots.ranks,
        pivot_error,
        tensor_ci.box.n_evals,
    )
    if not converged:
        logger.warning(
            "tci did not converge in %d half-sweep(s): relative pivot error "
            "%.2e against error_threshold %.2e",
            num_half_sweeps_run,
            pivot_error,
            error_threshold,
        )
    if return_info:
        return tensor_ci, TCIInfo(num_half_sweeps_run, converged, pivot_error)
    return tensor_ci
