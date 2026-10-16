# Copyright (c) 2026 QBoard LLC. Licensed under the AGPLv3. See LICENSE.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The tensor cross interpolation drivers, built on 2-site updates.

:func:`tci` builds a TCI form from given configurations, and
:func:`add_global_pivots` adds configurations to an existing one. Both then
alternate forward and backward half-sweeps of 2-site updates from
:mod:`ttnumpy.interpolation.sweeps`, each updating every bond once, until the
ranks settle with every pivot error within ``error_threshold``.

Cost, for ``n`` sites of dimension ``d`` and largest rank ``chi``:

- Half-sweep: ``O(n chi^2 d^2)`` requests to the sampler and
  ``O(n chi^2 d^2 (chi + n))`` time.
- The sampler caches every distinct value it evaluates, each keyed by its
  ``n`` indices; requests repeated across half-sweeps are served from the
  cache.
- Pivot lists: ``O(n^2 chi)`` integers.
- Contraction into a tensor train: ``O(n chi^2 d)`` requests and
  ``O(n d chi^3)`` time.
- Adding ``p`` configurations: ``O(n (chi + p)^2)`` requests, of which
  ``O(n (chi + p) p)`` are for values not yet sampled, and
  ``O(n (chi + p)^3)`` time, before the half-sweeps.

Overall, a run of ``h`` half-sweeps takes ``O(h n chi^2 d^2 (chi + n))`` time
plus ``e`` calls to the function, one per distinct configuration sampled,
with ``e = O(h n chi^2 d^2)``, and ``O(e n + n^2 chi + chi^2 d^2)`` memory.

Reference:
    Y. Núñez Fernández et al., Learning tensor networks with tensor cross
    interpolation: new algorithms and libraries, SciPost Phys. 18, 104
    (2025), Secs. 4.3.1 and 4.3.5.
"""

import logging
import operator
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

from ttnumpy.interpolation import tci_form
from ttnumpy.interpolation.black_box import BlackBox, BlackBoxFunction
from ttnumpy.interpolation.sweeps import (
    _check_threshold,
    _insert_pivots,
    _sampled_scale,
    two_site_update,
)
from ttnumpy.logger_config import PACKAGE_LOGGER_NAME

logger = logging.getLogger(PACKAGE_LOGGER_NAME)


@dataclass(frozen=True)
class TCIInfo:
    """Progress and outcome of a run of :func:`tci` or :func:`add_global_pivots`."""

    #: Number of half-sweeps performed.
    num_half_sweeps_run: int
    #: Whether the last half-sweep kept every pivot error within the threshold
    #: without changing a rank or the largest sampled modulus.
    converged: bool
    #: Largest pivot error of each half-sweep, relative to the largest modulus
    #: sampled before that half-sweep.
    pivot_errors: Tuple[float, ...]
    #: Largest rank after each half-sweep.
    max_ranks: Tuple[int, ...]


def _check_run(
    error_threshold: float, max_rank: Optional[int], max_half_sweeps: int
) -> None:
    """Reject settings that no run of half-sweeps can use.

    Raises:
        ValueError: If error_threshold is negative or NaN, or max_rank or
            max_half_sweeps is not positive.
    """
    _check_threshold(error_threshold, max_rank)
    if max_half_sweeps < 1:
        raise ValueError(f"max_half_sweeps must be positive; got {max_half_sweeps}.")


def _configurations(
    configurations: Sequence[Sequence[int]], dims: Sequence[int], name: str
) -> List[tci_form.MultiIndex]:
    """``configurations`` as tuples of ints, checked against ``dims``.

    Raises:
        TypeError: If a configuration is not a sequence, or an index is not an
            integer.
        ValueError: If there is no configuration, or one does not give an
            index per site or lies outside dims.
    """
    checked = []
    for configuration in configurations:
        if not hasattr(configuration, "__len__"):
            raise TypeError(
                f"{name} must hold configurations, each a sequence of indices; "
                f"got {configuration!r}."
            )
        checked.append(tuple(map(operator.index, configuration)))
    if not checked:
        raise ValueError(f"{name} must hold at least one configuration.")
    for configuration in checked:
        if len(configuration) != len(dims):
            raise ValueError(
                f"{name} must give one index per site; got {configuration} for "
                f"{len(dims)} sites."
            )
        if any(not 0 <= index < dim for index, dim in zip(configuration, dims)):
            raise ValueError(f"{name} holds {configuration}, outside dims {dims}.")
    return checked


def _two_site_sweep(
    tensor_ci: tci_form.TensorCI,
    error_threshold: float,
    max_rank: Optional[int],
    max_half_sweeps: int,
    name: str,
) -> TCIInfo:
    """Alternate 2-site half-sweeps, forward first, until the form converges.

    Each half-sweep is logged at DEBUG and the outcome at INFO to the
    ``ttnumpy`` logger under ``name``, and a run that reaches
    ``max_half_sweeps`` unconverged as a warning.
    """
    bonds = range(1, len(tensor_ci.dims))
    num_half_sweeps_run = 0
    pivot_errors: List[float] = []
    max_ranks: List[int] = []
    converged = False
    while not converged and num_half_sweeps_run < max_half_sweeps:
        scale = tensor_ci.box.max_modulus
        ranks_before = tensor_ci.pivots.ranks
        pivot_error = 0.0
        for bond in bonds if num_half_sweeps_run % 2 == 0 else reversed(bonds):
            update = two_site_update(tensor_ci, bond, error_threshold * scale, max_rank)
            pivot_error = max(pivot_error, update.pivot_error / scale)
        pivot_errors.append(pivot_error)
        max_ranks.append(max(tensor_ci.pivots.ranks))
        num_half_sweeps_run += 1
        # a larger modulus raises the threshold of the next half-sweep, which
        # can then drop pivots this one kept
        converged = (
            pivot_error <= error_threshold
            and tensor_ci.pivots.ranks == ranks_before
            and tensor_ci.box.max_modulus == scale
        )
        logger.debug(
            "%s half-sweep %d/%d: ranks %s, relative pivot error %.2e, "
            "%d evaluations",
            name,
            num_half_sweeps_run,
            max_half_sweeps,
            tensor_ci.pivots.ranks,
            pivot_error,
            tensor_ci.box.n_evals,
        )

    logger.info(
        "%s stopped after %d half-sweep(s) (%s): ranks %s, relative pivot "
        "error %.2e, %d evaluations",
        name,
        num_half_sweeps_run,
        "converged" if converged else "max_half_sweeps",
        tensor_ci.pivots.ranks,
        pivot_error,
        tensor_ci.box.n_evals,
    )
    if not converged:
        logger.warning(
            "%s did not converge in %d half-sweep(s): relative pivot error "
            "%.2e against error_threshold %.2e",
            name,
            num_half_sweeps_run,
            pivot_error,
            error_threshold,
        )
    return TCIInfo(
        num_half_sweeps_run, converged, tuple(pivot_errors), tuple(max_ranks)
    )


def tci(
    element_function: Union[BlackBoxFunction, BlackBox],
    dims: Sequence[int],
    initial_pivots: Sequence[Sequence[int]],
    error_threshold: float = 1e-7,
    max_rank: Optional[int] = None,
    max_half_sweeps: int = 20,
    return_info: bool = False,
) -> Union[tci_form.TensorCI, Tuple[tci_form.TensorCI, TCIInfo]]:
    """Interpolate a tensor by two-site tensor cross interpolation.

    Starts from the first initial pivot, adds the others as global pivots, and
    alternates forward and backward half-sweeps, each updating every bond once.
    An update samples the tensor on the slice through its bond, with the two
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
        initial_pivots: Configurations the first pivots are taken from, each
            with one index per site. The tensor must be nonzero at one or more
            of them.
        error_threshold: Threshold on the pivot error, relative to the largest
            modulus sampled before each half-sweep.
        max_rank: Upper bound on the number of pivots at each bond.
        max_half_sweeps: Largest number of half-sweeps to run.
        return_info: If True, also return a :class:`TCIInfo` with the pivot
            error and largest rank of every half-sweep.

    Raises:
        TypeError: If an initial pivot is not a sequence, or dims or an initial
            pivot has an entry that is not an integer.
        ValueError: If error_threshold is negative or NaN, max_rank or
            max_half_sweeps is not positive, dims has fewer than two sites or
            a dimension below 2, initial_pivots is empty, an initial pivot
            does not give one index per site or lies outside dims, the tensor
            vanishes at every initial pivot, or element_function returns a
            value that is not finite.

    Returns:
        TensorCI: The interpolation, or a tuple (TensorCI, TCIInfo) if
        return_info is True. It holds the sampler with every value sampled,
        and its :meth:`~ttnumpy.TensorCI.to_tensor_train`
        builds a :class:`~ttnumpy.tensor_train.TensorTrain` through the same
        cache.

    Reference:
        Y. Núñez Fernández et al., Learning tensor networks with tensor cross
        interpolation: new algorithms and libraries, SciPost Phys. 18, 104
        (2025), Secs. 4.3.1 and 4.3.5.
    """
    _check_run(error_threshold, max_rank, max_half_sweeps)
    dims = tuple(map(operator.index, dims))
    if len(dims) < 2 or any(dim < 2 for dim in dims):
        raise ValueError(
            f"tci needs at least two sites, each of dimension at least 2; "
            f"got dims {dims}."
        )
    configurations = _configurations(initial_pivots, dims, "initial_pivots")
    box = (
        element_function
        if isinstance(element_function, BlackBox)
        else BlackBox(element_function)
    )
    values = [box(configuration) for configuration in configurations]
    if not any(values):
        raise ValueError("the tensor vanishes at every initial pivot.")
    tensor_ci = tci_form.TensorCI(
        box, dims, tci_form.PivotLists.from_configuration(configurations[0])
    )
    if len(configurations) > 1:
        _insert_pivots(
            tensor_ci, configurations, error_threshold * box.max_modulus, max_rank
        )
    info = _two_site_sweep(tensor_ci, error_threshold, max_rank, max_half_sweeps, "tci")
    if return_info:
        return tensor_ci, info
    return tensor_ci


def add_global_pivots(
    tensor_ci: tci_form.TensorCI,
    configurations: Sequence[Sequence[int]],
    error_threshold: float = 1e-7,
    max_rank: Optional[int] = None,
    max_half_sweeps: int = 20,
    return_info: bool = False,
) -> Union[tci_form.TensorCI, Tuple[tci_form.TensorCI, TCIInfo]]:
    """Add global pivots to a TCI form and sweep it until it converges, in place.

    Each configuration is split at every bond, and its halves join the pivots
    there as candidates of a 0-site update. The form is then swept by 2-site
    updates as in :func:`tci`. A configuration whose halves no factorization
    selects leaves no pivot.

    Args:
        tensor_ci: Tensor in TCI form to extend.
        configurations: Global pivots, each with one index per site.
        error_threshold: Threshold on the pivot error, relative to the largest
            modulus sampled, which includes the values at the configurations.
        max_rank: Upper bound on the number of pivots at each bond.
        max_half_sweeps: Largest number of half-sweeps to run.
        return_info: If True, also return a :class:`TCIInfo` with the pivot
            error and largest rank of every half-sweep.

    Raises:
        TypeError: If a configuration is not a sequence or has an entry that is
            not an integer.
        ValueError: If error_threshold is negative or NaN, max_rank or
            max_half_sweeps is not positive, configurations is empty, a
            configuration does not give one index per site or lies outside
            the dims of tensor_ci, no nonzero value has been sampled, or the
            sampled function returns a value that is not finite.

    Returns:
        TensorCI: ``tensor_ci``, or a tuple (TensorCI, TCIInfo) if return_info
        is True.

    Reference:
        Y. Núñez Fernández et al., Learning tensor networks with tensor cross
        interpolation: new algorithms and libraries, SciPost Phys. 18, 104
        (2025), Sec. 4.3.5.
    """
    _check_run(error_threshold, max_rank, max_half_sweeps)
    proposals = _configurations(configurations, tensor_ci.dims, "configurations")
    for configuration in proposals:
        tensor_ci.box(configuration)
    scale = _sampled_scale(tensor_ci)
    _insert_pivots(tensor_ci, proposals, error_threshold * scale, max_rank)
    info = _two_site_sweep(
        tensor_ci, error_threshold, max_rank, max_half_sweeps, "add_global_pivots"
    )
    if return_info:
        return tensor_ci, info
    return tensor_ci
