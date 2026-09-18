"""Module for tensor train type and its basic operations."""

from typing import List, Optional, Tuple, Union, Literal
import logging
from math import prod
import numbers
from enum import Enum

import numpy as np
from scipy.linalg import svd

from ttnumpy import utils
from ttnumpy.logger_config import PACKAGE_LOGGER_NAME

logger = logging.getLogger(PACKAGE_LOGGER_NAME)


class DimensionMismatch(Exception):
    """Exception for dimension errors."""

    def __init__(self, msg="Dimension mismatch"):
        self.msg = msg
        super().__init__(self.msg)


class WrongTTFormat(Exception):
    """Exception for error in tensor train structure."""

    def __init__(self, msg="Wrong tensor train format"):
        self.msg = msg
        super().__init__(self.msg)


class WrongTTType(Exception):
    """Exception for error in tensor train type."""

    def __init__(self, msg="Wrong tensor train type"):
        self.msg = msg
        super().__init__(self.msg)


class AmbiguousFormat(Exception):
    """Exception for insufficient information leading to ambiguous format."""

    def __init__(self, msg="Not enough information to define object"):
        self.msg = msg
        super().__init__(self.msg)


def _update_bonds(tt: "TensorTrain") -> List[int]:
    """Update bond dimensions based on current cores."""
    return [tt.cores[i].shape[0] for i in range(tt.n_cores)] + [1]


class TensorTrainType(Enum):
    "Type of Tensor Train"

    TT = "Tensor Train"
    TTM = "Tensor Train Matrix"


class TensorTrain:
    """Tensor Train (TT) type.

    Contains TT initialization and basic operations.
    Supports Tensor Train Matrix (TTM) format. Both TT
    and TTM are treated as rank-4 tensors:

    ``ranks[i] x row_modes[i] x col_modes[i] x ranks[i+1]``,

    where, for TT, the column dimension is 1.


    Reference:
    - Matrix product state basics https://arxiv.org/abs/1008.3477;
    - Tensor train https://epubs.siam.org/doi/abs/10.1137/090752286
    """

    def __init__(self, cores: List[np.ndarray]):
        """Initialize a tensor train.

        Args:
            cores: Tensor train cores. Cores must be rank-3 or rank-4 tensors
                for Tensor Train and Tensor Train Matrix, respectively. First and
                last indices are treated as bond indices, while middle indices are
                physical indices. Bonds of neighboring cores must match.

        Raises:
            WrongTTFormat: If tensor does not satisfy open boundary conditions
                or a core tensor has rank other than 3 or 4
            DimensionMismatch: If neighboring cores have different bond connections

        Notes:
            This type supports only open boundary conditions, so first
            and last cores must have dummy indices in the first and last
            positions, respectively (e.g. core[0].shape = (1, 5, 3),
            core[-1].shape = (3, 5, 1)).
        """

        if len(cores) < 2:
            raise WrongTTFormat("Tensor train must has at least 2 sites")

        self.n_cores = len(cores)  # number of cores
        self.cores = cores  # list of cores
        self.max_rank = 1  # maximal bond dimension
        self.orth_center = None  # position of orthogonal center
        self.row_modes = []  # physical row dimensions
        self.col_modes = []  # physical column dimensions
        self.bonds = [1]  # bond dimensions

        if np.all([len(self.cores[i].shape) == 3 for i in range(self.n_cores)]):
            self.tt_type = TensorTrainType.TT
        elif np.all([len(self.cores[i].shape) == 4 for i in range(self.n_cores)]):
            if np.all([self.cores[i].shape[2] == 1 for i in range(self.n_cores)]):
                self.tt_type = TensorTrainType.TT
            else:
                self.tt_type = TensorTrainType.TTM
        else:
            raise WrongTTFormat("All cores must be 3 or 4 rank tensors")

        for i in range(self.n_cores):
            core_shape = self.cores[i].shape

            if i == 0 and core_shape[0] != 1:
                raise WrongTTFormat(
                    "Only open boundary conditions is supported, first core must has first index equal to 1"
                )
            if i == self.n_cores - 1 and core_shape[-1] != 1:
                raise WrongTTFormat(
                    "Only open boundary conditions is supported, last core must has last index equal to 1"
                )
            if core_shape[0] != self.bonds[i]:
                raise DimensionMismatch(
                    f"Connected sites bonds are mismatching: {core_shape[0]} != {self.bonds[i]}"
                )

            if self.tt_type == TensorTrainType.TT:
                self.row_modes.append(core_shape[1])
                self.col_modes.append(1)
                self.cores[i] = self.cores[i].reshape(
                    (core_shape[0], core_shape[1], 1, core_shape[-1])
                )
            elif self.tt_type == TensorTrainType.TTM:
                self.row_modes.append(core_shape[1])
                self.col_modes.append(core_shape[2])
            else:
                raise WrongTTFormat("All cores must be 3 or 4 rank tensors")

            self.max_rank = (
                core_shape[-1] if core_shape[-1] > self.max_rank else self.max_rank
            )
            self.bonds.append(core_shape[-1])

        logger.debug(
            "%s with rank %s saved.\nBond dimensions: %s",
            self.tt_type.value,
            self.max_rank,
            self.bonds,
        )

    def __getitem__(self, indexes: slice) -> List[np.ndarray]:
        """Get cores at the given index positions."""
        return self.cores[indexes]

    def __len__(self) -> int:
        """Get number of cores."""
        return self.n_cores

    def add(self, other: "TensorTrain", truncate: bool = True) -> "TensorTrain":
        """Sum of two tensor trains.

        Resulting tensor train will have bond dimensions equal to sum of
        bond dimensions of summands.

        Args:
            other: Tensor train to be added;
            truncate: If True, truncate resulting tensor train after addition (default=True).

        Raises:
            TypeError: If other is not TensorTrain type;
            DimensionMismatch: If tensor trains have different physical dimensions.

        Returns:
            TensorTrain: Sum of two tensor trains.
        """
        if not isinstance(other, TensorTrain):
            raise TypeError(
                f"unsupported operand type(s) for +: 'TensorTrain' and '{type(other)}'"
            )
        if self.physical_dims() != other.physical_dims():
            raise DimensionMismatch("Tensor trains must have the same dimensions")
        ranks = (
            [1] + [self.bonds[i] + other.bonds[i] for i in range(1, self.n_cores)] + [1]
        )

        cores = []

        for i in range(self.n_cores):
            cores.append(
                np.zeros([ranks[i], self.row_modes[i], self.col_modes[i], ranks[i + 1]])
            )

            # append lhs core
            cores[i][: self.bonds[i], :, :, : self.bonds[i + 1]] = self.cores[i]

            # append rhs core
            r_1 = ranks[i] - other.bonds[i]
            r_2 = ranks[i]
            r_3 = ranks[i + 1] - other.bonds[i + 1]
            r_4 = ranks[i + 1]
            cores[i][r_1:r_2, :, :, r_3:r_4] = other[i]

        tt_sum = TensorTrain(cores)
        if truncate:
            tt_sum.truncate(inplace=True)

        return tt_sum

    def __add__(self, other: "TensorTrain") -> "TensorTrain":
        """Addition operator for two tensor trains."""
        return self.add(other)

    def multiply(
        self,
        other: Union[int, float, complex, "TensorTrain"],
        truncate: bool = False,
    ) -> "TensorTrain":
        """
        Element-wise product of two tensor trains or tensor train and scalar.

        Args:
            other: Tensor train or scalar to be multiplied;
            truncate: If True, truncate resulting tensor train after multiplication (default=False).

        Raises:
            TypeError: If other is not TensorTrain type or scalar;
            DimensionMismatch: If tensor trains have different physical dimensions.
        """
        if isinstance(other, TensorTrain):
            if self.physical_dims() != other.physical_dims():
                raise DimensionMismatch("Tensor trains must have the same dimensions")

            prod_cores = []

            for i in range(self.n_cores):
                core = np.reshape(
                    np.einsum(
                        "aijb,mijn->amijbn", self.cores[i], other[i], optimize="optimal"
                    ),
                    [
                        self.bonds[i] * other.bonds[i],
                        self.row_modes[i],
                        self.col_modes[i],
                        self.bonds[i + 1] * other.bonds[i + 1],
                    ],
                )
                prod_cores.append(core)
            prod_res = TensorTrain(prod_cores)
        elif isinstance(other, numbers.Number):
            prod_res = self.copy()
            scaled_core = (
                prod_res.orth_center if prod_res.orth_center is not None else 0
            )
            prod_res.cores[scaled_core] = other * prod_res.cores[scaled_core]
        else:
            raise TypeError(
                f"unsupported operand type(s) for *: 'TensorTrain' and '{type(other)}'"
            )

        if truncate:
            prod_res.truncate(inplace=True)

        return prod_res

    def __mul__(
        self, other: Union[int, float, complex, "TensorTrain"]
    ) -> "TensorTrain":
        """Element-wise left product operator."""
        return self.multiply(other)

    def __rmul__(self, other: Union[int, float, complex]) -> "TensorTrain":
        """Element-wise right product operator."""
        return self.multiply(other)

    def __neg__(self) -> "TensorTrain":
        """Negative tensor train."""
        return self.multiply(-1)

    def __sub__(self, other: "TensorTrain") -> "TensorTrain":
        """Subtraction of two tensor trains."""
        return self.add(-other)

    def dot(
        self,
        other: "TensorTrain",
        truncate: bool = False,
    ) -> "TensorTrain":
        """Matrix product of two tensor trains.

        Four types of products are supported:
        - Matrix-vector product: TTM @ TT;
        - Matrix product: TTM @ TTM;
        - Outer product: TT @ TT.transpose();
        - Inner product: TT.transpose() @ TT.

        There is no implicit transposition: column modes of the left operand
        must match row modes of the right operand. The inner product is
        returned as a TT with all physical dimensions equal to 1; use
        get_full_tensor() to extract the scalar. Note that complex operands
        are not conjugated (matmul semantics).

        Args:
            other: Tensor train to be multiplied;
            truncate: If True, truncate resulting tensor train after multiplication (default=True).

        Raises:
            TypeError: If other is not TensorTrain type;
            DimensionMismatch: If tensor trains have different physical dimensions.

        Returns:
            TensorTrain: Matrix product of two tensor trains.
        """
        if not isinstance(other, TensorTrain):
            raise TypeError(
                f"unsupported operand type(s) for @: 'TensorTrain' and '{type(other)}'"
            )
        if self.col_modes != other.row_modes:
            raise DimensionMismatch(
                "Tensor trains must have the same last and first physical dimensions respectively"
            )

        prod_cores = []

        for i in range(self.n_cores):
            core = np.reshape(
                np.einsum(
                    "aijb,mjkn->amikbn", self.cores[i], other[i], optimize="optimal"
                ),
                [
                    self.bonds[i] * other.bonds[i],
                    self.row_modes[i],
                    other.col_modes[i],
                    self.bonds[i + 1] * other.bonds[i + 1],
                ],
            )
            prod_cores.append(core)

        tt_matmul = TensorTrain(prod_cores)

        if truncate:
            tt_matmul.truncate(inplace=True)

        return tt_matmul

    def __matmul__(self, other: "TensorTrain") -> "TensorTrain":
        """Matrix product operator."""
        return self.dot(other)

    def is_ttm(self) -> bool:
        """Check whether tensor train is an operator."""
        return self.tt_type == TensorTrainType.TTM

    def tt_norm(self) -> float:
        """Calculate Frobenius norm of tensor train.

        For TTM, norm is calculated as norm of the corresponding TT, so it is not
        the operator norm, but rather the Frobenius norm of the corresponding matrix.

        If orth_center is set, the norm is the norm of the center core and no
        orthogonalization is performed.
        """
        if self.orth_center is not None:
            return np.linalg.norm(self.cores[self.orth_center].reshape(-1))

        tt_tensor = self.copy()
        if tt_tensor.is_ttm():
            cores = [
                tt_tensor.cores[i].reshape(
                    tt_tensor.bonds[i],
                    tt_tensor.row_modes[i] * tt_tensor.col_modes[i],
                    1,
                    tt_tensor.bonds[i + 1],
                )
                for i in range(len(tt_tensor))
            ]

            tt_tensor = TensorTrain(cores)

        tt_tensor = tt_tensor.orthonormalize(canonical="right", inplace=True)

        # compute norm from first core
        norm = np.linalg.norm(tt_tensor.cores[0].reshape(-1))

        return norm

    def copy(self) -> "TensorTrain":
        """Create a copy of tensor train."""
        cores = [self.cores[i].copy() for i in range(self.n_cores)]
        tt_copy = TensorTrain(cores)
        tt_copy.orth_center = self.orth_center
        return tt_copy

    # TODO: QBRD-5336 rename to modes
    def physical_dims(self) -> List:
        """Get list of physical dimensions."""
        if self.tt_type is TensorTrainType.TT:
            return self.row_modes
        return list(zip(self.row_modes, self.col_modes))

    def get_element(self, element_index: List[int | Tuple[int]]) -> float:
        """Get one element from tensor train.

        All physical indices must be given in the element_index argument.
        If tensor train is in TTM format, element_index must contain tuples
        of index pairs.

        Args:
            element_index: list of physical indices for each core. For TTM,
                each index must be a tuple of (row_index, col_index).

        Raises:
            DimensionMismatch: If number of given indices is not equal to number
                of cores or if tensor train is in TTM format, but given indices are not
                tuples of (row_index, col_index).

        Returns:
            float: element of the tensor train at the given index.
        """

        if len(element_index) != self.n_cores:
            raise DimensionMismatch(
                f"Number of indexes not equal to number of tensor train cores: {len(element_index)} != {self.n_cores}"
            )

        if self.tt_type == TensorTrainType.TT and isinstance(element_index[0], int):
            element_index = list(zip(element_index, [0] * self.n_cores))
        contraction = np.tensordot(
            self.cores[0][:, element_index[0][0], element_index[0][1], :],
            self.cores[1][:, element_index[1][0], element_index[1][1], :],
            axes=1,
        )
        for i in range(2, self.n_cores):
            contraction = np.tensordot(
                contraction,
                self.cores[i][:, element_index[i][0], element_index[i][1], :],
                axes=1,
            )

        return contraction.item()

    def get_full_tensor(self) -> np.ndarray:
        """Collapse train into full tensor.

        For TT format, resulting tensor will have shape (row_dim_1, row_dim_2, ...).
        For TTM format, resulting tensor will have shape (row_dim_1, row_dim_2, ..., col_dim_1, col_dim_2, ...).
        """

        tensor = np.tensordot(self.cores[0], self.cores[1], axes=1)
        for i in range(2, self.n_cores):
            tensor = np.tensordot(tensor, self.cores[i], axes=1)
        shape = self.row_modes
        if self.tt_type == TensorTrainType.TTM:
            shape = [x for pair in zip(self.row_modes, self.col_modes) for x in pair]
            tensor = tensor.reshape(shape)
            shape = self.row_modes + self.col_modes
            permute = tuple(
                np.arange(len(shape))
                .reshape((len(shape) // 2, 2))
                .transpose()
                .flatten()
            )
            tensor = tensor.transpose(permute)

        if shape == [1] * len(shape):
            return tensor.item()

        return tensor.reshape(shape)

    def __str__(self):
        """Get main attributes of tensor train."""
        return f"{self.tt_type.value} with following properties:\n\t \
        Number of cores: {self.n_cores}\n\t \
        Bonds: {self.bonds}\n\t \
        Physical indexes: {self.physical_dims()}"

    def transpose(
        self,
        cores_ind: Optional[List[int]] = None,
        conjugate: bool = False,
        inplace: bool = False,
    ) -> "TensorTrain":
        """Transpose tensor train.

        Args:
            cores_ind: indices of cores that should be transposed. If
                cores_ind=None (default), all cores are transposed.
            conjugate: If True, conjugate tensor train matrix.
            inplace: If True, perform operation inplace (default=False).

        Returns:
            TensorTrain: Transposed tensor train.
        """

        tt_transpose = self if inplace else self.copy()
        new_cores = tt_transpose.cores

        if cores_ind is None:
            cores_ind = list(range(self.n_cores))
        for i in cores_ind:
            new_cores[i] = new_cores[i].transpose([0, 2, 1, 3])
            tt_transpose.row_modes[i], tt_transpose.col_modes[i] = (
                tt_transpose.col_modes[i],
                tt_transpose.row_modes[i],
            )

            if conjugate:
                if inplace:
                    np.conj(new_cores[i], out=new_cores[i])
                else:
                    new_cores[i] = new_cores[i].conj()

        if not tt_transpose.is_ttm():
            tt_transpose.tt_type = TensorTrainType.TTM
        else:
            if all(dim == 1 for dim in tt_transpose.col_modes):
                tt_transpose.tt_type = TensorTrainType.TT

        return tt_transpose

    def conj(self, inplace: bool = False) -> "TensorTrain":
        """Conjugate tensor train.

        Args:
            inplace: If True, perform operation inplace (default=False).

        Returns:
            TensorTrain: Conjugated tensor train.
        """

        tt_conj = self if inplace else self.copy()
        tt_conj.cores = [tt_conj.cores[i].conj() for i in range(self.n_cores)]
        return tt_conj

    def orthonormalize(
        self,
        canonical: Literal["left", "right", "mix"] = "left",
        center_position: Optional[int] = None,
        inplace: bool = False,
    ) -> "TensorTrain":
        """Orthonormalization of tensor train with QR decomposition.

        Args:
            canonical: type of final canonical form. "left" for left-orthonormalization,
                "right" for right-orthonormalization, "mix" for mixed method (default).
            center_position: position of the center core, up to which TT will be orthonormalized.
                If None (default), center position will be in the middle of the tensor train.
            inplace: If True, perform operation inplace (default=False).

        Raises:
            ValueError: If center_position is out of range or canonical is unknown.

        Returns:
            TensorTrain: Orthonormalized tensor train. orth_center is set only
            when the resulting train is in a valid canonical form: a partial
            "left"/"right" sweep on a train without a known orthogonal center
            leaves orth_center unset.
        """

        if center_position is not None and not 0 <= center_position < self.n_cores:
            raise ValueError(
                f"center_position must be in [0, {self.n_cores - 1}], but got {center_position}"
            )

        tt_orth = self if inplace else self.copy()
        had_center = tt_orth.orth_center is not None

        if canonical == "left":
            end = center_position if center_position is not None else self.n_cores - 1
            start = 0
            if tt_orth.orth_center is not None:
                start = tt_orth.orth_center
                tt_orth.orth_center = end if start <= end else tt_orth.orth_center
            if start >= end:
                logger.debug(
                    "Left orthogonality up to center position is already "
                    "satisfied. No changes applied."
                )
                return tt_orth
            tt_orth.cores = utils.left_orthonormalize(tt_orth.cores, start, end)

        elif canonical == "right":
            end = center_position if center_position is not None else 0
            start = tt_orth.n_cores - 1
            if tt_orth.orth_center is not None:
                start = tt_orth.orth_center
                tt_orth.orth_center = end if start >= end else tt_orth.orth_center
            if start <= end:
                logger.debug(
                    "Right orthogonality up to center position is already "
                    "satisfied. No changes applied."
                )
                return tt_orth
            tt_orth.cores = utils.right_orthonormalize(tt_orth.cores, start, end)

        elif canonical == "mix":
            if center_position is None:
                center_position = self.n_cores // 2
            end = center_position
            tt_orth.cores = utils.left_orthonormalize(tt_orth.cores, 0, center_position)
            tt_orth.cores = utils.right_orthonormalize(
                tt_orth.cores, self.n_cores - 1, center_position
            )
        else:
            raise ValueError(
                f"Canonical form must be 'left', 'right' or 'mix', but got {canonical}"
            )

        # a partial sweep on a train without a known center leaves the cores
        # beyond the swept range untouched, so no valid canonical form is reached
        if had_center or canonical == "mix" or end in (0, self.n_cores - 1):
            tt_orth.orth_center = end
        else:
            tt_orth.orth_center = None
        tt_orth.bonds = _update_bonds(tt_orth)
        tt_orth.max_rank = max(tt_orth.bonds)
        return tt_orth

    def compress_svd(
        self,
        error: float = 0.0,
        max_rank: Optional[int] = None,
        canonical: Literal["left", "right", "auto"] = "auto",
        inplace: bool = False,
    ) -> "TensorTrain":
        """Compress the tensor train with an SVD-based truncation.

        Args:
            error: Allowed truncation error. Defaults to ``0.0``.
            max_rank: Optional bound for the bond dimension.
            canonical: Canonical form to use. ``"left"`` performs a
                left-to-right SVD sweep, ``"right"`` performs a
                right-to-left sweep, and ``"auto"`` chooses the direction
                from ``orth_center``.
            inplace: If ``True``, modify this tensor train in place.

        Raises:
            ValueError: If canonical is not 'left', 'right' or 'auto'.

        Returns:
            Tensor train after SVD compression.
        """
        tt_compressed = self if inplace else self.copy()

        initial_bonds = tt_compressed.bonds

        if canonical == "auto":
            if (
                tt_compressed.orth_center is None
                or tt_compressed.orth_center <= tt_compressed.n_cores / 2
            ):
                canonical = "left"
            else:
                canonical = "right"

        if canonical == "right":
            tt_compressed.cores = utils.right_svd(tt_compressed.cores, error, max_rank)
            tt_compressed.orth_center = 0

        elif canonical == "left":
            tt_compressed.cores = utils.left_svd(tt_compressed.cores, error, max_rank)
            tt_compressed.orth_center = self.n_cores - 1
        else:
            raise ValueError(
                f"Canonical form must be 'left', 'right' or 'auto', but got {canonical}"
            )

        tt_compressed.bonds = [
            tt_compressed.cores[i].shape[0] for i in range(tt_compressed.n_cores)
        ] + [1]
        logger.debug(
            "Truncated bonds after SVD: %s\nTotal compression: %f",
            tt_compressed.bonds,
            np.prod(tt_compressed.bonds) / np.prod(initial_bonds),
        )
        tt_compressed.max_rank = max(tt_compressed.bonds)

        return tt_compressed

    def truncate(
        self,
        error: float = 1e-15,
        max_rank: Optional[int] = None,
        canonical: Literal["auto", "left", "right"] = "auto",
        inplace: bool = False,
    ) -> "TensorTrain":
        """Round tensor train to minimal ranks.

        Default strategy is left-to-right orthonormalization using QR decomposition,
        then right-to-left truncation using SVD decomposition. If orth_center is set and
        orth_direction is "auto", orthonormalization will start from orth_center position
        to the closest boundary, so direction of sweeps might change.

        Example:
            If orth_center is in the left half of the tensor train, right-to-left
            orthonormalization will be performed first, then left-to-right truncation.

        Args:
            error: Allowed error of truncation;
            max_rank: If given, set a bound for bond dimension;
            canonical: Type of final canonical form. If "auto" (default), direction will be chosen based on orth_center position.
            inplace: If True, perform operation inplace (default=False).

        Raises:
            ValueError: If canonical is not 'auto', 'left' or 'right'.

        Returns:
            TensorTrain: Truncated tensor train.
        """
        if canonical not in ("auto", "left", "right"):
            raise ValueError(
                f"Canonical form must be 'left', 'right' or 'auto', "
                f"but got {canonical}"
            )

        tt_trunc = self if inplace else self.copy()

        orth_form = "left" if canonical == "right" else "right"

        if canonical == "auto":
            if (
                tt_trunc.orth_center is None
                or tt_trunc.orth_center >= tt_trunc.n_cores / 2
            ):
                orth_form = "left"
            else:
                orth_form = "right"
        if inplace:
            tt_trunc.orthonormalize(canonical=orth_form, inplace=True)
            tt_trunc.compress_svd(error, max_rank, inplace=True)
        else:
            tt_trunc = tt_trunc.orthonormalize(canonical=orth_form)
            tt_trunc = tt_trunc.compress_svd(error, max_rank)

        return tt_trunc

    def tt_to_qtt(
        self,
        mode_size: int = 2,
        error: float = 1e-12,
        max_rank: Optional[int] = None,
    ) -> "TensorTrain":
        """Convert tensor train to quantized tensor train (QTT) format.

        Each core with mode m decomposes into log_{mode_size}(m) cores with mode_size physical dimensions. For example,
        a core with shape (r_i, 16, r_{i+1}) will decompose into 4 cores with shapes
        (r_i, 2, 1, new_bond), ..., (new_bond, 2, 1, r_{i+1}), where new_bond is determined by SVD
        truncation with given error and max_rank.

        Args:
            mode_size: size of quantized modes (default=2);
            error: Allowed error of truncation (default=1e-12);
            max_rank: If given, set a bound for bond dimension.

        Returns:
            TensorTrain: Tensor train in QTT format. The result is
            left-canonical with orth_center at the last core.

        Raises:
            DimensionMismatch: If any of the row modes is not a power of mode_size or if
                tensor train is in TTM format and any of the column modes is not a power of mode_size.
            ValueError: If mode_size is less than 2 or max_rank is not a positive integer.
        """
        if mode_size < 2:
            raise ValueError(f"mode_size must be at least 2, but got {mode_size}")
        if max_rank is not None and max_rank <= 0:
            raise ValueError("max_rank must be a positive integer.")

        qtt_nums = []
        for i, row_mode in enumerate(self.row_modes):
            qtt_num = 0
            while mode_size**qtt_num < row_mode:
                qtt_num += 1
            if mode_size**qtt_num != row_mode:
                raise DimensionMismatch(
                    f"Row mode {row_mode} not a power of {mode_size}"
                )
            if self.is_ttm() and mode_size**qtt_num != self.col_modes[i]:
                raise DimensionMismatch(
                    "Column mode size of TTM not a power of mode size"
                )
            qtt_nums.append(qtt_num)

        row_mode_size = mode_size
        col_mode_size = mode_size if self.is_ttm() else 1

        # every split of a core uses qtt_num - 1 SVDs; on top of that, each of
        # the n_cores - 1 group boundaries uses one SVD to push the remainder
        # into the next core, which keeps the train canonical during the sweep
        # so that each local truncation is bounded by delta
        total_svd = sum(max(0, qtt_num - 1) for qtt_num in qtt_nums) + self.n_cores - 1
        svd_count = max(1, total_svd)

        tt = self.orthonormalize(canonical="right")
        delta = error / np.sqrt(svd_count) * tt.tt_norm()

        qtt_cores = []
        for i, qtt_num in enumerate(qtt_nums):
            core = tt.cores[i]
            left_bond = core.shape[0]
            right_bond = core.shape[-1]

            for j in range(qtt_num - 1, 0, -1):
                logger.debug("Reshaping core %s and %s for SVD", i, j)
                core = core.reshape(
                    left_bond,
                    row_mode_size,
                    row_mode_size**j,
                    col_mode_size,
                    col_mode_size**j,
                    right_bond,
                ).transpose(0, 1, 3, 2, 4, 5)
                U, S, V = svd(
                    core.reshape(
                        left_bond * row_mode_size * col_mode_size,
                        row_mode_size**j * col_mode_size**j * right_bond,
                    ),
                    full_matrices=False,
                    overwrite_a=True,
                    check_finite=False,
                    lapack_driver="gesvd",
                )
                new_bond = utils.truncated_bond(S, delta, max_rank)
                logger.debug(
                    "Truncating bond dimension from %s to %s", len(S), new_bond
                )
                qtt_cores.append(
                    U[:, :new_bond].reshape(
                        left_bond, row_mode_size, col_mode_size, new_bond
                    )
                )
                core = np.dot(np.diag(S[:new_bond]), V[:new_bond, :])
                left_bond = new_bond

            if i < self.n_cores - 1:
                # push the group remainder into the next core: the emitted core
                # becomes left-orthogonal and the boundary bond is truncated too
                U, S, V = svd(
                    core.reshape(left_bond * row_mode_size * col_mode_size, right_bond),
                    full_matrices=False,
                    overwrite_a=True,
                    check_finite=False,
                    lapack_driver="gesvd",
                )
                new_bond = utils.truncated_bond(S, delta, max_rank)
                qtt_cores.append(
                    U[:, :new_bond].reshape(
                        left_bond, row_mode_size, col_mode_size, new_bond
                    )
                )
                tt.cores[i + 1] = np.tensordot(
                    np.dot(np.diag(S[:new_bond]), V[:new_bond, :]),
                    tt.cores[i + 1],
                    axes=(1, 0),
                )
            else:
                qtt_cores.append(
                    core.reshape(left_bond, row_mode_size, col_mode_size, right_bond)
                )

        qtt = TensorTrain(qtt_cores)
        qtt.orth_center = qtt.n_cores - 1
        return qtt

    def qtt_to_tt(
        self,
        physical_indexes: Union[List[int], List[Tuple[int]]],
        error: float = 1e-12,
        max_rank: Optional[int] = None,
    ) -> "TensorTrain":
        """Convert a quantized tensor train back to a standard TT.

        Args:
            physical_indexes: Original TT physical dimensions.
            error: Allowed truncation error after reconstruction.
            max_rank: Optional maximum TT rank after reconstruction.

        Returns:
            TensorTrain: Reconstructed standard tensor train.

        Raises:
            DimensionMismatch: If the QTT dimensions do not match the original TT dimensions.
        """
        tt_cores = []
        powers = []

        # dimensionality check: walk the QTT cores once, consuming a group of
        # consecutive cores for each requested physical index
        pos = 0
        for i, physical_index in enumerate(physical_indexes):
            if self.is_ttm():
                row_target, col_target = physical_index
            else:
                row_target, col_target = physical_index, 1
            prod_row = 1
            prod_col = 1
            count = 0
            while count == 0 or prod_row != row_target or prod_col != col_target:
                if pos + count >= self.n_cores:
                    raise DimensionMismatch(
                        f"Not enough QTT cores to match physical index for core {i}"
                    )
                prod_row *= self.row_modes[pos + count]
                prod_col *= self.col_modes[pos + count]
                count += 1
                if prod_row > row_target:
                    raise DimensionMismatch(
                        f"Row mode size of QTT exceeds original row mode size for core {i}"
                    )
                if prod_col > col_target:
                    raise DimensionMismatch(
                        f"Column mode size of QTT exceeds original column mode size for core {i}"
                    )
            powers.append(count)
            pos += count
        if pos != self.n_cores:
            raise DimensionMismatch(
                f"Physical indexes cover only {pos} of {self.n_cores} QTT cores"
            )
        left_rank = 1
        stride = 0
        for i, physical_index in enumerate(physical_indexes):
            if self.is_ttm():
                row_mode_size = physical_index[0]
                col_mode_size = physical_index[1]
            else:
                row_mode_size = physical_index
                col_mode_size = 1

            big_core = self.cores[stride]
            p = powers[i]
            right_rank = self.bonds[p + stride]
            for j in range(p - 1):
                big_core = np.tensordot(
                    big_core, self.cores[stride + j + 1], axes=(-1, 0)
                )

            if p > 1:
                perm = (
                    [0]
                    + [1 + 2 * k for k in range(p)]
                    + [1 + 2 * k + 1 for k in range(p)]
                    + [1 + 2 * p]
                )
                big_core = big_core.transpose(tuple(perm))

            tt_cores.append(
                big_core.reshape(left_rank, row_mode_size, col_mode_size, right_rank)
            )
            left_rank = right_rank
            stride += p
        tt = TensorTrain(tt_cores)

        tt = tt.truncate(error=error, max_rank=max_rank)

        return tt


def tt_svd(
    tensor: np.ndarray,
    error: float = 0.0,
    max_rank: Optional[int] = None,
    physical_indexes: Optional[Union[List[int], List[Tuple[int]]]] = None,
    ttm: bool = False,
) -> TensorTrain:
    """TT-SVD algorithm for constructing a tensor train.
    Perform singular value decomposition of a tensor and
    decompose it into tensor train format.

    Args:
        tensor: n-dimensional array. If physical_indexes is None,
            tensor indexes will be treated as physical indexes.
        error: Error of tensor approximation. All singular values
            lower than error * ||A|| * (n-1)^{-1/2}, where ||.|| is
            the Frobenius norm.
        max_rank: If given, set a bound for bond dimension
        physical_indexes: physical indices for tensor train format.
            For TTM, rows and columns must be paired: [(i_1, j_1), (i_2, j_2), ...]
        ttm: If True, decompose tensor as tensor train matrix (MPO).
            In the TTM case, tensor indices are considered as follows:
            A(i_1, i_2, ..., i_n, j_1, ..., j_n)

    Returns:
        TensorTrain: Approximation of the tensor in tensor train
        format with the given error.

    Raises:
        AmbiguousFormat: If tensor is rank 1 and physical indexes are not provided or
            if tensor is Tensor Train Matrix, but physical indexes are not provided

    Reference:
        TT-SVD (algorithm No. 1) https://epubs.siam.org/doi/abs/10.1137/090752286"""

    if len(tensor.shape) == 1 and physical_indexes is None:
        raise AmbiguousFormat(
            "For 1 dimensional tensor physical indexes need to be provided"
        )

    new_tensor = tensor.copy()
    if physical_indexes is not None:
        if ttm:
            permute = tuple(
                np.arange(len(physical_indexes) * 2)
                .reshape([2, len(physical_indexes)])
                .transpose()
                .flatten()
            )
            new_tensor = np.transpose(new_tensor, permute)
            new_tensor = new_tensor.reshape(
                [x for pair in physical_indexes for x in pair]
            )
        else:
            new_tensor = new_tensor.reshape(physical_indexes)
            physical_indexes = [(x, 1) for x in physical_indexes]
    else:
        if ttm:
            raise AmbiguousFormat(
                "For Tensor Train Matrix type physical indexes is needed"
            )
        physical_indexes = new_tensor.shape
        physical_indexes = [(x, 1) for x in physical_indexes]

    tensor_norm = utils.tensor_norm(new_tensor)
    delta = error / np.sqrt(len(new_tensor.shape) - 1) * tensor_norm
    cores = []
    bond = 1
    discarded_squared = 0.0
    for i in range(len(physical_indexes) - 1):
        matrix = new_tensor.reshape(
            (
                bond * prod(physical_indexes[i]),
                prod([prod(pair) for pair in physical_indexes[i + 1 :]]),
            )
        )
        U, S, V = svd(
            matrix,
            full_matrices=False,
            overwrite_a=True,
            check_finite=False,
            lapack_driver="gesvd",
        )
        next_bond = utils.truncated_bond(S, delta, max_rank)
        discarded_squared += float(np.sum(S[next_bond:] ** 2))
        U, C = (
            U[:, :next_bond].reshape((bond,) + physical_indexes[i] + (next_bond,)),
            np.diag(S[:next_bond]) @ V[:next_bond, :],
        )
        cores.append(U)
        next_bond = C.shape[0]
        if i == len(physical_indexes) - 2:
            cores.append(C.reshape((next_bond,) + physical_indexes[i + 1] + (1,)))
        else:
            bond = next_bond
            new_tensor = C
    tt = TensorTrain(cores)
    tt.orth_center = tt.n_cores - 1
    achieved = np.sqrt(discarded_squared)
    logger.debug(
        "tt_svd: requested relative error %.3e, achieved %.3e "
        "(absolute %.3e, budget %.3e per split); ranks %s",
        error,
        achieved / tensor_norm if tensor_norm else achieved,
        achieved,
        delta,
        tt.bonds,
    )
    return tt


def ones(
    physical_indexes: Union[List[int], List[Tuple[int]]],
    ranks: Union[int, List[int]] = 1,
) -> TensorTrain:
    """Create a tensor train filled with ones.

    Args:
        physical_indexes: List of physical dimensions for each core. For
            Tensor Train Matrix, each physical dimension should be a tuple
            of (row_dim, col_dim).
        ranks: List of TT ranks. Defaults to [1, ..., 1].

    Returns:
        TensorTrain: Tensor train filled with ones.
    """
    if len(physical_indexes) < 2:
        raise WrongTTFormat("Tensor train must have at least two cores.")

    if not isinstance(ranks, list):
        ranks = [1] + [ranks for _ in range(len(physical_indexes) - 1)] + [1]

    if isinstance(physical_indexes[0], int):
        cores = [
            np.ones([ranks[i], physical_indexes[i], 1, ranks[i + 1]])
            for i in range(len(physical_indexes))
        ]
    else:
        cores = [
            np.ones(
                [ranks[i], physical_indexes[i][0], physical_indexes[i][1], ranks[i + 1]]
            )
            for i in range(len(physical_indexes))
        ]
    tt_ones = TensorTrain(cores)

    return tt_ones


def zeros(
    physical_indexes: Union[List[int], List[Tuple[int]]],
    ranks: Union[int, List[int]] = 1,
) -> TensorTrain:
    """Create a tensor train filled with zeros.

    Args:
        physical_indexes: List of physical dimensions for each core. For
            Tensor Train Matrix, each physical dimension should be a tuple
            of (row_dim, col_dim).
        ranks: List of TT ranks. Defaults to [1, ..., 1].

    Returns:
        TensorTrain: Tensor train filled with zeros.
    """
    if len(physical_indexes) < 2:
        raise WrongTTFormat("Tensor train must have at least two cores.")

    if not isinstance(ranks, list):
        ranks = [1] + [ranks for _ in range(len(physical_indexes) - 1)] + [1]

    if isinstance(physical_indexes[0], int):
        cores = [
            np.zeros([ranks[i], physical_indexes[i], 1, ranks[i + 1]])
            for i in range(len(physical_indexes))
        ]
    else:
        cores = [
            np.zeros(
                [ranks[i], physical_indexes[i][0], physical_indexes[i][1], ranks[i + 1]]
            )
            for i in range(len(physical_indexes))
        ]
    tt_zeros = TensorTrain(cores)

    return tt_zeros


def eye(physical_indexes: List[int]) -> TensorTrain:
    """Create an identity tensor train.

    Args:
        physical_indexes: List of physical dimensions for each core. Row and
            column dimensions are the same. All ranks are 1.

    Returns:
        TensorTrain: Identity tensor train.
    """
    if len(physical_indexes) < 2:
        raise WrongTTFormat("Tensor train must have at least two cores.")

    cores = [
        np.zeros([1, physical_indexes[i], physical_indexes[i], 1])
        for i in range(len(physical_indexes))
    ]
    for i, ph in enumerate(physical_indexes):
        cores[i][0, :, :, 0] = np.eye(ph)
    tt_ones = TensorTrain(cores)

    return tt_ones


def rand(
    physical_indexes: Union[List[int], List[Tuple[int]]],
    ranks: Union[int, List[int]] = 1,
    seed=None,
) -> TensorTrain:
    """Create a random tensor train.

    Args:
        physical_indexes: List of physical dimensions for each core. For
            Tensor Train Matrix, each physical dimension should be a tuple
            of (row_dim, col_dim).
        ranks: List of TT ranks. Defaults to [1, ..., 1].

    Returns:
        TensorTrain: Random tensor train.
    """
    if len(physical_indexes) < 2:
        raise WrongTTFormat("Tensor train must have at least two cores.")

    rng = np.random.default_rng(seed)
    if not isinstance(ranks, list):
        ranks = [1] + [ranks for _ in range(len(physical_indexes) - 1)] + [1]

    if isinstance(physical_indexes[0], int):
        cores = [
            rng.random((ranks[i], physical_indexes[i], 1, ranks[i + 1]))
            for i in range(len(physical_indexes))
        ]
    else:
        cores = [
            rng.random(
                (ranks[i], physical_indexes[i][0], physical_indexes[i][1], ranks[i + 1])
            )
            for i in range(len(physical_indexes))
        ]
    tt_rand = TensorTrain(cores)

    return tt_rand


def canonical_tt(physical_indexes: List[int], max_rank: int) -> TensorTrain:
    """Create a full-rank tensor train consisting of tensor products of the canonical basis.

    Tensor train is created in the following format:
    A^1 A^2 ... A^{n//2} B^{n//2+1} ... B^{n}
    where A^i are left-orthogonal cores and B^i are right-orthogonal cores.
    For odd n, the middle core is left-orthogonal.

    Bond k (connecting cores k-1 and k) is set to
    min(max_rank, prod(dims[:k]), prod(dims[k:])), i.e. it is capped by the
    number of basis states on either side of the cut, so the left and right
    halves always meet with matching bond dimensions.

    Args:
        physical_indexes: List of physical dimensions for each core. Tensor
            Train Matrix format is not supported.
        max_rank: Maximum rank of the TT decomposition.

    Returns:
        TensorTrain: Canonical tensor train.
    """
    order = len(physical_indexes)
    if order < 2:
        raise WrongTTFormat("Tensor train must have at least two cores.")

    bonds = [1] * (order + 1)
    for k in range(1, order):
        bonds[k] = min(
            max_rank,
            prod(physical_indexes[:k]),
            prod(physical_indexes[k:]),
        )

    cores = [None] * order
    split = order // 2

    # left-orthogonal cores
    for i in range(split):
        cores[i] = np.eye(bonds[i] * physical_indexes[i], bonds[i + 1]).reshape(
            (bonds[i], physical_indexes[i], 1, bonds[i + 1])
        )

    # right-orthogonal cores
    for i in range(split + order % 2, order):
        cores[i] = np.eye(bonds[i], physical_indexes[i] * bonds[i + 1]).reshape(
            (bonds[i], physical_indexes[i], 1, bonds[i + 1])
        )

    # center core for odd order (left orthogonal)
    if order % 2 == 1:
        cores[split] = np.eye(
            bonds[split] * physical_indexes[split], bonds[split + 1]
        ).reshape((bonds[split], physical_indexes[split], 1, bonds[split + 1]))

    tt_canonical = TensorTrain(cores)
    tt_canonical.orth_center = split if order % 2 == 1 else split - 1
    return tt_canonical
