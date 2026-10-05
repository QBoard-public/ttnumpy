Getting Started
===============

`TTNumPy` is an open-source toolbox for working with Tensor Trains (TT) and
Tensor Train Matrices (TTM). It provides the tools to build, manipulate and
compute with tensor trains, and the variational solvers that make the format
useful for large numerical problems.

This short guide highlights the main features of `TTNumPy` and shows simple
examples. It assumes you know the basics of tensor networks (TT = matrix product
state). If you're new to the area, the Tensornetwork website is a good starting
point: https://www.tensornetwork.org/, and the :doc:`user_guide` builds the
concepts up from the beginning.

Install `TTNumPy` following the instructions in the :doc:`installation` guide.

Creation of TT and TTM
----------------------

There are a few ways to create a tensor train. You can build one manually from a list
of cores and construct it from a full tensor or randomly using the provided helpers.

Manual construction
~~~~~~~~~~~~~~~~~~~

Create a TensorTrain by supplying a list of cores. Each core is a NumPy array with
shape :math:`(r_i, n_i, r_{i+1})` for TT (or rank-4 for TTM). For example:

.. code-block:: python

   import numpy as np
   import ttnumpy as ttn

   # three cores for a TT with mode sizes [2, 3, 4]
   core1 = np.random.randn(1, 2, 3)
   core2 = np.random.randn(3, 3, 4)
   core3 = np.random.randn(4, 4, 1)

   tt = ttn.TensorTrain([core1, core2, core3])

   # or for a TTM with physical dimensions (2,2) and (3,2):
   core1 = np.random.randn(1, 2, 2, 3)   # r0=1, row_mode=2, col_mode=2, r1=3
   core2 = np.random.randn(3, 3, 2, 1)   # r1=3, row_mode=3, col_mode=2, r2=1

   ttm = ttn.TensorTrain([core1, core2])

Important note: even though TT consist of rank-3 cores, in `TTNumPy` they 
are stored as rank-4 cores with column dimension 1. This is to unify the interface with TTM, 
which have rank-4 cores and non-unit column dimensions.

Constructors: SVD and random
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can build a TT from a full tensor using SVD-based decomposition:

.. code-block:: python

   import numpy as np
   from ttnumpy import tt_svd

   tensor = np.arange(24).reshape(2, 3, 4)
   tt_tensor = tt_svd(tensor, error=1e-10)

Or create a random tensor train:

.. code-block:: python

   from ttnumpy import rand

   tt_random = rand([2, 3, 4], ranks=3, seed=42)

Both approaches return a `ttnumpy.TensorTrain` instance. For more constructors
and details see the API reference in the docs.

Arithmetic operations
---------------------

`TTNumPy` supports standard arithmetic on tensor trains. You can use Python operators
or the class methods to combine and manipulate TTs.

You can perform arithmetic using the full class methods (recommended for clarity)
and also the Python operator shortcuts. The main methods are:

- `TensorTrain.add(other, truncate=False)` — sum two tensor trains.
- `TensorTrain.multiply(other, truncate=False)` — element-wise product (or multiply by scalar).
- `TensorTrain.dot(other, truncate=False)` — matrix-by-vector / matrix-by-matrix product.

Examples using the explicit methods:

.. code-block:: python

   from ttnumpy import rand

   tt1 = rand([2, 3, 4], ranks=3, seed=1)
   tt2 = rand([2, 3, 4], ranks=4, seed=2)

   # assume tt1 and tt2 have the same physical dimensions
   tt_sum = tt1.add(tt2)           # explicit method call
   tt_prod = tt1.multiply(tt2)     # element-wise product via method

   # or, using operator shortcuts:
   tt_sum2 = tt1 + tt2             # same as tt1.add(tt2)
   tt_prod2 = tt1 * tt2            # same as tt1.multiply(tt2)


Matrix-by-vector (and matrix-by-matrix) products use the `@` operator (or
`TensorTrain.dot`):

.. code-block:: python

   from ttnumpy import rand

   x = rand([2, 3, 4], ranks=3, seed=1) # random TT vector
   A = rand([(2, 2), (3, 3), (4, 4)], ranks=4, seed=2) # random TTM (matrix-like)

   y = A.dot(x)
   y = A @ x # result is a TT vector with the same physical dimensions as x

Solving linear systems
----------------------
A discretized PDE, a least-squares fit or a stationary problem all come down to
solving a system of linear equations, often one far too large to form as a dense
matrix. `TTNumPy` provides several variational algorithms for this, which one
can find in :mod:`ttnumpy.solvers`.

Solver example
~~~~~~~~~~~~~~
A minimal example showing how to solve a linear system with the MALS and
AMEn solvers.

.. code-block:: python

   import numpy as np
   from ttnumpy import eye, rand
   from ttnumpy import amen, mals

   # build a simple identity operator A with two local dimensions (2, 3)
   A = eye([2, 3])   # returns a TTM identity with local dims 2 and 3

   # random right-hand side as a TT (vector-like)
   b = rand([2, 3], ranks=2, seed=1)

   # solve Ax = b with MALS, stopping at a relative residual of 1e-8
   x = mals(A, b, max_sweeps=5, residual_tol=1e-8)

   # the same system with the rank-adaptive AMEn solver
   x = amen(A, b, max_sweeps=5, residual_tol=1e-8, error_threshold=1e-10)

   # x is a TensorTrain instance approximating the solution

See :doc:`user_guide/linear_systems` for how the solvers work, how to choose
between them, and :mod:`ttnumpy.solvers` for the full list of
options.

Next step
----------------------
That is a brief introduction to the tools `TTNumPy` makes available. For
further information, please refer to the user guide: :doc:`user_guide`.