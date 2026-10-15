TTNumPy documentation
========================

**TTNumPy** is a tensor-train toolbox for numerical linear algebra. It
provides tools for solving optimization problems in the tensor-train
formalism. Problems that this formalism handles efficiently include systems
of linear equations, black-box function interpolation and numerical
integration.

The library works with tensor trains (TT) and tensor train matrices (TTM),
known in the physics literature as matrix product states and matrix product
operators. It is written in Python on top of NumPy and SciPy and pulls in no
other runtime dependencies.

.. rubric:: Features

- **Tensor train container.** ``TensorTrain`` holds both the TT and the TTM
  format, tracking bond dimensions, canonical form and the position of the
  orthogonality centre.
- **Constructors.** TT-SVD decomposition of a full tensor, plus ``ones``,
  ``zeros``, ``eye``, ``rand`` and ``canonical_tt``.
- **Arithmetic.** Addition, element-wise and scalar multiplication,
  matrix-by-vector and matrix-by-matrix products, transposition and
  conjugation, each with optional rounding.
- **Quantization.** Conversion between TT and quantized tensor train (QTT)
  form, which turns a grid of :math:`2^n` points into :math:`n` cores of
  mode size 2.
- **Linear solvers.** ALS, MALS and AMEn for :math:`A x = b` in TT format,
  sharing environments, local solvers and stopping criteria.
- **Tensor cross interpolation.** ``tci`` builds a tensor train from a
  function of indices, evaluating only a small part of the tensor.
  ``add_global_pivots`` adds configurations the interpolation must cover.
  ``recompress`` lowers its ranks.
- **Benchmarks.** A runner that measures the TT solvers against SciPy CG,
  PyAMG and PETSc on the Poisson equation.

.. rubric:: At a glance

.. code-block:: python

   import ttnumpy as tt

   A = tt.eye([2, 2, 2])                      # TTM operator
   b = tt.rand([2, 2, 2], ranks=2, seed=0)    # TT right-hand side

   x, info = tt.amen(A, b, residual_tol=1e-8, return_info=True)
   print(info.stop_reason, info.residuals[-1])

Start with :doc:`installation`, then follow :doc:`getting_started` for a tour
of the main features. The :doc:`user_guide` covers tensor trains and the
algorithms in depth, and :doc:`api` documents every public entry point.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   getting_started
   user_guide
   examples
   api
