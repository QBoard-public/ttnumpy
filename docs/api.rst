API Reference
=============

Every public entry point, grouped by module. ``ttnumpy`` is the one
most code needs: it re-exports the container, the constructors and the
solvers, so ``import ttnumpy as tt`` reaches all of them.

Tensor trains and solvers
-------------------------

The public surface: the :class:`~ttnumpy.TensorTrain` container, the
constructors, the three linear solvers and the types they report with.

.. automodule:: ttnumpy
   :members:
   :undoc-members:
   :show-inheritance:

Exceptions
----------

Raised by the container and its constructors on malformed input. They live
in ``ttnumpy.tensor_train`` and are not re-exported from
``ttnumpy``, so import them from their defining module.

.. autoexception:: ttnumpy.tensor_train.DimensionMismatch
.. autoexception:: ttnumpy.tensor_train.WrongTTFormat
.. autoexception:: ttnumpy.tensor_train.WrongTTType
.. autoexception:: ttnumpy.tensor_train.AmbiguousFormat

Solver internals
----------------

.. py:module:: ttnumpy.solvers

The three solvers are implemented in the ``ttnumpy.solvers``
package — one module per method (``als``, ``mals``, ``amen``) over a shared
``common`` module holding the environments, the local solves and the
stopping machinery — and are re-exported from ``ttnumpy`` above.
Two-site and one-site local systems live in ``two_site`` and ``one_site``
respectively.

Low-level utilities
-------------------

Norms, orthonormalization and SVD-based truncation, used by the container
and the solvers. Most code should prefer the ``TensorTrain`` methods that
wrap these.

.. automodule:: ttnumpy.utils
   :members:
   :undoc-members:
   :show-inheritance:

Logging
-------

Application-facing logging setup. The library attaches no handlers of its
own; call :func:`~ttnumpy.logger_config.setup_logging` once from a script or
CLI, or configure the ``ttnumpy`` logger directly.

.. automodule:: ttnumpy.logger_config
   :members:
   :undoc-members:
   :show-inheritance:

Benchmarks
----------

The benchmark harness: the Poisson problem in both sparse and TT form, and
the runner that times the TT solvers against SciPy, PyAMG and PETSc. This
is research and CI tooling rather than library API.

.. automodule:: benchmarks.poisson_problem
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: benchmarks.poisson
   :members:
   :undoc-members:
   :show-inheritance:
