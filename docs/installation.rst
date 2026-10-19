Installation
============

TTNumPy requires Python 3.11 or newer. Its only runtime dependencies are
NumPy and SciPy.

Install the package and its runtime dependencies from the project root:

.. code-block:: bash

   python -m pip install .

For development work, install the editable package with the ``dev`` dependency
group. Dependency groups need pip 25.1 or newer:

.. code-block:: bash

   python -m pip install -U pip
   python -m pip install -e . --group dev

For documentation-only work, install the ``docs`` group:

.. code-block:: bash

   python -m pip install -e . --group docs

The benchmark suite compares the tensor-train solvers against classical
backends and needs PyAMG and PETSc on top of the runtime dependencies:

.. code-block:: bash

   python -m pip install -e . --group benchmark

Building PETSc from source expects BLAS and LAPACK development libraries to be
available on the system.

Check the installation
----------------------

.. code-block:: bash

   python -c "import ttnumpy as tt; print(tt.eye([2, 2]))"
