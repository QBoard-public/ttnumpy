Installation
============

TTNumPy requires Python 3.11 or newer. Its only runtime dependencies are
NumPy and SciPy.

Install the package and its runtime dependencies from the project root:

.. code-block:: bash

   python -m pip install .

For development work, install the editable package with the developer extras:

.. code-block:: bash

   python -m pip install -e '.[dev]'

For documentation-only work, install the docs extras:

.. code-block:: bash

   python -m pip install -e '.[docs]'

The benchmark suite compares the tensor-train solvers against classical
backends and needs PyAMG and PETSc on top of the runtime dependencies:

.. code-block:: bash

   python -m pip install -e '.[benchmark]'

Building PETSc from source expects BLAS and LAPACK development libraries to be
available on the system.

Check the installation
----------------------

.. code-block:: bash

   python -c "import ttnumpy as tt; print(tt.eye([2, 2]))"
