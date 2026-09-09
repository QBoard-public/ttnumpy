User guide
==========

This guide is an introduction to tensor networks and the tensor train
format, not only to the library that implements them. It builds up from the
problem the format solves, through the decomposition and its numerics, to
the algorithms that operate in it.

:doc:`user_guide/motivation` explains why low-rank representations exist and
introduces tensor networks. :doc:`user_guide/tt_numerics` covers the working
mechanics: construction, accuracy, complexity and quantization.
:doc:`user_guide/linear_systems` covers the variational solvers.

.. toctree::
   :maxdepth: 2

   user_guide/motivation
   user_guide/tt_numerics
   user_guide/linear_systems
