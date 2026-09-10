Motivation
==========

Before any algorithm, the question of why a special format is needed at
all. This part answers it. The first chapter explains why the arrays that
describe multi-variable systems cannot be stored, and why that turns out
not to matter. The second introduces the networks that exploit the gap.

Read this part if tensor networks are new to you. If you already work with
MPS/TT or MPO/TTM, the second chapter still fixes the notation and the naming
conventions used throughout the rest of the guide.

.. _why-low-rank:

Low-rank representations
----------------------------

Every method in this library exists to avoid writing down one particular
array. This chapter explains which array that is, why it cannot be written
down, and why it usually does not have to be.

The exponential wall
~~~~~~~~~~~~~~~~~~~~

Take a system described by :math:`d` variables, each taking :math:`n`
values: :math:`d` particles with :math:`n` internal states, :math:`d` grid
axes with :math:`n` points, :math:`d` parameters sampled at :math:`n`
settings. A general state of that system is an array

.. math::

   A[i_1, i_2, \ldots, i_d], \qquad i_k = 1, \ldots, n,

with :math:`n^d` entries. The cost of storing it does not grow with the
number of variables. Instead, it grows *exponentially* in the number of variables.
For double precision:

==========  ===================  ==================
:math:`d`   Entries              Memory
==========  ===================  ==================
10          :math:`2^{10}`       8 KB
30          :math:`2^{30}`       8.6 GB
50          :math:`2^{50}`       9.0 PB
100         :math:`2^{100}`      :math:`10^{16}` EB
==========  ===================  ==================

At :math:`n = 2` the wall arrives around :math:`d = 40`. Nothing about the
hardware moves it meaningfully: every variable added doubles the
requirement, so a thousandfold increase in memory buys ten more variables.

This is not a statement about quantum mechanics in particular. It is a
statement about product spaces. The state space of the whole is the tensor
product of the state spaces of the parts, and dimensions multiply. So everytime
when problem include kroneker products, tensor products, or multi-dimensional arrays,
the exponential wall is there.

The states we care about are not generic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The wall is real but the conclusion — that these problems are hopeless — is
not, because the arrays that arise in practice occupy a vanishingly small
corner of that space.

Make the statement precise by cutting the variables into two groups,
:math:`\{i_1 \ldots i_k\}` and :math:`\{i_{k+1} \ldots i_d\}`, and
reshaping :math:`A` into a matrix :math:`A_k` across that cut. Its singular
value decomposition

.. math::

   A_k = \sum_{\alpha=1}^{r_k} \sigma_\alpha \, u_\alpha \, v_\alpha^{H},
   \qquad \sigma_1 \ge \sigma_2 \ge \cdots \ge 0,

is the *Schmidt decomposition* across the cut. The number of nonzero
:math:`\sigma_\alpha` is the **Schmidt rank** :math:`r_k`, and the
distribution of the :math:`\sigma_\alpha` is what decides whether :math:`A`
is compressible. Writing :math:`p_\alpha = \sigma_\alpha^2 / \sum_\beta
\sigma_\beta^2`, the **entanglement entropy** across the cut is

.. math::

   S_k = -\sum_{\alpha} p_\alpha \log p_\alpha .

:math:`S_k` measures how much the two halves must know about each other.
Two regimes matter:

**Volume law.** A state drawn at random from the full space has
:math:`S_k` proportional to the number of variables on the smaller side of
the cut — the *volume*. Its singular values are all comparable, nothing can
be discarded, and no compression is possible. Almost every array in the
space is of this kind.

**Area law.** The states that physics and numerics actually produce are
not random. For the ground state of a gapped local Hamiltonian in one
dimension, :math:`S_k` is bounded by a constant independent of system size
[Hastings2007]_ — proportional to the size of the *boundary* between the
two halves, which in a 1D chain is a single point. The singular values then
decay fast, typically exponentially, and truncating them costs almost
nothing. Area laws and their range of validity are surveyed in
[Eisert2010]_.

The gap between these two regimes is the whole opportunity. An area-law
state needs a Schmidt rank :math:`r` that does not grow with :math:`d`,
so it is fixed by :math:`O(d n r^2)` numbers rather than :math:`n^d`.

The same argument outside physics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Nothing above used the Schrödinger equation. The Schmidt rank is a property
of a reshaped array, and arrays from classical numerics are structured in
exactly the same way:

- **Smooth functions on a grid.** Sampling :math:`f(x) = e^{3x}` on
  :math:`2^{L}` points gives an array whose every cut has Schmidt rank 1,
  at any :math:`L`. A polynomial of degree :math:`p` gives rank
  :math:`p + 1`. These are measured in :ref:`qtt`.
- **Discretized differential operators.** A finite-difference Laplacian
  couples each point to its neighbours only, so a cut through the grid
  carries a couple of numbers, not a matrix. The 2D Laplacian used by this
  library's benchmarks has bond dimension 6 — at every grid size.
- **Low-rank structure by construction.** Sums of separable terms, Green's
  functions, and covariance kernels all decay fast across a cut.

What breaks the argument is genuine randomness, and the failure is easy to
see: white noise on the same :math:`2^{12}` grid needs the maximal rank of
64. Structure is the resource being spent, and a format that exploits it
gives nothing back when there is none.

**So, we do not need all the information in the array -- we need the part that
carries the physics, and that part is usually small.**

Tensor networks are the machinery for keeping only that part. The next
chapter introduces them. :ref:`tt-svd` shows how to extract one
from an array and :ref:`tt-accuracy` shows how to decide what to discard.

.. _tensor-networks:

Tensor networks
---------------

A tensor network represents one large tensor as a set of small tensors
contracted together. This chapter fixes the notation, shows the common
network topologies, and then narrows to the one this library implements:
the tensor train.

Tensors and diagrams
~~~~~~~~~~~~~~~~~~~~

A tensor is an array with several indices; the number of indices is its
*order*. A scalar has order 0, a vector order 1, a matrix order 2. In
diagram notation a tensor is a node and each index is a leg:

.. figure:: /images/tensors.png
   :width: 45%
   :align: center

   The diagram notation: a node per tensor, a leg per index.

Joining two legs means summing over that index — a *contraction*. 
Self contraction is known as a trace. Matrix multiplication 
:math:`C[i,k] = \sum_j A[i,j] B[j,k]` is one shared leg:

.. figure:: /images/matrix-mult.png
   :width: 60%
   :align: center

   Matrix multiplication as a contraction: the shared leg is summed 
   over and disappears.

So, the important rules of the diagram are:

- A leg left open is an index of the result.
- A leg joining two nodes is summed over and disappears.
- The cost of contracting an index is the product of *all* dimensions
  involved, so the order of contractions matters enormously.

Common topologies
~~~~~~~~~~~~~~~~~

Networks are classified by how the nodes are wired.

**Matrix product state (MPS) / tensor train (TT).** Nodes in an open
chain, one open leg each:

.. figure:: /images/tt.png
   :width: 75%
   :align: center

   A tensor train: an open chain of cores, one physical leg each.

**Matrix product operator (MPO) / tensor train matrix (TTM).** The same
chain with two open legs per node, so it maps one chain to another — the
tensor-network form of a matrix.

.. figure:: /images/ttm.png
   :width: 75%
   :align: center

   A tensor train matrix: the same chain with row and column legs, 
   mapping one train to another.

**PEPS.** A 2D lattice of nodes. It captures 2D area laws that a chain
cannot, but contracting it exactly is computationally hard in general, so
it needs approximate contraction schemes.

**MERA.** A layered, hierarchical network aimed at critical systems, where
entanglement grows logarithmically and the plain area law fails.

This library implements the chain: TT and TTM. The reason is not that
chains are the most expressive topology but that they are the most
*tractable* one. On an open chain every contraction, norm, inner product
and truncation is exact and costs a polynomial in the bond dimension —
there is no approximate contraction step to worry about, which is what
makes reliable numerics possible.

The tensor train format
~~~~~~~~~~~~~~~~~~~~~~~

A tensor train writes each entry of :math:`A` as a product of matrices,
one per index:

.. math::

   A[i_1, i_2, \ldots, i_d] \;=\;
   G_1[i_1] \, G_2[i_2] \cdots G_d[i_d],

where :math:`G_k[i_k]` is an :math:`r_{k-1} \times r_k` matrix. Collecting
the matrices for all values of :math:`i_k` gives the order-3 **core**

.. math::

   G_k \in \mathbb{C}^{\,r_{k-1} \times n_k \times r_k},

and the product is a scalar because the train has *open boundary
conditions*: :math:`r_0 = r_d = 1`. The numbers :math:`r_k` are the **bond
dimensions** or TT-ranks. They are the Schmidt ranks of
:ref:`why-low-rank` — bond :math:`k` is exactly the cut between
:math:`\{i_1 \ldots i_k\}` and the rest.

The saving is immediate. Writing :math:`n = \max n_k` and :math:`r = \max
r_k`, a train stores

.. math::

   O(d \, n \, r^2) \quad\text{numbers instead of}\quad n^d .

Linear in the number of variables, quadratic in the bond dimension. The
exponential is gone, and what replaces it is :math:`r` — which is small
exactly when the array obeys an area law.

Cores in this library
~~~~~~~~~~~~~~~~~~~~~

:class:`~ttnumpy.TensorTrain` stores both formats uniformly, as
order-4 cores

.. code-block:: text

   ranks[k] x row_modes[k] x col_modes[k] x ranks[k+1]

with ``col_modes[k] == 1`` for a TT and the full column dimension for a
TTM. Construct one from a list of cores:

.. code-block:: python

   import numpy as np
   import ttnumpy as tt

   # a TT with mode sizes [2, 3, 4] and bonds [1, 3, 4, 1]
   train = tt.TensorTrain([
       np.random.randn(1, 2, 3),
       np.random.randn(3, 3, 4),
       np.random.randn(4, 4, 1),
   ])

   print(train.bonds)           # [1, 3, 4, 1]
   print(train.physical_dims())  # [2, 3, 4]
   print(train.is_ttm())        # False

Rank-3 cores are accepted and read as TT; rank-4 cores are read as TTM.
The first and last cores must carry the dummy bonds :math:`r_0 = r_d = 1`,
and neighbouring bonds must match, or the constructor raises
``WrongTTFormat`` or ``DimensionMismatch``.

A TTM is built the same way, with a genuine column dimension:

.. code-block:: python

   # a TTM mapping (2, 3) -> (2, 2)
   operator = tt.TensorTrain([
       np.random.randn(1, 2, 2, 3),
       np.random.randn(3, 3, 2, 1),
   ])

   print(operator.is_ttm())       # True
   print(operator.physical_dims())  # [(2, 2), (3, 2)]

Naming
~~~~~~

The physics and numerical-analysis communities developed these objects
independently, so most of them have two names. They are the same thing:

=========================  =========================
Numerical analysis         Physics
=========================  =========================
Tensor train (TT)          Matrix product state (MPS)
Tensor train matrix (TTM)  Matrix product operator (MPO)
TT-rank / bond dimension   Bond dimension
TT-rounding                Compression / truncation
Orthogonality centre       Canonical centre
=========================  =========================

This documentation uses the TT vocabulary, matching the API. The MPS
literature is worth reading regardless; [Schollwoeck2011]_ is the standard
review, and the format was introduced to numerical analysis in
[Oseledets2011]_.

Building a train is the subject of the next chapter.

References
----------

.. [Hastings2007] M. B. Hastings, *An area law for one-dimensional quantum
   systems*, J. Stat. Mech. (2007) P08024.
   https://doi.org/10.1088/1742-5468/2007/08/P08024

.. [Eisert2010] J. Eisert, M. Cramer and M. B. Plenio, *Colloquium: Area
   laws for the entanglement entropy*, Rev. Mod. Phys. 82, 277 (2010).
   https://doi.org/10.1103/RevModPhys.82.277

.. [Oseledets2011] I. V. Oseledets, *Tensor-train decomposition*, SIAM J.
   Sci. Comput. 33(5), 2295–2317, 2011.
   https://doi.org/10.1137/090752286

.. [Schollwoeck2011] U. Schollwöck, *The density-matrix renormalization
   group in the age of matrix product states*, Annals of Physics 326(1),
   96–192, 2011. https://arxiv.org/abs/1008.3477
