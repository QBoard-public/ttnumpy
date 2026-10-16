Tensor cross interpolation
==========================

.. _tci-intro:

Introduction
------------

TT-SVD builds a tensor train from a full array and the solvers of
:doc:`linear_systems` build one from an equation. This part builds one from
a function. Given a way to evaluate any single entry of a tensor, **tensor
cross interpolation** (TCI) finds a tensor train for it from a small number
of entries, which it chooses as it goes.

Such tensors typically hold a smooth function sampled on a grid. Take

.. math::

   F(x_0, \ldots, x_{d-1}) = \frac{1}{1 + x_0^2 + x_1^2 + \cdots + x_{d-1}^2}

on 16 nodes in :math:`[0, 1]` per variable. With :math:`d = 10` variables
the tensor has :math:`16^{10} \approx 1.1 \cdot 10^{12}` entries, 8.8 TB in
double precision. It is compressible all the same. For a smooth function the
singular values of every unfolding decay fast, as they did for the functions
of :ref:`qtt`, so a tensor train with small ranks approximates it closely.
That is a property of the function: nothing about the cores is known in
advance, only how to evaluate :math:`F`.

TT-SVD cannot find this train. It starts from the full array, which is the
wall of :ref:`why-low-rank` again: a low-rank train exists, but TT-SVD reads
every entry to find it.

TCI evaluates :math:`F` only where it decides to look:

.. code-block:: python

   import numpy as np
   import ttnumpy as tt

   d = 10
   nodes = np.linspace(0.0, 1.0, 16)

   def f(indices):
       return 1.0 / (1.0 + sum(nodes[i] ** 2 for i in indices))

   tensor_ci = tt.tci(f, [16] * d, initial_pivots=[[0] * d], error_threshold=1e-8)
   train = tensor_ci.to_tensor_train()

   print(train.bonds)              # [1, 7, 7, 8, 8, 8, 8, 8, 7, 7, 1]
   print(tensor_ci.box.n_evals)    # 163912

The train approximates :math:`F` with a largest error of
:math:`1.4 \cdot 10^{-8}` on 10 000 random entries. It was built from
163 912 values of :math:`F`, about one entry in 6.7 million.

The method is TT-SVD with one substitution. At every bond TT-SVD factorizes
an unfolding by an SVD, which reads the whole unfolding. TCI factorizes the
same unfolding by a *cross* built from a few of its rows and columns. Only
those are evaluated. :ref:`tci-form` derives the train this produces from
TT-SVD step by step. :ref:`tci-algorithm` shows how the rows and columns
are found. The remaining chapters cover functions on grids, features the
method can miss, recompression, accuracy and cost.

Cross approximation of matrices goes back to the pseudo-skeleton
decompositions of [GTZ1997]_. Its tensor-train form is TT-cross [OT2010]_.
The implementation follows [NF2025]_, which builds on [NF2022]_ and
formulates the algorithm around a partially rank-revealing LU
decomposition. Variables and sites are numbered from 0 throughout this part,
as in the code. The part assumes :doc:`tt_numerics`.

.. _tci-form:

The TCI form
------------

This chapter fixes what TCI produces before saying how it gets there. It
factorizes one unfolding with a cross, then repeats the cross at every bond
the way TT-SVD repeats its SVD. The result, the TCI form, is a tensor train
written entirely in sampled values of the tensor.

One cut: matrix cross interpolation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Start with two variables, where the tensor is a matrix,
:math:`A[i, j] = \sin(x_i + y_j)`. Since
:math:`\sin(x + y) = \sin x \cos y + \cos x \sin y`, it has rank 2. A matrix
of rank :math:`r` is fixed by :math:`r` of its rows and :math:`r` of its
columns. For rows :math:`I` and columns :math:`J` with
:math:`|I| = |J| = r`,

.. math::

   A = A[:, J] \; P^{-1} \; A[I, :], \qquad P = A[I, J],

whenever :math:`A` has rank :math:`r` and :math:`P` is invertible. This is
the **cross interpolation** of :math:`A` [GTZ1997]_: the chosen columns,
times the inverse of the block where they cross the chosen rows, times the
chosen rows. The chosen rows and columns are the **pivots**. :math:`P` is the
**pivot matrix**.

.. figure:: /images/tci-matrix-cross.png
   :width: 100%
   :align: center

   Cross interpolation of a rank-2 matrix. The pivot rows :math:`I` and the
   pivot columns :math:`J` cross in the pivot matrix :math:`P`, whose entries
   are marked in each factor of :math:`A[:, J] \, P^{-1} A[I, :]`.

On a :math:`1000 \times 1000` grid, two rows and two columns reproduce the
whole matrix:

.. code-block:: python

   import numpy as np

   x = np.linspace(0.0, 3.0, 1000)
   A = np.sin(x[:, None] + x[None, :])

   rows, cols = [100, 700], [200, 900]
   cross = A[:, cols] @ np.linalg.solve(A[np.ix_(rows, cols)], A[rows, :])

   print(f"{np.abs(cross - A).max():.1e}")    # 1.2e-15

The cross reads 3 996 of the :math:`10^6` entries.

Compare it with the SVD that TT-SVD applies at every bond. Both write
:math:`A` as a product through an :math:`r`-dimensional middle; they differ
in the basis of that middle. The SVD uses singular vectors, which minimize
the error in the Frobenius norm and depend on every entry. The cross uses
rows and columns of :math:`A` itself, which are entries. Two properties
follow.

**It interpolates.** On a pivot row :math:`i \in I` the factor
:math:`A[i, J]` is a row of :math:`P`, which :math:`P^{-1}` turns into a unit
vector. The product then returns row :math:`i` of :math:`A[I, :]` unchanged.
The same holds for the pivot columns. The cross reproduces :math:`A` exactly
on every pivot row and column, whatever the rank of :math:`A`, hence the
name.

**Its accuracy depends on the pivots.** At exact rank :math:`r`, any pivots
with an invertible :math:`P` reproduce :math:`A`. When :math:`A` is only
close to rank :math:`r`, the error away from the pivots depends on which
rows and columns were chosen. :ref:`tci-prrlu` shows how they are chosen.

Every cut: repeating the cross
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A tensor train needs a factorization at every bond. TT-SVD obtains them by
repeating one split. The cross can be repeated the same way.

**Sites, bonds and pivots.** Site :math:`s = 0, \ldots, d - 1` carries the
index :math:`i_s \in \{0, \ldots, n_s - 1\}`. Bonds keep their meaning from
:ref:`tt-svd`: bond :math:`b` has the :math:`b` sites :math:`0, \ldots,
b - 1` on its left and the rest on its right. The unfolding :math:`F_b` at
bond :math:`b` has configurations of the left sites as rows and
configurations of the right sites as columns. A *configuration* of some
sites is one index for each of them. :math:`F[u, v]` is the value at the
configuration :math:`u` followed by :math:`v`. A cross at bond :math:`b`
chooses :math:`r_b` rows and :math:`r_b` columns of :math:`F_b`:

- the **row pivots** :math:`\mathrm{left}(b)`, configurations of the sites
  left of the bond;
- the **column pivots** :math:`\mathrm{right}(b)`, configurations of the
  sites right of the bond;
- their pivot matrix :math:`P_b = F[\mathrm{left}(b), \mathrm{right}(b)]`.

Every object of a bond carries the number of the bond. At the ends,
:math:`\mathrm{left}(0)` and :math:`\mathrm{right}(d)` hold the empty
configuration.

**Three sites example.** Take the ranks to be exact for now, so that every cross
reproduces its matrix. At bond 1 the rows of :math:`F_1` are the values of
:math:`i_0` and its columns are the pairs :math:`(i_1, i_2)`. The cross is

.. math::

   F_1 = F_1[:, \mathrm{right}(1)] \; P_1^{-1} \; F_1[\mathrm{left}(1), :].

The first factor is :math:`F` with site 0 free and sites 1 and 2 pinned to
the column pivots. The last factor is :math:`r_1` complete rows of
:math:`F`: for each row pivot :math:`u`, all values :math:`F[u, i_1, i_2]`.

At this point TT-SVD carries :math:`\Sigma V^\dagger` forward and reshapes it
so that the next site joins the rows. The cross carries its pivot rows
forward instead. They are values of :math:`F` itself, reshaped the same way
into a matrix with rows :math:`(u, i_1)` and columns :math:`i_2`. Crossing
that matrix at bond 2, with the row pivots :math:`\mathrm{left}(2)` chosen
among its rows, gives

.. math::

   F[\mathrm{left}(1), i_1, i_2]
   = F[\mathrm{left}(1), i_1, \mathrm{right}(2)] \; P_2^{-1} \;
     F[\mathrm{left}(2), i_2].

Together the two crosses give :math:`F = T_0 P_1^{-1} T_1 P_2^{-1} T_2` with

.. math::

   T_0 = F[i_0, \mathrm{right}(1)], \qquad
   T_1 = F[\mathrm{left}(1), i_1, \mathrm{right}(2)], \qquad
   T_2 = F[\mathrm{left}(2), i_2].

**Any number of sites.** Repeated at every bond, the cross gives the **TCI
form**

.. math::

   F \approx T_0 \, P_1^{-1} \, T_1 \, P_2^{-1} \cdots P_{d-1}^{-1} \,
   T_{d-1},
   \qquad
   T_s = F[\mathrm{left}(s), i_s, \mathrm{right}(s + 1)].

Each **site tensor** :math:`T_s` is a slice of :math:`F` with one site free:
site :math:`s` runs over its values while the sites on either side are
pinned to the pivots of the two bonds around it. It has the shape of a core,
:math:`r_s \times n_s \times r_{s+1}`. The products contract over the pivots
of each bond.

.. figure:: /images/tci-form.png
   :width: 100%
   :align: center

   The TCI form in the diagram notation of :ref:`tensor-networks`. Each site
   tensor :math:`T_s` has one open leg :math:`i_s`. Each inverse pivot matrix
   :math:`P_b^{-1}` sits on bond :math:`b`, between the column pivots
   :math:`\mathrm{right}(b)` and the row pivots :math:`\mathrm{left}(b)`.

The form is exact when :math:`r_b` is the rank of :math:`F_b` and
:math:`P_b` is invertible at every bond. This is the statement of
:ref:`tt-svd` that the TT-ranks are the ranks of the unfoldings, with each
SVD replaced by a cross. With fewer pivots than the ranks the form
approximates :math:`F`. As for a matrix, its error then depends on the
pivots. Either way the form is made of values of :math:`F`: the site tensors
of the train in the introduction hold 7 680 of them. The run that chose
their pivots evaluated 163 912, which is the subject of
:ref:`tci-algorithm`.

From TCI to tensor train form
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Grouping each pivot matrix with the site tensor on its left,

.. math::

   G_s = T_s \, P_{s+1}^{-1} \quad (s < d - 1), \qquad G_{d-1} = T_{d-1},

turns the TCI form into an ordinary tensor train :math:`G_0 G_1 \cdots
G_{d-1}` with bond dimensions :math:`r_b`.
:meth:`TensorCI.to_tensor_train <ttnumpy.TensorCI.to_tensor_train>` computes
these cores. Everything in :doc:`tt_numerics` applies to the result.

Two features set this train apart from one made by TT-SVD. The first is the
meaning of a bond index. In TT-SVD the index on bond :math:`b` labels a
singular vector. Here it labels a row pivot: one configuration
:math:`u \in \mathrm{left}(b)` of the sites left of the bond. The bond
dimension is the number of pivots. The pivots name the partial
configurations the train is built around.

The second is the structure of the cores. Suppose every row pivot of bond
:math:`s + 1` is a row of :math:`T_s`: a row pivot of bond :math:`s`
followed by a value of site :math:`s`. The repeated cross above chose its
pivots this way. Those rows of :math:`T_s` then make up :math:`P_{s+1}`, so
:math:`G_s` holds the identity matrix on them:

.. math::

   G_s[w, w'] = \delta_{w w'}, \qquad w, w' \in \mathrm{left}(s + 1),

where a row index :math:`w = (u, i_s)` combines a pivot of bond :math:`s`
with a value of site :math:`s`. This is the counterpart of the canonical
forms of :ref:`tt-accuracy`, where the cores left of the centre satisfy
:math:`\sum_{i} G[i]^{H} G[i] = I`. Orthonormal cores turn a local singular
value into a global error. Cores with identity blocks make the train
reproduce the values it was built from, as the next section shows.

Nesting and the interpolation property
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The condition behind the identity blocks, together with its mirror image
for the column pivots, is called nesting. The pivots of neighbouring bonds
are **nested** when one extends the other by a site:

- **left-nested** at bond :math:`b`: every row pivot of bond :math:`b` is a
  row pivot of bond :math:`b - 1` followed by one more index,
  :math:`\mathrm{left}(b) \subseteq \mathrm{left}(b - 1) \times \{0, \ldots,
  n_{b-1} - 1\}`;
- **right-nested** at bond :math:`b`: every column pivot of bond :math:`b` is
  one more index followed by a column pivot of bond :math:`b + 1`,
  :math:`\mathrm{right}(b) \subseteq \{0, \ldots, n_b - 1\} \times
  \mathrm{right}(b + 1)`.

They are **fully nested** when both hold at every bond, as in this example
on four binary sites:

=====  ========================  =========================
Bond   :math:`\mathrm{left}(b)`  :math:`\mathrm{right}(b)`
=====  ========================  =========================
1      \(1)                      (0, 0, 1)
2      (1, 0), (1, 1)            (0, 1), (1, 1)
3      (1, 0, 1), (1, 1, 0)      (1), (0)
=====  ========================  =========================

Read down the table, the row pivots form a tree: each grows out of a row
pivot one line up. Read up the table, the column pivots grow the same way
from the right. Adding :math:`(0, 0)` to :math:`\mathrm{left}(2)` would
break the nesting at bond 2, since :math:`(0)` is not a row pivot of bond 1.

The repeated cross above yields left-nested pivots without asking for them:
the candidates at bond 2 were the pairs :math:`(u, i_1)` with
:math:`u \in \mathrm{left}(1)`. TT-SVD run from the left likewise yields
left-orthonormal cores. Neither guarantees anything about the other side.

Nesting is what makes the train an interpolation. **With fully nested
pivots, the train equals the tensor on every entry of every site tensor and
pivot matrix**, that is, on every value it was built from ([NF2025]_,
Sec. 4.2). To see why, read the train at an entry :math:`F[u, i, v]` of
:math:`T_s`, with :math:`u \in \mathrm{left}(s)` and
:math:`v \in \mathrm{right}(s + 1)`:

- From the left along :math:`u`, the identity blocks pass a unit vector from
  core to core, so the first :math:`s` cores select :math:`u`.
- From the right along :math:`v`, the cores return the column
  :math:`F[\mathrm{left}(s + 1), v]` of :math:`P_{s+1}`, which
  :math:`P_{s+1}^{-1}` inside :math:`G_s` turns into a unit vector selecting
  :math:`v`.
- What remains is :math:`T_s[u, i, v] = F[u, i, v]`.

A broken nesting stops the unit vectors at that bond. The result is still a
tensor train, but no longer guaranteed to match :math:`F` on the values it
was built from.
:meth:`PivotLists.is_fully_nested
<ttnumpy.interpolation.tci_form.PivotLists.is_fully_nested>` checks the
condition. The updates of :ref:`tci-refine` restore it.

.. _tci-notation:

Notation of the literature
~~~~~~~~~~~~~~~~~~~~~~~~~~

[NF2025]_ numbers sites from 1 and gives the two pivot lists of one bond
different subscripts. This guide and the code address both by the bond: bond
:math:`b`, its rank :math:`r_b`, its unfolding :math:`F_b`, its pivot matrix
:math:`P_b` and its pivot lists :math:`\mathrm{left}(b)` and
:math:`\mathrm{right}(b)`. The slices keep their letters. The pivot matrix
:math:`P_b` and the 2-site slice :math:`\Pi_b` of :ref:`tci-two-site` keep
their subscripts as well. The site tensor :math:`T_s` is the paper's
:math:`T_{s+1}`. The rest, in terms of :math:`s` and :math:`b`:

.. list-table::
   :header-rows: 1

   * - Object
     - This guide
     - Paper
     - Code
   * - Sites
     - :math:`s = 0, \ldots, d - 1`
     - :math:`\ell = s + 1`, up to :math:`L = d`
     - ``site``; ``len(dims)`` sites
   * - Index at site :math:`s`
     - :math:`i_s`, from 0
     - :math:`\sigma_{s+1}`, from 1
     - an entry of a configuration
   * - Values at site :math:`s`
     - :math:`n_s`
     - :math:`d_{s+1}`
     - ``dims[s]``
   * - Bond between sites :math:`b - 1` and :math:`b`
     - :math:`b`
     - :math:`\ell = b`
     - ``bond``
   * - Rank at bond :math:`b`
     - :math:`r_b`
     - :math:`\chi_b`
     - ``pivots.ranks[b]``
   * - Row pivots at bond :math:`b`
     - :math:`\mathrm{left}(b)`
     - :math:`\mathcal{I}_b`
     - ``pivots.left(b)``
   * - Column pivots at bond :math:`b`
     - :math:`\mathrm{right}(b)`
     - :math:`\mathcal{J}_{b+1}`
     - ``pivots.right(b)``
   * - Core
     - :math:`G_s = T_s P_{s+1}^{-1}`
     - :math:`M_{s+1} = T_{s+1} P_{s+1}^{-1}`
     - ``to_tensor_train().cores[s]``
   * - Nesting at bond :math:`b`
     - left- and right-nested
     - :math:`\mathcal{I}_{b-1} < \mathcal{I}_b`,
       :math:`\mathcal{J}_{b+1} > \mathcal{J}_{b+2}`
     - ``pivots.is_left_nested_at(b)``, ``pivots.is_right_nested_at(b)``

.. _tci-algorithm:

The TCI algorithm
-----------------

The repeated cross chose its pivots from whole unfoldings: at bond 1, from
all :math:`n_1 \cdots n_{d-1}` columns of :math:`F_1`. That is as far out of
reach as TT-SVD. The form it produced needs much less, only the slices
:math:`T_s` and :math:`P_b`. TCI keeps the form and changes how the pivots
are chosen. It looks at small slices only and chooses pivots in each with a
greedy factorization. Sweeps along the train improve them, as the solvers of
:doc:`linear_systems` improve their cores.

.. _tci-prrlu:

Pivots of one matrix
~~~~~~~~~~~~~~~~~~~~

The error of a cross is the matrix it leaves behind,

.. math::

   A - A[:, J] \, A[I, J]^{-1} \, A[I, :],

the **Schur complement** of the pivot matrix. It vanishes on the pivot rows
and columns. Elsewhere it is small when :math:`|\det A[I, J]|` is large:
the maximum-volume principle ([NF2025]_, Sec. 3.2). Finding the pivots of
largest volume is a combinatorial problem. Gaussian elimination with
complete pivoting chooses them greedily instead:

1. Take the entry of largest modulus as the next pivot.
2. Subtract the rank-one cross through it,
   :math:`A \leftarrow A - A[:, j] \, A[i, j]^{-1} A[i, :]`, which zeroes
   its row and its column.
3. Repeat on what is left until its largest modulus falls to a tolerance.

What is left after :math:`k` steps is the Schur complement of the :math:`k`
pivots taken. Its largest modulus is the **pivot error**. Stopping at a
tolerance fixes both the pivots and their number, the rank, in one pass.
This is the partially rank-revealing LU decomposition (prrLU) of [NF2025]_:
the elimination is an LU factorization stopped at rank :math:`k`. It never
forms :math:`P^{-1}`.

.. _tci-two-site:

Candidates from the neighbours: the 2-site update
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

prrLU can choose pivots only among the rows and columns of the matrix it
factorizes. Factorizing the whole unfolding :math:`F_b` would mean evaluating
the whole tensor again. The 2-site update builds a small matrix from the
pivots of the two neighbouring bonds instead. Its rows are the row pivots of
bond :math:`b - 1`, each extended by every value of site :math:`b - 1`. Its
columns are the column pivots of bond :math:`b + 1`, each preceded by every
value of site :math:`b`:

.. math::

   \text{rows } \mathrm{left}(b - 1) \times \{0, \ldots, n_{b-1} - 1\},
   \qquad
   \text{columns } \{0, \ldots, n_b - 1\} \times \mathrm{right}(b + 1).

The values of :math:`F` there form the **2-site slice**

.. math::

   \Pi_b = F[\mathrm{left}(b - 1), i_{b-1}, i_b, \mathrm{right}(b + 1)],

:math:`F` with the two free sites at bond :math:`b` and all others pinned to
the pivots around them. As a matrix it has :math:`r_{b-1} n_{b-1}` rows and
:math:`n_b r_{b+1}` columns. prrLU of :math:`\Pi_b` at the tolerance selects
the new :math:`\mathrm{left}(b)` and :math:`\mathrm{right}(b)`.

Three properties follow from this choice of candidates:

- **The new pivots are nested** with the neighbours they extend: the rows
  with bond :math:`b - 1`, the columns with bond :math:`b + 1`.
- **The rank adapts.** prrLU keeps as many pivots as the slice needs at the
  tolerance, more or fewer than before, up to
  :math:`\min(r_{b-1} n_{b-1}, \, n_b r_{b+1})`.
- **The pivot error is the error of the train on the slice**, as long as
  bonds :math:`b - 1` and :math:`b + 1` are nested with their own
  neighbours. The train restricted to the entries of :math:`\Pi_b` is then
  the cross through the new pivots. The Schur complement that prrLU leaves
  is therefore :math:`F - \tilde F` there ([NF2025]_, Eq. 40).

The pattern is that of MALS in :doc:`linear_systems`: two neighbouring
sites, a local object spanning both and a split in which the rank adapts.
MALS solves for its supercore and splits it by an SVD; the 2-site update
samples its slice and splits it by prrLU. An update requests
:math:`r_{b-1} n_{b-1} n_b r_{b+1}` values.

Sweeps and stopping
~~~~~~~~~~~~~~~~~~~

:func:`~ttnumpy.tci` runs these updates along the train.

**Start.** The first initial pivot, a complete configuration
:math:`c = (c_0, \ldots, c_{d-1})`, is split at every bond:
:math:`\mathrm{left}(b) = \{(c_0, \ldots, c_{b-1})\}` and
:math:`\mathrm{right}(b) = \{(c_b, \ldots, c_{d-1})\}`. Every rank is 1 and
the pivots are fully nested. The only condition on the initial pivot is
:math:`F[c] \neq 0`.

**Half-sweeps.** A forward half-sweep updates bonds :math:`1, \ldots, d - 1`
in turn, a backward half-sweep bonds :math:`d - 1, \ldots, 1`. Each update
builds on the pivots its predecessor has just installed. A forward
half-sweep therefore leaves every bond left-nested and a backward one every
bond right-nested. Each update finds its neighbours nested, as the pivot
error requires. The ranks grow from 1 as the slices reveal structure.

**Threshold.** The largest :math:`|F|` is not known in advance, so the
tolerance of each half-sweep is ``error_threshold`` times the largest
modulus sampled before it: the threshold is relative.

**Stopping.** The run stops after a half-sweep in which every pivot error is
within the threshold, no rank changes and no larger modulus is sampled. It
then reports ``converged``. A run that reaches ``max_half_sweeps`` first
stops unconverged and logs a warning. ``max_rank`` caps the number of pivots
at every bond. A run held back by it does not converge.

With ``return_info=True``, :func:`~ttnumpy.tci` also returns a
:class:`~ttnumpy.TCIInfo` with the largest pivot error and the largest rank
of every half-sweep. For the run of the introduction:

.. code-block:: python

   tensor_ci, info = tt.tci(
       f, [16] * d, initial_pivots=[[0] * d], error_threshold=1e-8, return_info=True
   )

   print(info.converged, info.num_half_sweeps_run)   # True 4
   print(info.max_ranks)                              # (7, 8, 8, 8)

What the pivot error measures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each pivot error is the true error of the train, on the slice it was
computed on. The slices of a run cover a vanishing part of the tensor.
Nothing outside them is checked: a feature that no slice reaches cannot
raise a pivot error, so the run converges without it.

A corner shows this. On ten nodes in :math:`[-1, 1]` per variable, let
:math:`F` be 2 where all six variables are positive and 1 elsewhere:

.. code-block:: python

   grid = np.linspace(-1.0, 1.0, 10)

   def corner(indices):
       return 1.0 + float(all(grid[i] > 0.0 for i in indices))

   tensor_ci, info = tt.tci(corner, [10] * 6, initial_pivots=[[0] * 6], return_info=True)

   print(info.converged, info.pivot_errors)    # True (0.0,)
   print(tensor_ci.pivots.ranks)               # [1, 1, 1, 1, 1, 1, 1]

The initial pivot puts every variable at :math:`-1`. A 2-site slice frees
two variables and pins the other four to pivots, all of them at
:math:`-1`, so every value sampled is 1. The run converges after one
half-sweep with a pivot error of 0. The train is the constant 1, wrong by 1
on the whole corner.

.. figure:: /images/tci-corner.png
   :width: 70%
   :align: center

   The corner for three variables. Started at :math:`(-1, -1, -1)`, every
   slice of bond 1 lies in the plane :math:`x_2 = -1` and every slice of
   bond 2 in the plane :math:`x_0 = -1`. Neither plane enters the corner,
   where :math:`F = 2`.

A pivot error is a lower bound: some values of :math:`F` were approximated
at least that badly when it was measured. It is not an upper bound on
anything outside the slices, in the way a training error says nothing about
unseen data. Where the error matters, check the train on
configurations of interest; :ref:`tci-global` puts a missed region on a
slice.

TT-SVD and TCI side by side
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :stub-columns: 1

   * -
     - TT-SVD
     - TCI
   * - Reads
     - the whole array
     - slices of it, through a function of indices
   * - Factorization at bond :math:`b`
     - SVD of the unfolding :math:`F_b`
     - cross of :math:`F_b` through :math:`\mathrm{left}(b)` and
       :math:`\mathrm{right}(b)`
   * - Basis on bond :math:`b`
     - singular vectors
     - rows and columns of the tensor
   * - Carried to the next bond
     - :math:`\Sigma V^\dagger`
     - the pivot rows of :math:`F`
   * - Cores
     - orthonormal
     - an identity block on the pivot rows
   * - Rank set by
     - singular values against the error budget
     - pivot errors against ``error_threshold``
   * - Error statement
     - a bound on the Frobenius norm of the error
     - the true error on the slices sampled, nothing elsewhere

.. _tci-practice:

Interpolating functions
-----------------------

Indices, not coordinates
~~~~~~~~~~~~~~~~~~~~~~~~

TCI samples a tensor. The function it calls receives a tuple of integers
with one index per site and returns the value there. Nothing in the algorithm
uses what an index means, only how many values each site takes. A function
of real variables reaches TCI through a grid of the caller's choosing:

.. code-block:: python

   import numpy as np
   import ttnumpy as tt

   gauss_nodes = np.polynomial.legendre.leggauss(32)[0]    # in (-1, 1)

   def gaussian(indices):
       x = gauss_nodes[list(indices)]
       return np.exp(-x @ x)

   tensor_ci = tt.tci(gaussian, [32] * 5, initial_pivots=[[16] * 5])
   print(tensor_ci.pivots.ranks)    # [1, 1, 1, 1, 1, 1]

Any nodes work, uniform or not, on any interval. Each variable can have its
own nodes and its own number of them, given in ``dims``. Problems that
are discrete from the start, such as spin configurations or an existing
tensor, need no grid at all.

The function is wrapped in a :class:`~ttnumpy.BlackBox`, which evaluates
each configuration once and serves repeats from a cache. Neighbouring slices
share pivots, so a run requests the same configurations many times. The
counters of the box report this:

=================  ==========================================================
``n_evals``        calls to the function, one per distinct configuration
``n_requests``     configurations requested, evaluated or cached
``n_hits``         requests served from the cache
``max_modulus``    largest :math:`|F|` evaluated, the scale of relative thresholds
=================  ==========================================================

Passing a ``BlackBox`` to :func:`~ttnumpy.tci` in place of a function lets
several runs share one cache. The form a run returns keeps its box with
every value sampled, which is the dominant memory of a long run. Values
must be finite: a function that returns ``inf`` or ``nan`` stops the run
with a ``ValueError``.

Initial pivots
~~~~~~~~~~~~~~

The first initial pivot decides what the first slices see, so it belongs
where :math:`|F|` is large or where the structure is. Since the threshold is
relative to the values sampled, :func:`~ttnumpy.tci` needs :math:`F` to be
nonzero at one initial pivot at least. It raises a ``ValueError``
otherwise.

Further initial pivots seed features or symmetry sectors that one start
would miss: :func:`~ttnumpy.tci` starts from the first and adds the others as
global pivots before the first half-sweep. Started from both opposite
corners, the corner example converges to the exact train:

.. code-block:: python

   tensor_ci = tt.tci(corner, [10] * 6, initial_pivots=[[0] * 6, [9] * 6])
   print(tensor_ci.pivots.ranks)    # [1, 2, 2, 2, 2, 2, 1]

.. _tci-global:

Global pivots
~~~~~~~~~~~~~

A **global pivot** is a complete configuration offered to every bond at
once: at bond :math:`b` its first :math:`b` indices join the row candidates
and the rest join the column candidates. :func:`~ttnumpy.add_global_pivots`
does this to an existing form. At every bond it enlarges the pivot matrix by
the rows and columns of the new halves and keeps what prrLU of the enlarged
matrix selects. It then sweeps as :func:`~ttnumpy.tci` does until the form
converges again. It works in place:

.. code-block:: python

   tensor_ci = tt.tci(corner, [10] * 6, initial_pivots=[[0] * 6])
   tt.add_global_pivots(tensor_ci, [[9] * 6])

   print(tensor_ci.pivots.ranks)    # [1, 2, 2, 2, 2, 2, 1]
   print(tensor_ci.box.n_evals)     # 1804

The train is now exact. Global pivots belong where the slices cannot be
expected to reach: narrow peaks, isolated nonzero entries of a sparse
tensor, one configuration per symmetry sector and configurations where a
check of the train found a large error.

A configuration is kept at a bond only where it adds rank to the pivot
matrix. The sweeps then learn what the new slices show.

For a function of continuous variables, global pivots carry knowledge about
the function into the interpolation ([NF2025]_, Sec. 4.3.5). The location of
a narrow peak, of a singularity or of the maximum of :math:`|F|` makes a
natural pivot. So do the pivots of a nearby function in a parameter sweep,
where :math:`F` changes slowly with the parameter. A narrow bump shows the
effect. On 32 nodes in :math:`[0, 1]` per variable, add a Gaussian of width
0.03 centred at :math:`x_s = 0.7` to the function of the introduction:

.. code-block:: python

   points = np.linspace(0.0, 1.0, 32)

   def bump(indices):
       x = points[list(indices)]
       return 1.0 / (1.0 + x @ x) + np.exp(-np.sum((x - 0.7) ** 2) / 0.03**2)

   tensor_ci = tt.tci(bump, [32] * 4, initial_pivots=[[0] * 4], error_threshold=1e-8)
   print(tensor_ci.pivots.ranks)    # [1, 6, 7, 6, 1]

   # node 22 lies at 0.71, the nearest to the centre
   tt.add_global_pivots(tensor_ci, [[22] * 4], error_threshold=1e-8)
   print(tensor_ci.pivots.ranks)    # [1, 7, 8, 7, 1]

Started at :math:`x = 0`, the run converges without seeing the bump. Its
largest error is 0.66. The global pivot at the node nearest the centre
raises every rank by one and brings the largest error to
:math:`1.7 \cdot 10^{-8}`.

When the location is not known, a search can supply it. Compare the train
with the function at many points and offer the worst as a global pivot. For
the bump, the worst of 2 000 random grid points lay next to the bump in each
of five runs with different points. Adding it brought the largest error from
0.66 to between :math:`1.5 \cdot 10^{-8}` and :math:`1.5 \cdot 10^{-5}`.
The search stays separate from the interpolation, so any optimizer of
:math:`|F - \tilde F|` can replace the random points.

.. _tci-refine:

Refining an interpolation
-------------------------

Three kinds of update
~~~~~~~~~~~~~~~~~~~~~

The 2-site update is one of three. All three replace the pivots at one bond
by what prrLU selects from a matrix of values of :math:`F`. They differ in
whether the rows and the columns are the bond's own pivots or a neighbour's
pivots extended by a site:

.. list-table::
   :header-rows: 1
   :stub-columns: 1

   * - Candidates at bond :math:`b`
     - Columns :math:`\mathrm{right}(b)`
     - Columns :math:`\{i_b\} \times \mathrm{right}(b + 1)`
   * - Rows :math:`\mathrm{left}(b)`
     - **0-site**: :math:`P_b`, :math:`r_b^2` values
     - **1-site backward**: :math:`T_b`, :math:`r_b n_b r_{b+1}` values
   * - Rows :math:`\mathrm{left}(b - 1) \times \{i_{b-1}\}`
     - **1-site forward**: :math:`T_{b-1}`, :math:`r_{b-1} n_{b-1} r_b`
       values
     - **2-site**: :math:`\Pi_b`, :math:`r_{b-1} n_{b-1} n_b r_{b+1}` values

An extended side lets new pivots in and nests them with that neighbour. A
side of the bond's own can only keep or drop pivots, so only the 2-site
update can raise a rank. The other two lower ranks. The 1-site update also
restores nesting:

- :func:`~ttnumpy.interpolation.sweeps.zero_site_sweep` runs a 0-site update
  at every bond. It factorizes only the pivot matrices, whose values a run
  of :func:`~ttnumpy.tci` has already sampled. Pivots that the threshold does
  not need are dropped. The nesting can break.
- :func:`~ttnumpy.interpolation.sweeps.one_site_sweep` runs 1-site updates
  forward over every site and then backward, repeated until a backward pass
  keeps every rank, which leaves the pivots fully nested. It never raises a
  rank.

The single updates are
:func:`~ttnumpy.interpolation.sweeps.two_site_update`,
:func:`~ttnumpy.interpolation.sweeps.one_site_update` and
:func:`~ttnumpy.interpolation.sweeps.zero_site_update`, with an absolute
tolerance.

Recompression
~~~~~~~~~~~~~

TT-rounding lowers the ranks of a train to a looser accuracy without
expanding it. :func:`~ttnumpy.recompress` does the same for a TCI form. It
runs a 0-site sweep and then a 1-site sweep, both at the new threshold, so
the result is again an interpolation with fully nested pivots. For the
function of the introduction:

.. code-block:: python

   tensor_ci = tt.tci(f, [16] * d, initial_pivots=[[0] * d], error_threshold=1e-10)
   print(tensor_ci.pivots.ranks)                # [1, 8, 9, 10, 10, 10, 10, 10, 9, 8, 1]
   print(tensor_ci.pivots.is_fully_nested())    # False

   tt.recompress(tensor_ci, error_threshold=1e-6)
   print(tensor_ci.pivots.ranks)                # [1, 5, 6, 7, 7, 7, 7, 7, 6, 5, 1]
   print(tensor_ci.pivots.is_fully_nested())    # True

The result matches a run of :func:`~ttnumpy.tci` at :math:`10^{-6}`: the
same ranks and the same largest error on 10 000 random entries,
:math:`2.1 \cdot 10^{-6}`. Recompressing called the function 2 871 times,
the run from scratch 105 072 times.

A converged run is not always fully nested, as the run at :math:`10^{-10}`
shows. Recompression has two uses: deriving coarser forms from one tight
run and restoring full nesting. It belongs at the end of a pipeline, since
the form keeps no threshold: a later 2-site sweep at a tighter one grows the
ranks back.

From a tensor train
~~~~~~~~~~~~~~~~~~~

Any tensor train can be brought into TCI form without sampling it.
:meth:`TensorCI.from_tensor_train <ttnumpy.TensorCI.from_tensor_train>`
chooses pivots from the cores alone, by exact factorizations in a forward, a
backward and a second forward half-sweep ([NF2025]_, Sec. 4.5.1). It samples
the train once, at the largest entry of the slices that the backward
half-sweep factorizes, which fixes the scale of relative thresholds. The
pivots are fully nested. Each bond holds as many as the numerical rank of
the tensor there. Like rounding at error 0, it finds the true ranks rather
than those of the representation:

.. code-block:: python

   x = tt.rand([4] * 5, ranks=2, seed=7)
   inflated = x + x + x
   print(inflated.bonds)                    # [1, 6, 6, 6, 6, 1]

   tensor_ci = tt.TensorCI.from_tensor_train(inflated)
   print(tensor_ci.pivots.ranks)            # [1, 2, 2, 2, 2, 1]
   print(tensor_ci.box.n_evals)             # 1

The form samples the train itself, so the values it requests later are
computed from the cores. This brings the TCI operations to trains from any
source, TT-SVD, arithmetic or a solver: :func:`~ttnumpy.recompress`, for
instance, lowers their ranks with a threshold on entries rather than on the
Frobenius norm.

.. _tci-cost:

Accuracy and cost
-----------------

What the threshold controls
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``error_threshold`` bounds the pivot errors relative to the largest value
sampled. For the function of the introduction, the largest error on 10 000
random entries follows it within a factor of about 2:

=========  ========  ===========  =============
Threshold  Max rank  Evaluations  Largest error
=========  ========  ===========  =============
1e-02      2         7 680        1.4e-02
1e-04      4         40 768       2.1e-04
1e-06      7         105 072      2.1e-06
1e-08      8         163 912      1.4e-08
1e-10      10        234 880      1.4e-10
1e-12      12        339 104      6.2e-13
=========  ========  ===========  =============

With five variables instead of ten, the full array has 1 048 576 entries
and TT-SVD can decompose it at the same nominal accuracy:

=========  ============  ==============  =========  =============  ============
Threshold  TCI ranks     Values read     TCI error  TT-SVD ranks   TT-SVD error
=========  ============  ==============  =========  =============  ============
1e-04      4, 4, 4, 4    12 928 (1.2 %)  3.8e-05    3, 4, 4, 3     3.5e-04
1e-06      5, 6, 6, 5    22 416 (2.1 %)  8.1e-07    5, 5, 5, 5     6.1e-06
1e-08      7, 7, 7, 7    35 296 (3.4 %)  6.2e-09    6, 7, 7, 6     2.6e-08
1e-10      8, 9, 9, 8    52 480 (5.0 %)  3.2e-11    7, 8, 8, 7     1.1e-09
=========  ============  ==============  =========  =============  ============

TCI reads a few percent of the entries. Its threshold acts on the largest
entry error and that of TT-SVD on the Frobenius norm, so at the same nominal
accuracy TCI keeps up to one rank more and reaches a smaller largest error.

Cost
~~~~

The costs are given for :math:`d` sites of :math:`n` values and largest rank
:math:`r`. A 2-site or 1-site sweep is a forward half-sweep followed by a
backward one. A 0-site sweep updates every bond once. Requests for values
sampled before are served from the cache, so the function is called once
per distinct configuration, :math:`e` times in a run. Building and hashing the
configuration of a request costs :math:`O(d)`, which is the :math:`d` in the
time column.

.. list-table::
   :header-rows: 1

   * - Operation
     - Values requested
     - Time
     - Memory
   * - 2-site sweep
     - :math:`O(d r^2 n^2)`
     - :math:`O(d r^2 n^2 (r + d))`
     - :math:`O(r^2 n^2)`
   * - 1-site sweep
     - :math:`O(d r^2 n)`
     - :math:`O(d r^2 n (r + d))`
     - :math:`O(r^2 n)`
   * - 0-site sweep
     - :math:`O(d r^2)`
     - :math:`O(d r^2 (r + d))`
     - :math:`O(r^2)`
   * - :func:`~ttnumpy.tci`, :math:`h` half-sweeps
     - :math:`O(h d r^2 n^2)`, of which :math:`e` distinct
     - :math:`O(h d r^2 n^2 (r + d))`
     - :math:`O(e d + d^2 r + r^2 n^2)`

The memory of a run is dominated by the cache, :math:`e` values with
:math:`d` indices each, which the returned form keeps.

The single updates, adding global pivots and the conversions between a TCI
form and a train have costs of their own. They are listed in the source, in
the module docstrings of ``ttnumpy/interpolation/sweeps.py`` and
``ttnumpy/interpolation/tci.py`` and in the methods of
:class:`~ttnumpy.TensorCI`. The first and the last also appear in
:doc:`/api`, under :mod:`ttnumpy.interpolation.sweeps` and
:class:`~ttnumpy.TensorCI`. The docstrings write ``n`` for the number of
sites, ``d`` for the number of values per site and ``chi`` for the largest
rank.

Limitations
~~~~~~~~~~~

- **Only sampled slices are checked.** No bound covers the entries outside
  them, as the corner example shows.
- **Features the slices miss stay missed**: narrow peaks, isolated nonzero
  entries, separate symmetry sectors. The first initial pivot decides what
  the first slices see. Global pivots put the rest on a slice.
- **Structure that is not low-rank stays expensive.** A discontinuity along
  a surface not aligned with the axes, or noise, needs large ranks. TCI then
  samples a growing share of the tensor, up to all of it for random data.
- **Ranks can cycle.** When pivot errors sit just below the threshold, a run
  can keep changing ranks between half-sweeps without evaluating anything
  new. It then stops at ``max_half_sweeps`` unconverged, with every pivot
  error within the threshold.
  ``TCIInfo.pivot_errors`` tells such a run from one that has not reached
  the threshold.
- **Shapes.** :func:`~ttnumpy.tci` needs two sites at least and two values
  at least at every site.

References
----------

.. [GTZ1997] S. A. Goreinov, E. E. Tyrtyshnikov and N. L. Zamarashkin, *A
   theory of pseudoskeleton approximations*, Linear Algebra Appl. 261,
   1–21, 1997. https://doi.org/10.1016/S0024-3795(96)00301-1

.. [OT2010] I. V. Oseledets and E. E. Tyrtyshnikov, *TT-cross approximation
   for multidimensional arrays*, Linear Algebra Appl. 432(1), 70–88, 2010.
   https://doi.org/10.1016/j.laa.2009.07.024

.. [NF2022] Y. Núñez Fernández, M. Jeannin, P. T. Dumitrescu, T. Kloss,
   J. Kaye, O. Parcollet and X. Waintal, *Learning Feynman diagrams with
   tensor trains*, Phys. Rev. X 12, 041018, 2022.
   https://doi.org/10.1103/PhysRevX.12.041018

.. [NF2025] Y. Núñez Fernández, M. K. Ritter, M. Jeannin, J.-W. Li,
   T. Kloss, T. Louvet, S. Terasaki, O. Parcollet, J. von Delft,
   H. Shinaoka and X. Waintal, *Learning tensor networks with tensor cross
   interpolation: new algorithms and libraries*, SciPost Phys. 18, 104,
   2025. https://doi.org/10.21468/SciPostPhys.18.3.104
