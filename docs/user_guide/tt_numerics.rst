Tensor trains in numerics
=========================

The working mechanics of the format. How a tensor train is built from data,
how its accuracy is controlled, what every operation costs, and how a
representation designed for many-variable problems is turned into a tool for
grid-based numerics.

These are the chapters to keep at hand while writing code. The first three
apply to any tensor train, while the fourth, on quantization, is what makes the
format apply to functions and operators discretized on a grid.

.. _tt-svd:

Building a tensor train: TT-SVD
-------------------------------

Given a full array, TT-SVD produces a tensor train that represents it,
with ranks that are optimal when the decomposition is exact and
quasi-optimal when it is truncated. It is the constructive proof that the
format is usable, and it is the algorithm behind
:func:`~ttnumpy.tt_svd`.

Unfoldings
~~~~~~~~~~

Everything rests on one reshaping. The :math:`k`-th **unfolding** of
:math:`A` is the matrix obtained by grouping the first :math:`k` indices
as rows and the rest as columns:

.. math::

   A_k \in \mathbb{C}^{\,(n_1 \cdots n_k) \times (n_{k+1} \cdots n_d)},
   \qquad
   A_k\big[\overline{i_1 \ldots i_k},\; \overline{i_{k+1} \ldots i_d}\big]
   = A[i_1, \ldots, i_d].

There are :math:`d - 1` of them, one per bond of the train. The central
result of [Oseledets2011]_ ties them directly to the ranks:

   If the unfoldings of :math:`A` satisfy :math:`\operatorname{rank}(A_k) =
   r_k`, then there exists a tensor train representing :math:`A` exactly
   with bond dimensions :math:`r_k`.

So the TT-ranks are not a property of the algorithm -- they are a property
of the array. TT-SVD merely finds them.

The algorithm
~~~~~~~~~~~~~

Sweep left to right, splitting off one core at a time. Start with
:math:`C = A` reshaped to :math:`n_1 \times (n_2 \cdots n_d)`, and at each
step:

1. Compute the SVD: :math:`C = U \Sigma V^{\dagger}`.
2. Truncate to rank :math:`r_k`, keeping the largest singular values.
3. Reshape :math:`U` into the core :math:`G_k` of shape
   :math:`(r_{k-1}, n_k, r_k)`.
4. Carry :math:`\Sigma V^{\dagger}` forward as the new :math:`C`, reshaped to
   :math:`(r_k n_{k+1}) \times (n_{k+2} \cdots n_d)`.

After :math:`d - 1` splits the remainder is the last core. Every core to
the left of the current position is left-orthonormal by construction,
which is what makes step 2 safe — the topic of :ref:`tt-accuracy`.

In this library:

.. code-block:: python

   import numpy as np
   import ttnumpy as tt

   full = np.arange(24, dtype=float).reshape(2, 3, 4)

   train = tt.tt_svd(full, error=1e-10)
   print(train.bonds)  # [1, 2, 2, 1]

   # exact round trip
   print(np.allclose(train.get_full_tensor(), full))  # True

The signature is

.. code-block:: python

   tt_svd(tensor, error=0.0, max_rank=None, physical_indexes=None, ttm=False)

``error`` sets the relative accuracy target, ``max_rank`` caps every bond,
and ``ttm=True`` with ``physical_indexes`` given as pairs produces a TTM
instead of a TT.

The accuracy guarantee
~~~~~~~~~~~~~~~~~~~~~~

Truncating each SVD introduces an error and the errors accumulate . If 
:math:`\varepsilon_k` is the error made at bond :math:`k`, the train 
:math:`B` returned by TT-SVD satisfies

.. math::

   \| A - B \|_F \;\le\; \sqrt{\sum_{k=1}^{d-1} \varepsilon_k^2}.

The errors add *in quadrature*, not linearly. To hit a target relative
accuracy :math:`\varepsilon` overall, give each of the :math:`d-1` SVDs
the budget

.. math::

   \delta = \frac{\varepsilon}{\sqrt{d-1}} \, \|A\|_F ,

which is what ``error`` does internally. The result is also
*quasi-optimal*: it is no worse than :math:`\sqrt{d-1}` times the best
possible approximation at those ranks.

In practice the achieved error sits comfortably inside the budget, because
the bound is a worst case over the individual truncations. And when an array
has no low-rank structure to find, the ranks simply saturate: for a random
:math:`6 \times 6 \times 6 \times 6` array the middle bond reaches
:math:`36 = 6^2`, the largest that cut admits, and TT-SVD correctly reports
that nothing can be discarded.

The limitation
~~~~~~~~~~~~~~

TT-SVD needs the full array in memory. That is precisely the array
:ref:`why-low-rank` says cannot be stored, so the algorithm is a
demonstration and a tool for small :math:`d`, not the route to large
problems. Its cost is dominated by the first SVD, at
:math:`O(n^{\,d+1})`.

Large tensor trains are built by other routes:

- **Analytically**, by writing the cores down. Finite-difference operators
  are the standard example: the QTT Laplacian in :ref:`qtt` is
  assembled directly at bond dimension 6, at any grid size.
- **By arithmetic**, combining existing trains with the operations in
  :ref:`tt-arithmetic` and rounding as you go.
- **As the solution of an equation**, which is what the solvers in
  :doc:`linear_systems` do — they never form the full array at any point.
- **By cross interpolation**, which samples a black-box function at
  adaptively chosen entries, the subject of :doc:`cross_interpolation`.

Once a train exists, by whatever route, its ranks can be reduced again
without ever expanding it. That is the subject of the next chapter.

.. _tt-accuracy:

Accuracy, rounding and truncation
---------------------------------

The previous chapter built a train from a full array. This chapter is
about the operation that makes the format practical: **any tensor train
can be re-compressed to quasi-optimal ranks at a prescribed accuracy,
without ever expanding it back to full form.** This is TT-rounding, and it
is what keeps the ranks from growing without bound as trains are combined.

Why orthogonalization comes first
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Truncating a bond means discarding small singular values of a matrix. The
question is *which* matrix, and the naive answer is wrong.

Split the train at bond :math:`k` into the left part, the core, and the
right part. The singular values of the core alone say nothing about the
global error, because the left and right parts can rescale and skew any
direction arbitrarily — a tiny singular value in the core may correspond
to a large contribution in the represented array.

The fix is to make the surrounding cores **orthonormal**. Sweep from the
left, replacing each core by the :math:`Q` of a QR factorization and
pushing :math:`R` into its neighbour. This changes the cores but not the
array they represent, and it leaves every core to the left satisfying

.. math::

   \sum_{i_k} G_k[i_k]^{H} G_k[i_k] = I .

Do the same from the right, and everything except one core — the
**orthogonality centre** — is orthonormal. Now the frames on both sides are
isometries, the core carries the entire norm of the array, and a local
singular value *is* a global error contribution. Truncation becomes safe
and measurable.

Canonical forms
~~~~~~~~~~~~~~~

Where the centre sits is the canonical form:

============  ===============================================
Form          Meaning
============  ===============================================
Left          All cores left-orthonormal; centre at the last
Right         All cores right-orthonormal; centre at the first
Mixed         Centre at a chosen position, orthonormal both sides
============  ===============================================

.. code-block:: python

   import ttnumpy as tt

   train = tt.rand([4, 4, 4, 4], ranks=3, seed=1)

   left = train.orthonormalize(canonical="left")
   right = train.orthonormalize(canonical="right")
   mixed = train.orthonormalize(canonical="mix", center_position=2)

   print(train.orth_center, left.orth_center, mixed.orth_center)

Orthogonalization is a QR sweep: :math:`O(d \, n \, r^3)`, and exact up to
round-off. The solvers in :doc:`linear_systems` move the centre along the
train as they sweep, for exactly this reason.

Rounding
~~~~~~~~

TT-rounding is orthogonalization followed by a truncated SVD sweep back:

1. Sweep one way with QR to reach a canonical form.
2. Sweep the other way with SVD, discarding singular values within budget
   and moving the centre as you go.

The cost is :math:`O(d \, n \, r^3)` — polynomial in the bond dimension,
independent of :math:`n^d`. The error obeys the same quadrature bound as
TT-SVD, with the same per-bond budget
:math:`\delta = \varepsilon / \sqrt{d-1} \cdot \|A\|_F`, and the result is
quasi-optimal within :math:`\sqrt{d-1}`.

The library exposes this three ways:

=============================  ==============================================
Method                         Use
=============================  ==============================================
``truncate(error, max_rank)``  General rounding; ``error`` defaults to 0.0
``compress_svd(error, ...)``   SVD sweep assuming a canonical form is present
``orthonormalize(canonical)``  Canonical form only, no truncation
=============================  ==============================================

Both ``truncate`` and ``compress_svd`` accept ``max_rank`` to cap bonds
regardless of accuracy, ``canonical`` to choose the sweep direction, and
``inplace``.

``error`` bounds the relative Frobenius error of the result. 
``error=0.0`` sets no target: only singular values at the rounding level of
one SVD, :math:`\min(m, n)\,\varepsilon_{\mathrm{mach}}\,\sigma_1` for an
:math:`m \times n` unfolding, are discarded. In a chain of SVDs over many small
modes, as in ``tt_svd`` of quantized data, the rounding of earlier splits can
exceed that level and stay in the train as extra rank. Positive ``error``
such as the default ``1e-12`` discards it.

The cutoff follows the precision of the data: :math:`\varepsilon_{\mathrm{mach}}`
is the machine epsilon of its dtype, near :math:`10^{-7}` for ``float32`` and
``complex64``. ``error=0.0`` therefore drops single-precision rounding noise as
it drops double-precision noise. The default ``error=1e-12`` lies below single
precision and acts as ``0.0`` there, so in a long chain of splits a few extra
ranks can remain. An ``error`` above the precision of the data discards them,
and the margin it needs grows with the chain: for ``sin(10x)`` sampled in
``float32`` the ranks are exact at ``1e-5`` on :math:`2^{16}` points and at
``1e-4`` on :math:`2^{24}`.

Any tensor can be truncated
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The practical consequence is worth demonstrating, because it is what makes
long chains of operations viable. For example, later we will learn about :ref:`tt-arithmetic`
operations on tensor trains. There will be an addition -- operation that makes 
rank of tensor train grow really fast. Let's try to add a train to itself three times without
rounding. The ranks grow every time, but the represented array never
changes — so all that growth is redundant, and rounding removes it exactly:

.. code-block:: python

   import numpy as np
   import ttnumpy as tt

   x = tt.rand([4] * 5, ranks=2, seed=7)

   inflated = x + x + x
   print(inflated.bonds)            # [1, 6, 6, 6, 6, 1]

   rounded = inflated.truncate(error=1e-12)
   print(rounded.bonds)             # [1, 2, 2, 2, 2, 1]

The rounded train differs from the inflated one by a relative
:math:`6 \cdot 10^{-16}` — round-off. Rounding found the true rank of the
array, not the rank its representation happened to carry.

This is the general situation. A tensor train's ranks are a property of
*this particular representation*. The array's true ranks are what rounding
recovers.

Choosing the tolerance
~~~~~~~~~~~~~~~~~~~~~~

``error`` is a *relative* target on the represented array, so:

- Set it at or below the accuracy you actually need. Rounding harder than
  the problem requires buys ranks you will pay for in every later
  operation.
- Set it no lower than the accuracy your data deserves. Rounding a train
  built from :math:`10^{-6}`-accurate measurements to :math:`10^{-14}`
  preserves noise at full rank.
- ``max_rank`` is a memory guard, not an accuracy control. When it binds,
  the accuracy target is silently not met; check the achieved ranks if this
  matters.

One caveat carries over to linear systems: bounding the error of a
*solution* is not the same as bounding its *residual*. For an operator with
condition number :math:`\kappa`, a solution accurate to
:math:`\varepsilon` may leave a residual as large as :math:`\kappa \cdot
\varepsilon`, and discretized differential operators are ill-conditioned by
nature. :doc:`linear_systems` treats this in detail — it is the reason the
solvers there prefer a residual target over a singular-value threshold.

.. _tt-arithmetic:

Arithmetic and complexity
-------------------------

Tensor trains support the usual algebra: addition, subtraction, scalar and
element-wise multiplication, and matrix products. Every operation is exact
and stays in the format — nothing is expanded to full form. What every
operation *does* do is change the ranks, and since cost grows with the cube
of the rank, knowing how is the whole skill of using the format.

Summary
~~~~~~~

For trains of :math:`d` cores, mode size :math:`n`, and bond dimensions
:math:`r_x`, :math:`r_y`, :math:`r_A`:

==========================  =================  ==============================  ====================================
Operation                   Resulting rank     Memory                          Cost to form
==========================  =================  ==============================  ====================================
:math:`\alpha x`            :math:`r_x`        :math:`O(d\,n\,r_x^2)`          :math:`O(n r_x^2)`
:math:`x + y`               :math:`r_x + r_y`  :math:`O(d\,n\,(r_x + r_y)^2)`  none — copy only
:math:`x \odot y`           :math:`r_x r_y`    :math:`O(d\,n\,r_x^2 r_y^2)`    :math:`O(d\,n\,r_x^2 r_y^2)`
:math:`A x`                 :math:`r_A r_x`    :math:`O(d\,n\,r_A^2 r_x^2)`    :math:`O(d\,n^2 r_A^2 r_x^2)`
:math:`\langle x,y\rangle`  scalar             :math:`O(1)`                    :math:`O(d\,n\,r_x r_y (r_x + r_y))`
:math:`\|x\|`               scalar             :math:`O(1)`                    :math:`O(d\,n\,r_x^3)`
==========================  =================  ==============================  ====================================

The pattern in the rank column: **addition adds ranks, everything else
multiplies them.**

For most operations the two columns agree, but the extremes are worth reading
carefully. Addition performs *no floating-point arithmetic at all*: each
core is allocated and the two operands' cores are copied into disjoint
blocks of it, so the only work is the copy, and its size is the memory
column. The matrix product is the opposite — the contraction sums over an
index of size :math:`n` that does not appear in the result, so it costs a
factor of :math:`n` more arithmetic than the train it produces.

The norm is quoted for a train in general position. When the orthogonality
centre is already known — as it is throughout a solver sweep — the norm is
just the norm of that one core, at :math:`O(n r_x^2)`.

Now, let's cover every operation in detail.

Scalar multiplication
~~~~~~~~~~~~~~~~~~~~~

Scaling multiplies one core by :math:`\alpha` and leaves every bond
untouched — the cheapest operation in the format.

.. code-block:: python

   import ttnumpy as tt

   x = tt.rand([4, 4, 4, 4], ranks=3, seed=1)

   print((2.0 * x).bonds)   # [1, 3, 3, 3, 1] — unchanged
   print((-x).bonds)        # [1, 3, 3, 3, 1]

Addition
~~~~~~~~

The sum of two trains is formed by placing their cores block-diagonally:

.. math::

   G_k^{x+y}[i_k] =
   \begin{pmatrix} G_k^{x}[i_k] & 0 \\ 0 & G_k^{y}[i_k] \end{pmatrix},

with the first and last cores stacked as a row and a column block to keep
the boundary bonds at 1. No arithmetic on the entries is needed at all —
only copying — so forming the sum is essentially free. The price is that
**bonds add**: :math:`r_x + r_y`.

.. code-block:: python

   x = tt.rand([4, 4, 4, 4], ranks=3, seed=1)
   y = tt.rand([4, 4, 4, 4], ranks=2, seed=2)

   print((x + y).bonds)                # [1, 5, 5, 5, 1] = 3 + 2
   print((x + y).truncate().bonds)     # [1, 4, 5, 4, 1]

Because ranks add, a sum of :math:`m` trains has rank :math:`m r` before
rounding. Rounding the finished sum once costs :math:`O(d\,n\,(m r)^3)`;
rounding after every term keeps the rank at its rounded value and costs
:math:`m` smaller roundings, which pays off when :math:`m r` is large
compared with the rounded rank. Subtraction is addition with a scaled
operand and behaves identically.

Element-wise product
~~~~~~~~~~~~~~~~~~~~

The Hadamard product multiplies entries index by index. Its cores are
Kronecker products of the input cores,

.. math::

   G_k^{x \odot y}\big[(\alpha_1 \alpha_2),\, i_k,\, (\beta_1 \beta_2)\big]
   = G_k^{x}[\alpha_1, i_k, \beta_1] \; G_k^{y}[\alpha_2, i_k, \beta_2],

so **bonds multiply**.

.. code-block:: python

   print((x * y).bonds)                # [1, 6, 6, 6, 1] = 3 * 2
   print((x * y).truncate().bonds)     # [1, 4, 6, 4, 1]

``multiply`` accepts a scalar or a train, dispatching to scaling or to the
Hadamard product.

Matrix products
~~~~~~~~~~~~~~~

``dot`` applies a TTM to a TT (matrix by vector) or to another TTM (matrix
by matrix), contracting the column indices of the operator against the row
indices of the operand:

.. math::

   G_k^{Ax}\big[(\alpha_1 \alpha_2),\, i_k,\, (\beta_1 \beta_2)\big]
   = \sum_{j_k} A_k[\alpha_1, i_k, j_k, \beta_1] \;
     G_k^{x}[\alpha_2, j_k, \beta_2].

Bonds multiply again, :math:`r_A r_x`, and the extra summed index makes
this the most expensive of the elementwise-structured operations at
:math:`O(d\,n^2 r_A^2 r_x^2)`.

.. code-block:: python

   A = tt.rand([(4, 4)] * 4, ranks=3, seed=3)   # TTM
   x = tt.rand([4, 4, 4, 4], ranks=3, seed=1)   # TT

   print((A @ x).bonds)                # [1, 9, 9, 9, 1] = 3 * 3
   print((A @ x).truncate().bonds)     # [1, 4, 9, 4, 1]

Operators and methods
~~~~~~~~~~~~~~~~~~~~~

The operators are shorthand for the methods. No arithmetic operation rounds
by default: the result holds the exact sum or product, with the ranks below.
``truncate()`` on the result compresses it, and ``truncate=True`` on a
method does the same in one call.

==============  ==========================  =============================
Operator        Method                      Ranks of the result
==============  ==========================  =============================
``x + y``       ``x.add(y)``                :math:`r_x + r_y`
``x - y``       ``x.add(-y)``               :math:`r_x + r_y`
``a * x``       ``x.multiply(a)``           :math:`r_x`
``x * y``       ``x.multiply(y)``           :math:`r_x r_y`
``A @ x``       ``A.dot(x)``                :math:`r_A r_x`
==============  ==========================  =============================

When the operands nearly cancel, as in ``x - y`` with ``x`` close to ``y``,
their rounding error is large compared with the result, and ``truncate()``
keeps it as rank: ``(x - x).truncate()`` is zero up to rounding but keeps
ranks of :math:`2 r_x` wherever the mode sizes allow.

Norms and inner products return plain numbers: ``tt_norm()`` gives the
Frobenius norm, computed in TT arithmetic without expanding anything. And this is really matter.

Consider evaluating :math:`r = b - A x` — the residual, computed once per
sweep by every solver in :doc:`linear_systems`. With
:math:`r_A = r_x = r_b = 10`:

- :math:`A x` has rank 100.
- :math:`b - Ax` has rank 110.
- Rounding that costs :math:`O(d \, n \cdot 110^3)`, a factor of
  :math:`10^3` above the rank-10 operands.

Iterate without rounding and the ranks compound: ten unrounded matrix
products take rank 10 to :math:`10^{11}`. Round after each with
``truncate(error=...)``, and they stay wherever the accuracy target puts them. This is the single most important
habit in the format, and it is why the solvers spend as much effort on
truncation strategy as on the local solves — see the treatment of
residual-based truncation in :doc:`linear_systems`.

.. _qtt:

Quantized tensor trains (QTT)
-----------------------------

Everything so far assumed a tensor with many indices. But the arrays of
classical numerics usually have *few* indices and *huge* dimensions: a
function sampled on a grid of :math:`2^{30}` points is a vector, order 1.
Quantization manufactures the missing indices, and it is what turns the
tensor train from a many-body tool into a numerical one.

The idea
~~~~~~~~

A vector of length :math:`2^{L}` can be reshaped into an order-:math:`L`
tensor with all mode sizes 2, by reading the index in binary:

.. math::

   \sigma = \sum_{l=1}^{L} i_l \, 2^{\,L-l}, \qquad i_l \in \{0, 1\},
   \qquad
   v[\sigma] \;\longrightarrow\; V[i_1, i_2, \ldots, i_L].

Then apply a tensor train to :math:`V`. The result is a **quantized tensor
train**: :math:`L = \log_2 n` cores of mode size 2.

The new indices are not arbitrary. :math:`i_1` selects which half of the
grid, :math:`i_2` which quarter, and so on — the QTT cores are ordered
from coarse scale to fine, and the bond between core :math:`l` and
:math:`l+1` measures how much the coarse structure of the function needs to
know about its fine structure. For a function with a genuine separation of
scales, that is very little.

The payoff is the change in how storage scales:

.. math::

   O(n) \;\longrightarrow\; O(r^2 \log n).

Logarithmic, not linear. Measured on :math:`e^{3x}`, which has QTT rank 1:

=============  ==================  ==============  ==============
:math:`L`      Grid points         Dense entries   QTT entries
=============  ==================  ==============  ==============
10             1 024               1 024           20
20             1 048 576           1 048 576       40
30             1 073 741 824       1 073 741 824   60
=============  ==================  ==============  ==============

A billion-point grid in sixty numbers. Quantization was introduced in
[Oseledets2010]_ and [Khoromskij2011]_.

Which functions compress
~~~~~~~~~~~~~~~~~~~~~~~~

The ranks are not a matter of hope; for the standard function classes they
are known exactly and they are tiny. Sampled on :math:`2^{12} = 4096`
points at :math:`\varepsilon = 10^{-10}`:

=============================  ============
Function                       Max QTT rank
=============================  ============
:math:`e^{3x}`                 1
:math:`x`                      2
:math:`x^2`                    3
:math:`x^3`                    4
:math:`\sin(2\pi x)`           2
:math:`\sin(2\pi x) + e^{3x}`  3
White noise                    64
=============================  ============

The pattern is the classical one: exponentials have rank 1, a polynomial
of degree :math:`p` has rank :math:`p+1`, and a trigonometric function has
rank 2 — each independent of the grid size. The sum of the sine and the
exponential has rank 3, exactly as :ref:`tt-arithmetic` predicts for a sum
:math:`(2 + 1)`.

The last row is the important one. White noise reaches 64, which is
:math:`2^{L/2}` — the maximum any bond of a 12-core binary train can have.
**QTT compresses smoothness, not size.** Applied to unstructured data it
returns a full-rank train and a wasted SVD, and the diagnosis is always the
same: check the ranks.

Operators quantize too
~~~~~~~~~~~~~~~~~~~~~~

A matrix of size :math:`2^{L} \times 2^{L}` quantizes into a TTM with
:math:`L` cores of mode size :math:`2 \times 2`. Finite-difference
operators do this exceptionally well, because they couple neighbouring
points only.

The 2D Laplacian shipped with this library's benchmarks is assembled
directly in QTT form, never as a matrix:

===========  ==============  ========  ============  ================
:math:`N`    Grid            Cores     Bond          After rounding
===========  ==============  ========  ============  ================
4            16 × 16         8         6             4
6            64 × 64         12        6             4
8            256 × 256       16        6             4
===========  ==============  ========  ============  ================

The bond dimension does not grow with the grid. Only the *number of cores*
grows, and it grows logarithmically: doubling the resolution adds two
cores. The operator for a :math:`10^6 \times 10^6` grid costs the same per
core as the one for :math:`16 \times 16`. This, combined with solvers whose
cost is polynomial in the bond dimension, is what puts grids beyond the
memory wall in reach.

Using it
~~~~~~~~

Conversion is a method on the train, in both directions:

.. code-block:: python

   import numpy as np
   import ttnumpy as tt

   L = 12
   x = np.linspace(0, 1, 2 ** L, endpoint=False)

   # reshape the sampled function to L binary modes, then decompose
   quantized = tt.tt_svd(np.exp(3 * x).reshape([2] * L), error=1e-10)
   print(len(quantized), max(quantized.bonds))    # 12 cores, bond 1

   # and back to coarser modes
   restored = quantized.qtt_to_tt([2 ** 6, 2 ** 6], error=1e-10)
   print(restored.physical_dims())                # [64, 64]

``tt_to_qtt`` converts an existing train, splitting each core into
:math:`\log_{m} n_k` cores of mode size :math:`m`:

.. code-block:: python

   train = tt.rand([16, 16], ranks=3, seed=0)
   qtt = train.tt_to_qtt(mode_size=2, error=1e-12)
   print(len(qtt))   # 8 cores: 4 bits per mode

Both directions take ``error`` and ``max_rank``, because both perform SVDs.
Every mode size must be an exact power of ``mode_size``, or the call raises
``DimensionMismatch``. Note also that a tensor train needs at least two
cores, so ``qtt_to_tt`` cannot reconstruct a single combined mode — split
the target across two or more physical indices.

Choosing the mode size
~~~~~~~~~~~~~~~~~~~~~~

``mode_size=2`` gives the most cores and the smallest local problems;
larger values fuse bits into fewer, fatter cores. The trade-off:

- **Smaller mode size** — more cores, more sweeping, smaller local systems
  in the solvers, maximum compression.
- **Larger mode size** — fewer cores, fewer sweeps, but local systems in
  the solvers grow with the mode size, and the direct local solve becomes
  expensive sooner (see ``direct_solve_size`` in :doc:`linear_systems`).

Mode size 2 is the default and the right starting point. Fuse only when
sweep overhead measurably dominates.

References
----------

.. [Oseledets2010] I. V. Oseledets, *Approximation of* :math:`2^d \times
   2^d` *matrices using tensor decomposition*, SIAM J. Matrix Anal. Appl.
   31(4), 2130–2145, 2010. https://doi.org/10.1137/090757861

.. [Khoromskij2011] B. N. Khoromskij, :math:`O(d \log N)`-*quantics
   approximation of* :math:`N`-*d tensors in high-dimensional numerical
   modeling*, Constructive Approximation 34, 257–280, 2011.
   https://doi.org/10.1007/s00365-011-9131-1
