High-dimensional integration
============================

Integrals over many variables appear wherever an expectation is taken over
many random inputs, or a quantity sums over many degrees of freedom: Feynman
diagrams in quantum field theory [NF2022]_, expected payoffs in finance,
marginal likelihoods in statistics. Monte Carlo is the standard method. Its
error falls like :math:`N^{-1/2}` with the number :math:`N` of evaluations,
whatever the dimension. A tensor train of a smooth integrand integrates
through one contraction of its cores.

**The integrand.** The test integrand of [NF2025]_, Sec. 5.2,

.. math::

   &f(x) = \frac{2^d}{1 + 2 (x_1 + \cdots + x_d)} \quad \text{on } [0, 1]^d,

   &\int_{[0, 1]^d} f \, dx = \frac{1}{(d - 1)!} \sum_{k=0}^{d} (-1)^{d-k}
   \binom{d}{k} (1 + 2k)^{d-1} \ln(1 + 2k),

has a closed-form integral. The integrand depends on the sum
:math:`s = x_1 + \cdots + x_d` only. Its integral over the cube is the
:math:`d`-th forward difference of a :math:`d`-fold antiderivative of
:math:`2^d / (1 + 2s)`. One such antiderivative is
:math:`(1 + 2s)^{d-1} \ln(1 + 2s) / (d - 1)!`, up to a polynomial of degree
:math:`d - 1` that the difference removes.
The terms of the sum are far larger than the result, so the script evaluates
it with 50-digit decimal arithmetic.

**Quadrature on a tensor train.** With :math:`n` Gauss–Legendre nodes
:math:`x^{(j)}` and weights :math:`w^{(j)}` on :math:`[0, 1]` per variable,
the tensor-product rule sums over :math:`n^d` nodes, :math:`8^{20} \approx
10^{18}` here:

.. math::

   \int_{[0, 1]^d} f \, dx \approx \sum_{i_1, \ldots, i_d}
   w^{(i_1)} \cdots w^{(i_d)} \, F[i_1, \ldots, i_d],
   \qquad F[i_1, \ldots, i_d] = f\big(x^{(i_1)}, \ldots, x^{(i_d)}\big).

When :math:`F` is a train with cores :math:`G_s`, the sum separates into a
product of :math:`d` small matrices,

.. math::

   \Big( \sum_{i} w^{(i)} G_1[i] \Big) \Big( \sum_{i} w^{(i)} G_2[i] \Big)
   \cdots \Big( \sum_{i} w^{(i)} G_d[i] \Big),

at a cost of :math:`O(d n r^2)`. TCI builds :math:`F` from its values at the
nodes it chooses, with the element function mapping indices to nodes
(:doc:`../user_guide/cross_interpolation`).

**Monte Carlo at the same cost.** For every threshold the script draws as many
uniform random points as TCI evaluated :math:`f`. It prints the Monte Carlo
error with its standard error :math:`\sigma / \sqrt{N}`, both relative to the
integral.

**Results.** The error of TCI follows the threshold, down to
:math:`6 \cdot 10^{-11}` in 10 dimensions and :math:`3 \cdot 10^{-10}` in
20, from 70 297 and 295 132 evaluations of :math:`f`. Monte Carlo with the
same evaluations stays between :math:`10^{-4}` and :math:`10^{-3}`, as its
standard error predicts. Eight nodes per variable suffice here: at the
tightest threshold the whole error, quadrature included, is below
:math:`10^{-9}`. The ranks stay at 12 or below. Doubling the dimension
multiplies the evaluations by about four.

Script
------

:download:`integral.py <../../examples/integral.py>`

.. literalinclude:: ../../examples/integral.py
   :language: python
   :linenos:
   :lines: 4-

Output
------

.. code-block:: text

     d threshold  evaluations  rank  TCI error   MC error   MC sigma
    10     1e-04        12432     5    1.6e-04    8.5e-04    1.6e-03
    10     1e-07        31168     8    1.2e-07    4.1e-04    1.0e-03
    10     1e-10        70297    11    6.0e-11    6.7e-04    6.9e-04
    20     1e-04        47754     5    3.6e-05    1.8e-04    5.9e-04
    20     1e-07       176700     9    5.8e-08    4.4e-04    3.1e-04
    20     1e-10       295132    12    3.0e-10    8.6e-05    2.4e-04
