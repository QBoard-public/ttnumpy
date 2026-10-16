Poisson equation in 2D
======================

The Poisson problem on the unit square,

.. math::

   -\Big( \frac{\partial^2 u}{\partial x^2}
   + \frac{\partial^2 u}{\partial y^2} \Big) = 2 \pi^2 \sin(\pi x) \sin(\pi y)
   \quad \text{in } (0, 1)^2, \qquad u = 0 \quad \text{on the boundary},

has the solution :math:`u(x, y) = \sin(\pi x) \sin(\pi y)`. Five-point finite
differences on :math:`K \times K` interior points :math:`x_i = (i + 1) h`,
:math:`h = 1 / (K + 1)`, turn it into the linear system

.. math::

   A u_h = f_h, \qquad
   A = A_1 \otimes I + I \otimes A_1, \qquad
   A_1 = \frac{1}{h^2} \begin{pmatrix}
   2 & -1 & & \\ -1 & 2 & \ddots & \\ & \ddots & \ddots & -1 \\ & & -1 & 2
   \end{pmatrix}.

The script solves it on grids of up to :math:`2^{15}` points per axis,
:math:`1.1 \cdot 10^9` unknowns, without forming :math:`A` or any grid
function in full.

**Quantization.** With :math:`K = 2^N`, each index is written in binary, most
significant bit first. Each bit becomes a site of mode size 2 (:ref:`qtt`).
The bits of :math:`x` come first, then those of :math:`y`. A grid function is
then a train of :math:`2N` sites and :math:`A` a train matrix over them.

**The 1D operator from its cores.** :math:`A_1 = (2I - S - S^\top) / h^2`,
where the shift :math:`S` maps index :math:`i` to :math:`i + 1`. Adding one in
binary flips the last bit and passes a carry towards the first bit for as long
as the bits it meets are 1. On one bit, with

.. math::

   I = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}, \qquad
   J = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix},

:math:`J` turns a 0 into a 1 without a carry. :math:`J^\top` turns a 1 into a
0 and passes a carry on. The cores of :math:`A_1` are matrices whose entries
are such :math:`2 \times 2` blocks [KK2012]_:

.. math::

   A_1 = \begin{pmatrix} I & J & J^\top \end{pmatrix}
   \bowtie
   \begin{pmatrix} I & J & J^\top \\ 0 & J^\top & 0 \\ 0 & 0 & J \end{pmatrix}
   \bowtie \cdots \bowtie
   \begin{pmatrix} I & J & J^\top \\ 0 & J^\top & 0 \\ 0 & 0 & J \end{pmatrix}
   \bowtie
   \frac{1}{h^2} \begin{pmatrix} 2I - J - J^\top \\ -J^\top \\ -J
   \end{pmatrix}.

The product :math:`\bowtie` multiplies the cores as matrices, with the
Kronecker product of two blocks in place of the product of two numbers. The
bond index is 0 while no carry is pending, 1 while :math:`S` carries and 2
while :math:`S^\top` carries, so the rank is 3 on every grid.

**The 2D operator.** The Kronecker product of two trains is the list of the
cores of the first followed by those of the second. Adding
:math:`A_1 \otimes I` and :math:`I \otimes A_1` adds their bond dimensions, so
:math:`A` has bond dimension at most 4.

**Right-hand side and exact solution.** The samples of :math:`\sin(\pi x)`
form a train of rank 2, found by TT-SVD of its :math:`K` values
(:ref:`tt-svd`). The right-hand side and the exact solution are Kronecker
products of it with itself. Neither is formed in full.

**The error, predicted.** The grid function :math:`\sin(\pi x) \sin(\pi y)`
is an eigenvector of :math:`A` with eigenvalue
:math:`\lambda_h = (8 / h^2) \sin^2(\pi h / 2)`. The discrete solution is
therefore :math:`(2 \pi^2 / \lambda_h) \, u`, with the relative error

.. math::

   \Big| \frac{2 \pi^2}{\lambda_h} - 1 \Big| \approx \frac{\pi^2 h^2}{12}.

The script prints this prediction next to the error of every computed
solution and asserts that the two agree.

**Results.** Up to :math:`N = 12`, :math:`1.7 \cdot 10^7` unknowns, the error
equals the prediction to three digits. The solution has rank 2 on every grid.
On the two finest grids AMEn stagnates with a relative residual near
:math:`10^{-8}`. There the error no longer follows the prediction. The residual
floor grows about fourfold per extra bit, as the condition number of
:math:`A` does: it comes from the finite precision of the computation, which
[BK2020]_ analyse for low-rank representations of such operators and remove
with a multilevel preconditioner.

On :math:`2^{10} \times 2^{10}` points AMEn, MALS and ALS all reach the
predicted error. ALS keeps the ranks of its initial guess
(:doc:`../user_guide/linear_systems`). Its default guess, an all-ones train
with the ranks of the right-hand side, already has the ranks of the solution.
From a guess of rank 1 it stagnates with an error of order 1.

Script
------

:download:`poisson_2d.py <../../examples/poisson_2d.py>`

.. literalinclude:: ../../examples/poisson_2d.py
   :language: python
   :linenos:
   :lines: 4-

Output
------

.. code-block:: text

   AMEn on grids of 2^N x 2^N points
     N   unknowns  rank  sweeps  residual      error  predicted  stop
     4   2.56e+02     2       1   6.0e-15  2.851e-03  2.851e-03  converged
     6   4.10e+03     2       1   1.3e-13  1.947e-04  1.947e-04  converged
     8   6.55e+04     2       2   2.2e-12  1.245e-05  1.245e-05  converged
    10   1.05e+06     2       2   3.0e-11  7.828e-07  7.828e-07  converged
    12   1.68e+07     2       3   5.1e-10  4.904e-08  4.900e-08  stagnated
    14   2.68e+08     2       4   9.2e-09  1.849e-09  3.064e-09  stagnated
    15   1.07e+09     2       6   3.1e-08  2.177e-08  7.659e-10  stagnated

   The three solvers on 2^10 x 2^10 points
   solver                        sweeps      error  stop
   amen                               2  7.828e-07  converged
   mals                               1  7.828e-07  converged
   als, x0 with the ranks of f        3  7.828e-07  converged
   als, x0 of rank 1                  2  9.968e-01  stagnated

References
----------

.. [KK2012] V. A. Kazeev and B. N. Khoromskij, *Low-rank explicit QTT
   representation of the Laplace operator and its inverse*, SIAM J. Matrix
   Anal. Appl. 33(3), 742–758, 2012. https://doi.org/10.1137/100820479

.. [BK2020] M. Bachmayr and V. Kazeev, *Stability of low-rank tensor
   representations and structured multilevel preconditioning for elliptic
   PDEs*, Found. Comput. Math. 20(5), 1175–1236, 2020.
   https://doi.org/10.1007/s10208-020-09446-z
