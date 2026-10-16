Diffusion with uncertain conductivities
=======================================

The properties of a medium are rarely known exactly. The permeability of the
ground changes from layer to layer and is measured at a few points only. The
conductivities of the layers of a wall or a composite vary from batch to
batch. Uncertainty quantification treats such properties as parameters and
asks how the solution depends on them: which values a quantity of interest
takes, with which mean and spread. Elliptic equations with a coefficient that
depends on parameters are its standard model problem [BNT2007]_. Groundwater
flow is one of its main applications [CGST2011]_.

**The model.** A column of six layers, with a uniform source and the solution
held at zero at both ends:

.. math::

   -\frac{d}{dx} \Big( a(x; y) \, \frac{du}{dx} \Big) = 1
   \; \text{on } (0, 1), \quad u(0) = u(1) = 0, \quad
   a(x; y) = e^{y_k} \quad \text{in layer } k = 0, \ldots, 5.

Layer :math:`k` covers :math:`(k/6, (k + 1)/6)`. In groundwater terms
:math:`u` is the hydraulic head under uniform recharge; in a wall it is the
temperature under a uniform heat source. Each :math:`y_k` takes 8 equally
spaced values in :math:`[-1, 1]`, so a conductivity varies by a factor
:math:`e^2 \approx 7.4`. The six parameters have :math:`8^6 = 262\,144`
combinations, all equally likely.

**One system for every combination.** Finite differences on :math:`n = 127`
interior points, with the conductivity on the cell edges, give
:math:`A(y) \, u(y) = 1` with :math:`A(y) = \sum_k e^{y_k} A_k`, where
:math:`A_k` holds the edges of layer :math:`k`. The solutions for all
combinations form one tensor :math:`u(x, y_0, \ldots, y_5)` of
:math:`3.3 \cdot 10^7` entries, a train of 7 sites: the grid point and one
index per parameter. It solves one block-diagonal system

.. math::

   \mathcal{A} = \sum_{k=0}^{5} A_k \otimes I \otimes \cdots \otimes I \otimes
   \underbrace{E}_{\text{site of } y_k} \otimes I \otimes \cdots \otimes I,
   \qquad E = \mathrm{diag}\big(e^{y^{(1)}}, \ldots, e^{y^{(8)}}\big),

where :math:`y^{(1)}, \ldots, y^{(8)}` are the parameter values.
:math:`\mathcal{A}` is a sum of six train matrices of rank 1, so its bond
dimension is 6. AMEn solves the system once for all combinations. Solving
parametric systems this way, rather than one parameter value at a time, is
the approach of [KT2011]_, [KS2011]_ and [DKLM2015]_.

**Reading the solution.** Fixing every parameter site of the train at one
combination leaves the solution for that combination: the first core times
one matrix per parameter site. The script compares 50 random combinations
with direct solves.

**A quantity of interest.** The average of :math:`u`,
:math:`Q(y) = \int_0^1 u \, dx \approx h \sum_i u_i(y)`. Contracting the grid
site of the train with the weights :math:`h` leaves a train over the
parameters alone. Its mean and second moment over all combinations are inner
products with the all-ones train and with itself,

.. math::

   \mathbb{E}[Q] = \frac{\langle Q, 1 \rangle}{8^6}, \qquad
   \mathbb{E}[Q^2] = \frac{\langle Q, Q \rangle}{8^6},

exact over all 262 144 combinations. Monte Carlo with 2 000 direct solves
estimates the same mean, with a standard error of :math:`7 \cdot 10^{-4}`.

**Results.** The solution has bond dimensions up to 38. The train holds
32 009 numbers instead of :math:`3.3 \cdot 10^7`. At 50 random combinations it
agrees with direct solves to :math:`2.4 \cdot 10^{-7}`, within the requested
relative residual of :math:`10^{-6}`. The mean and standard deviation from the
train are exact. At this size the 262 144 problems can still be solved one by
one with a tridiagonal solver: their mean and standard deviation agree with
the train's to all printed digits. Monte Carlo reaches the mean to within its
standard error. Every added layer multiplies the number of combinations by
8, while the train grows only through its ranks.

Script
------

:download:`parametric_diffusion.py <../../examples/parametric_diffusion.py>`

.. literalinclude:: ../../examples/parametric_diffusion.py
   :language: python
   :linenos:
   :lines: 4-

Output
------

.. code-block:: text

   combinations 262144, unknowns 33292288
   operator bonds   [1, 6, 6, 6, 6, 6, 6, 1]
   solution bonds   [1, 15, 35, 38, 30, 20, 8, 1]
   AMEn: converged after 9 sweeps, relative residual 9.9e-07
   numbers stored   32009 instead of 33292288
   largest relative error at 50 random combinations: 2.4e-07

   average of u                         mean          std
   tensor train, all 262144       0.09601388   0.03321518
   Monte Carlo, 2000 samples      0.09584164   0.03238680   +- 7.2e-04 on the mean

References
----------

.. [BNT2007] I. Babuška, F. Nobile and R. Tempone, *A stochastic collocation
   method for elliptic partial differential equations with random input
   data*, SIAM J. Numer. Anal. 45(3), 1005–1034, 2007.
   https://doi.org/10.1137/050645142

.. [CGST2011] K. A. Cliffe, M. B. Giles, R. Scheichl and A. L. Teckentrup,
   *Multilevel Monte Carlo methods and applications to elliptic PDEs with
   random coefficients*, Comput. Visual. Sci. 14(1), 3–15, 2011.
   https://doi.org/10.1007/s00791-011-0160-x

.. [KT2011] D. Kressner and C. Tobler, *Low-rank tensor Krylov subspace
   methods for parametrized linear systems*, SIAM J. Matrix Anal. Appl.
   32(4), 1288–1316, 2011. https://doi.org/10.1137/100799010

.. [KS2011] B. N. Khoromskij and C. Schwab, *Tensor-structured Galerkin
   approximation of parametric and stochastic elliptic PDEs*, SIAM J. Sci.
   Comput. 33(1), 364–385, 2011. https://doi.org/10.1137/100785715

.. [DKLM2015] S. Dolgov, B. N. Khoromskij, A. Litvinenko and H. G. Matthies,
   *Polynomial chaos expansion of random coefficients and the solution of
   stochastic partial differential equations in the tensor train format*,
   SIAM/ASA J. Uncertain. Quantif. 3(1), 1109–1135, 2015.
   https://doi.org/10.1137/140972536
