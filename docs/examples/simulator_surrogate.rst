Surrogate of a black-box simulator
==================================

A simulator turns input parameters into outputs: a finite-element code, a
reservoir model, a battery model. It is often a **black box**. It can be run
for one set of inputs at a time, but its equations are out of reach, because
the code is commercial, legacy, or too complex to restructure. Every run costs
time. Many tasks need the output for far more inputs than can be run. A
**surrogate** replaces the black box by a cheap approximation built from a
limited number of runs [BG2015]_, [DS2019]_.

**All outputs as one tensor.** When each of :math:`d` parameters takes
:math:`n` values, the outputs for all combinations form one tensor
:math:`Q[i_1, \ldots, i_d]` with :math:`n^d` entries. Index :math:`i_k` says
which value parameter :math:`k` has. Computing an entry means running the
black box once. This is the setting of TCI: a tensor too large to compute in
full, whose entries can be computed one at a time. TCI chooses which runs to
make and returns the tensor as a train. Reading any entry from the train is a
product of :math:`d` small matrices. Statistics over all combinations are
contractions of the train.

**Where surrogates are used.**

- *Uncertainty quantification.* Material properties are uncertain: the
  permeability of soil, the stiffness of composite parts, the parameters of
  battery cells. The distribution of an output matters, not one value: its
  mean, its spread, the probability that it exceeds a limit.
- *Calibration and inverse problems.* Fitting model parameters to
  measurements, in particular by Bayesian inference with Markov chain Monte
  Carlo, takes :math:`10^5` to :math:`10^6` runs of the model.
- *Design studies and optimisation.* Scanning combinations of design
  parameters for the best one evaluates the surrogate instead of the black
  box for every candidate.
- *Sensitivity analysis.* Variance-based measures of which parameters matter
  reduce to contractions of the train.
- *Real-time use.* A surrogate built once offline answers in microseconds,
  for control or for a digital twin of a running system.

**The black box here.** The layered medium of :doc:`parametric_diffusion`,
with ten layers instead of six. The conductivity of layer :math:`k` is
:math:`e^{y_k}`, with each :math:`y_k` one of 8 values in :math:`[-1, 1]`.
The output is the average of the solution, :math:`Q(y) = h \sum_i u_i`.
One run is one tridiagonal solve. The ten parameters have
:math:`8^{10} \approx 1.07 \cdot 10^9` combinations.
:doc:`parametric_diffusion` assembled the operator for all combinations as a
train matrix, which needs the equations inside the simulator. Here only the
function ``simulator(config)`` is available. A real simulator takes minutes or
hours per run; this one takes microseconds, so that the example runs in a
second and every result can be checked.

**The surrogate.** :func:`~ttnumpy.tci` samples :math:`Q` on the parameter
grid through a :class:`~ttnumpy.BlackBox`, which also counts the runs. The
result is a train over the ten parameters. Its mean and standard deviation
over all combinations are inner products, as in :doc:`parametric_diffusion`,
exact for the surrogate.

**Checks.** The script runs the simulator at 2 000 random combinations,
outside the :class:`~ttnumpy.BlackBox`. The runs measure the error of the
surrogate and give a Monte Carlo estimate of the mean.

**Results.** TCI made 76 721 runs, 0.007 % of the combinations and about
14 000 times fewer than running all of them. The surrogate has rank 14 and
reproduces the simulator to :math:`2.7 \cdot 10^{-6}` at the random
combinations, for a threshold of :math:`10^{-6}`. Its mean lies within a
quarter of a standard error of the Monte Carlo estimate. With six layers the
same approach needs 23 750 runs and reproduces the exact mean and standard
deviation of :doc:`parametric_diffusion` to all printed digits.

**Where a surrogate fails.** :math:`Q` depends smoothly on the parameters,
which keeps the ranks small. TCI loses its advantage when that fails:

- *Jumps and kinks across the parameters.* An output that switches along a
  direction mixing several parameters, such as pass or fail of a safety
  criterion, or the payoff of an option on the sum of several assets, is not
  of low rank. TCI then samples most of the grid.
- *Noisy black boxes.* A simulator with Monte Carlo inside returns a
  different value at every run. TCI interpolates the noise. A threshold
  below the noise level makes the ranks grow without converging.
- *Continuous parameters* have to be put on a grid first. The surrogate is
  defined at the grid nodes.
- *Features no run reaches*, such as a narrow region of failure, are missed
  without a sign in the pivot errors (:doc:`../user_guide/cross_interpolation`).
  Runs at independent points, like the checks of this script, are the test.

Script
------

:download:`simulator_surrogate.py <../../examples/simulator_surrogate.py>`

.. literalinclude:: ../../examples/simulator_surrogate.py
   :language: python
   :linenos:
   :lines: 4-

Output
------

.. code-block:: text

   combinations     1073741824
   simulator calls  76721 (0.0071 %)
   largest rank     14
   half-sweeps      5, converged: True
   largest relative error at 2000 random combinations: 2.7e-06

   average of u                       mean          std
   surrogate, all 8^10          0.09839461   0.02678155
   Monte Carlo, 2000 calls      0.09852223   0.02664758   +- 6.0e-04 on the mean

References
----------

.. [BG2015] J. Ballani and L. Grasedyck, *Hierarchical tensor approximation of
   output quantities of parameter-dependent PDEs*, SIAM/ASA J. Uncertain.
   Quantif. 3(1), 852–872, 2015. https://doi.org/10.1137/140960980

.. [DS2019] S. Dolgov and R. Scheichl, *A hybrid alternating least
   squares–TT-cross algorithm for parametric PDEs*, SIAM/ASA J. Uncertain.
   Quantif. 7(1), 260–291, 2019. https://doi.org/10.1137/17M1138881
