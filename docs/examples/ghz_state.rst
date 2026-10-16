GHZ and W states
================

Two entangled states of :math:`d` qubits,

.. math::

   |\mathrm{GHZ}\rangle = \frac{|0 \ldots 0\rangle + |1 \ldots 1\rangle}{\sqrt 2},
   \qquad
   |W\rangle = \frac{|10 \ldots 0\rangle + |010 \ldots 0\rangle + \cdots
   + |0 \ldots 01\rangle}{\sqrt d},

have Schmidt rank 2 across every cut, so each is a tensor train of bond
dimension 2 (:ref:`why-low-rank`). The script prints the cores of GHZ on two
qubits. It then builds both states on :math:`d = 10` qubits in more than one
way, applies two operators to them and compares every result with its closed
form.

**GHZ from its cores.** Each core holds one matrix per value of its index.
On two qubits GHZ is :math:`(|00\rangle + |11\rangle) / \sqrt 2`, with the
cores

.. math::

   G_0[0] = \begin{pmatrix} 1 & 0 \end{pmatrix}, \quad
   G_0[1] = \begin{pmatrix} 0 & 1 \end{pmatrix}, \qquad
   G_1[0] = \frac{1}{\sqrt 2} \begin{pmatrix} 1 \\ 0 \end{pmatrix}, \quad
   G_1[1] = \frac{1}{\sqrt 2} \begin{pmatrix} 0 \\ 1 \end{pmatrix}.

An amplitude is the product of one matrix per site. The amplitude of
:math:`|00\rangle` is :math:`G_0[0] \, G_1[0] = 1 / \sqrt 2`, the amplitude
of :math:`|01\rangle` is :math:`G_0[0] \, G_1[1] = 0`. On more qubits every
core between the first and the last is

.. math::

   G_s[0] = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}, \quad
   G_s[1] = \begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}.

The bond index records whether the qubits so far are all 0 or all 1, so the
product vanishes unless all indices agree.

**Sums of product states.** A basis state such as
:math:`|011\rangle = |0\rangle \otimes |1\rangle \otimes |1\rangle` is a train
of rank 1. Both states are sums of such terms:

.. math::

   &|\mathrm{GHZ}\rangle = \frac{1}{\sqrt 2} \Big(
   |0\rangle \otimes |0\rangle \otimes \cdots \otimes |0\rangle
   + |1\rangle \otimes |1\rangle \otimes \cdots \otimes |1\rangle \Big),

   &|W\rangle = \frac{1}{\sqrt d} \Big(
   |1\rangle \otimes |0\rangle \otimes \cdots \otimes |0\rangle
   + |0\rangle \otimes |1\rangle \otimes \cdots \otimes |0\rangle
   + \cdots
   + |0\rangle \otimes \cdots \otimes |0\rangle \otimes |1\rangle \Big).

Addition adds ranks (:ref:`tt-arithmetic`). GHZ comes out with bond dimension
2 and equals the train built from its cores. W comes out with bond dimension
:math:`d`. Rounding brings it to 2 (:ref:`tt-accuracy`). Across any cut, W
is one term with the 1 left of the cut and :math:`|0 \ldots 0\rangle` right of
it, plus one term with :math:`|0 \ldots 0\rangle` left of the cut and the 1
right of it, so its Schmidt rank is 2.

**Operators.** The Ising chain

.. math::

   H = -\sum_{s=0}^{d-2} Z_s Z_{s+1}, \qquad Z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}.

is diagonal in the basis states: :math:`Z_s Z_{s+1}` is :math:`+1` when
qubits :math:`s` and :math:`s + 1` agree and :math:`-1` when they differ. It
becomes a tensor train matrix by TT-SVD of its dense
:math:`2^d \times 2^d` matrix (:ref:`tt-svd`). TT-SVD finds bond dimension 3,
the exact value for this sum of nearest-neighbour terms. At :math:`d = 10` the
dense matrix already has :math:`10^6` entries, which limits this route to
small :math:`d`. The global spin flip :math:`X \otimes X \otimes \cdots
\otimes X`, with :math:`X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}`
turning :math:`|0\rangle` into :math:`|1\rangle` and back, is written down
from its cores as a train matrix of rank 1.

**Checks.** An inner product applies the transposed, conjugated train, a
matrix with one row, to the other train:
``x.transpose(conjugate=True).dot(y)`` is a train whose modes all have size 1,
holding :math:`\langle x | y \rangle`. The closed forms are:

- :math:`H |\mathrm{GHZ}\rangle = -(d - 1) |\mathrm{GHZ}\rangle`, since every
  bond of :math:`|0 \ldots 0\rangle` and of :math:`|1 \ldots 1\rangle`
  contributes :math:`-1`.
- :math:`\langle W | H | W \rangle = -(d - 1)(d - 4) / d`. The basis state with
  its 1 at site :math:`j` has energy :math:`-(d - 1) + 2 m_j`, where
  :math:`m_j` is the number of bonds touching :math:`j`: one at the ends, two
  inside.
- :math:`X^{\otimes d} |\mathrm{GHZ}\rangle = |\mathrm{GHZ}\rangle` and
  :math:`\langle \mathrm{GHZ} | W \rangle = 0`.

Applying :math:`H` to GHZ multiplies the bond dimensions to
:math:`3 \cdot 2 = 6`. Rounding returns them to 2. The script asserts every
check, so it fails when a result departs from its closed form.

Script
------

:download:`ghz_state.py <../../examples/ghz_state.py>`

.. literalinclude:: ../../examples/ghz_state.py
   :language: python
   :linenos:
   :lines: 4-

Output
------

.. code-block:: text

   G_0[0] = [[1.0, 0.0]]
   G_0[1] = [[0.0, 1.0]]
   G_1[0] = [[0.7071], [0.0]]
   G_1[1] = [[0.0], [0.7071]]

   bonds of GHZ from cores         [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]
   bonds of W as a sum             [1, 10, 10, 10, 10, 10, 10, 10, 10, 10, 1]
   bonds of W after rounding       [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]
   bonds of H from TT-SVD          [1, 2, 3, 3, 3, 3, 3, 3, 3, 2, 1]
   bonds of H GHZ                  [1, 4, 6, 6, 6, 6, 6, 6, 6, 4, 1]
   bonds of H GHZ after rounding   [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]

   quantity                                       computed      exact
   || GHZ from cores - GHZ as a sum ||      0.000000000000     0.0000
   <GHZ|GHZ>                                1.000000000000     1.0000
   <W|W>                                    1.000000000000     1.0000
   <GHZ|W>                                  0.000000000000     0.0000
   || H GHZ - E GHZ ||                      0.000000000000     0.0000
   <GHZ|H|GHZ>                             -9.000000000000    -9.0000
   <W|H|W>                                 -5.400000000000    -5.4000
   || flip GHZ - GHZ ||                     0.000000000000     0.0000
