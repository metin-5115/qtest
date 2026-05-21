Property-based testing
======================

Quantum software is unusually amenable to property-based testing: laws
like unitarity, reversibility, and Bell-state symmetry hold for *every*
valid circuit, not just one example. qtest builds on `Hypothesis
<https://hypothesis.readthedocs.io/>`_ to make those laws cheap to express
and to shrink failing inputs down to minimal reproducers.

.. note::

   The ``qtest.strategies`` subpackage requires the optional Hypothesis
   extra:

   .. code-block:: bash

      pip install "qtest[hypothesis]"

The strategies at a glance
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Strategy
     - Generates
   * - :func:`~qtest.strategies.quantum_circuits`
     - Random :class:`~qiskit.QuantumCircuit` instances.
   * - :func:`~qtest.strategies.random_gates`
     - Single gate descriptors from a configurable gate set.
   * - :func:`~qtest.strategies.pauli_strings`
     - :math:`n`-qubit Pauli strings over ``{I, X, Y, Z}``.
   * - :func:`~qtest.strategies.random_states`
     - Haar-random pure state vectors.
   * - :func:`~qtest.strategies.product_states`
     - Tensor-product (separable) pure states.
   * - :func:`~qtest.strategies.random_density_matrices`
     - Random mixed states (density matrices).

Your first property test
------------------------

Every unitary circuit, by definition, has a unitary matrix
representation. Hypothesis makes that one-liner:

.. code-block:: python

   from hypothesis import given, settings
   from qtest import assert_unitary
   from qtest.strategies import quantum_circuits

   @given(qc=quantum_circuits(min_qubits=2, max_qubits=4, max_depth=8))
   @settings(deadline=None, max_examples=50)
   def test_random_circuits_are_unitary(qc):
       assert_unitary(qc, tolerance=1e-9)

The ``deadline=None`` matters: Hypothesis' default per-example deadline
is 200 ms, which random circuit simulation can blow past for larger
qubit counts. Disabling it (or setting an explicit, generous value) is
recommended for quantum properties.

Reversibility — a circuit composed with its inverse is the identity
-------------------------------------------------------------------

.. code-block:: python

   from hypothesis import given, settings
   from qtest.strategies import quantum_circuits

   @given(qc=quantum_circuits(min_qubits=1, max_qubits=3, max_depth=6))
   @settings(deadline=None, max_examples=30)
   def test_circuit_inverse_round_trip(qc):
       round_trip = qc.compose(qc.inverse())
       # round_trip should implement the identity up to global phase
       from qiskit.quantum_info import Operator
       import numpy as np
       U = Operator(round_trip).data
       I = np.eye(U.shape[0])
       # Phase-insensitive comparison: |U_ij| should match |I_ij|
       assert np.allclose(np.abs(U), np.abs(I), atol=1e-9)

When Hypothesis finds a counter-example it will *shrink* it — the failing
circuit you actually see is typically the smallest circuit, in qubits
and in depth, that still reproduces the bug. That is the single biggest
debugging win of property-based testing in quantum: a flaky-looking
failure in an 8-qubit, 30-deep circuit shrinks to a 2-qubit, 2-deep one
you can read by eye.

Shrinking in practice
---------------------

Suppose you introduce a bug in your gate decomposition — a subtle phase
error in the implementation of ``RY``. A standard test on a Bell state
might pass (the bug cancels at measurement) and a hand-rolled test on a
single gate might miss it entirely. A property test like the one above
will find a failing circuit and shrink it down to roughly:

.. code-block::

   QuantumCircuit(1)
   qc.ry(0.5, 0)

The minimal reproducer is *handed to you* with a stack trace; you do not
need to bisect, instrument, or guess.

Tuning Hypothesis settings for quantum
--------------------------------------

A few defaults are worth overriding for quantum properties:

.. code-block:: python

   from hypothesis import HealthCheck, settings

   QUANTUM = settings(
       deadline=None,           # circuit simulation is slow; no per-example timeout
       max_examples=50,         # 50 random examples per property is a good default
       suppress_health_check=[
           HealthCheck.too_slow,
           HealthCheck.data_too_large,
       ],
       print_blob=True,         # show the shrunk failing input as a copy-pasteable blob
   )

   @QUANTUM
   @given(qc=quantum_circuits(min_qubits=2, max_qubits=4))
   def test_whatever(qc):
       ...

For long-running CI properties, register a Hypothesis profile (e.g.
``"ci"``) with higher ``max_examples`` and pick it up in ``conftest.py``.

What makes a good quantum property?
-----------------------------------

The best properties are ones that hold *exactly*, by mathematical
definition, regardless of the specific circuit:

* **Unitarity**: :math:`U^\dagger U = I`.
* **Reversibility**: ``qc.compose(qc.inverse())`` is the identity.
* **Commutation**: gates that commute analytically commute numerically.
* **Symmetry**: a circuit that is symmetric in two qubits produces a
  distribution that is symmetric in those bits.
* **Conservation laws**: a circuit that preserves total :math:`Z` parity
  always produces outcomes with the expected parity.

Properties that *don't* work well as property tests:

* Anything that depends on a specific, hand-crafted output state — that
  is what unit tests are for.
* Anything whose tolerance needs to grow with circuit depth — Hypothesis
  will happily find a 30-deep circuit where your generous tolerance
  finally bites.

Where to go next
----------------

* :doc:`writing_assertions` — the assertions you'll combine with
  Hypothesis strategies.
* :doc:`pytest_integration` — how to gate slow property tests behind
  markers and CI profiles.
* :doc:`../api/strategies` — the full autogenerated API reference for
  every strategy.
