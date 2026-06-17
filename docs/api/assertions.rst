Assertions
==========

The :mod:`qtest.assertions` subpackage provides the core quantum-aware
assertions: distribution closeness, state closeness, unitarity, circuit
equivalence, noise robustness, and circuit resource/cost limits. Each is
re-exported at the top-level :mod:`qtest` namespace for convenience.

.. currentmodule:: qtest

Public API
----------

.. autosummary::
   :nosignatures:

   assert_distribution_close
   assert_state_close
   assert_unitary
   assert_circuit_equivalent
   assert_robust_to_noise
   assert_entangled
   assert_separable
   assert_measurement_probabilities
   assert_phase
   assert_commutes
   assert_max_depth
   assert_max_gate_count
   assert_max_two_qubit_count
   assert_max_t_count

Reference
---------

.. automodule:: qtest.assertions
   :members:
   :show-inheritance:
   :member-order: bysource

Submodules
----------

.. automodule:: qtest.assertions.distribution
   :members:
   :show-inheritance:
   :member-order: bysource

.. automodule:: qtest.assertions.state
   :members:
   :show-inheritance:
   :member-order: bysource

.. automodule:: qtest.assertions.unitary
   :members:
   :show-inheritance:
   :member-order: bysource

.. automodule:: qtest.assertions.equivalence
   :members:
   :show-inheritance:
   :member-order: bysource

.. automodule:: qtest.assertions.robustness
   :members:
   :show-inheritance:
   :member-order: bysource

.. automodule:: qtest.assertions.entanglement
   :members:
   :show-inheritance:
   :member-order: bysource

.. automodule:: qtest.assertions.marginals
   :members:
   :show-inheritance:
   :member-order: bysource

.. automodule:: qtest.assertions.phase
   :members:
   :show-inheritance:
   :member-order: bysource

.. automodule:: qtest.assertions.commutation
   :members:
   :show-inheritance:
   :member-order: bysource

.. automodule:: qtest.assertions.resources
   :members:
   :show-inheritance:
   :member-order: bysource
