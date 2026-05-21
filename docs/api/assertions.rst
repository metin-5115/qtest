Assertions
==========

The :mod:`qtest.assertions` subpackage provides the four core
quantum-aware assertions: distribution closeness, state closeness,
unitarity, and circuit equivalence. Each is re-exported at the
top-level :mod:`qtest` namespace for convenience.

.. currentmodule:: qtest

Public API
----------

.. autosummary::
   :nosignatures:

   assert_distribution_close
   assert_state_close
   assert_unitary
   assert_circuit_equivalent

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
