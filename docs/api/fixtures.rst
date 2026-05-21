Fixtures
========

The :mod:`qtest.fixtures` subpackage provides ready-to-use pytest
fixtures for common quantum circuits, plus plain factory functions
that can be called outside of pytest.

To make the fixtures discoverable in your suite, list the relevant
plugin modules under ``pytest_plugins`` in your top-level
``conftest.py``::

   pytest_plugins = [
       "qtest.fixtures.common_states",
       "qtest.fixtures.common_gates",
   ]

.. currentmodule:: qtest.fixtures

Public API
----------

.. autosummary::
   :nosignatures:

   bell_circuit
   ghz_circuit
   plus_circuit
   minus_circuit
   w_circuit
   hadamards
   random_clifford_circuit

Reference
---------

.. automodule:: qtest.fixtures
   :members:
   :show-inheritance:
   :member-order: bysource

Common states
~~~~~~~~~~~~~

.. automodule:: qtest.fixtures.common_states
   :members:
   :show-inheritance:
   :member-order: bysource

Common gates
~~~~~~~~~~~~

.. automodule:: qtest.fixtures.common_gates
   :members:
   :show-inheritance:
   :member-order: bysource
