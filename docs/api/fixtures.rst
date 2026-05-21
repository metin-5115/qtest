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

State-preparation fixtures
--------------------------

.. currentmodule:: qtest.fixtures.common_states

.. autosummary::
   :nosignatures:

   bell_state
   plus_state
   minus_state
   ghz_state
   ghz_3
   ghz_4
   ghz_5
   w_state

.. automodule:: qtest.fixtures.common_states
   :members:
   :show-inheritance:
   :member-order: bysource

Gate-layer fixtures
-------------------

.. currentmodule:: qtest.fixtures.common_gates

.. autosummary::
   :nosignatures:

   hadamard_circuit
   random_clifford

.. automodule:: qtest.fixtures.common_gates
   :members:
   :show-inheritance:
   :member-order: bysource

Plain factories (no pytest required)
------------------------------------

The state and gate builders are also exposed as plain factory
functions via the :mod:`qtest.fixtures` package — usable in scripts,
notebooks, or any non-pytest context::

   from qtest.fixtures import bell_circuit, ghz_circuit

   qc = bell_circuit()
   ghz5 = ghz_circuit(5)

.. automodule:: qtest.fixtures
   :members:
   :show-inheritance:
   :member-order: bysource
