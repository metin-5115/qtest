Strategies
==========

The :mod:`qtest.strategies` subpackage exposes `Hypothesis
<https://hypothesis.readthedocs.io/>`_ strategies for property-based
testing of quantum circuits, gates, and states.

.. note::

   Strategies require the optional ``hypothesis`` extra:
   ``pip install "qtest[hypothesis]"``.

.. currentmodule:: qtest.strategies

Public API
----------

.. autosummary::
   :nosignatures:

   quantum_circuits
   random_gates
   pauli_strings
   random_states
   product_states
   random_density_matrices
   DEFAULT_GATE_SET

Reference
---------

.. automodule:: qtest.strategies
   :members:
   :show-inheritance:
   :member-order: bysource

Circuits
~~~~~~~~

.. automodule:: qtest.strategies.circuits
   :members:
   :show-inheritance:
   :member-order: bysource

Gates
~~~~~

.. automodule:: qtest.strategies.gates
   :members:
   :show-inheritance:
   :member-order: bysource

States
~~~~~~

.. automodule:: qtest.strategies.states
   :members:
   :show-inheritance:
   :member-order: bysource
