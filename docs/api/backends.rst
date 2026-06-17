Backends
========

The :mod:`qtest.backends` subpackage abstracts circuit execution behind
a single :class:`~qtest.backends.Backend` protocol. Qiskit (default), Cirq,
and PennyLane implementations all ship in the box and are auto-registered
under the names ``"qiskit"``, ``"cirq"``, and ``"pennylane"``. Select one
per call (``backend=CirqBackend()``) or globally
(:func:`set_default_backend` / ``--qtest-backend``).

Each backend operates on its SDK's *native* circuit type — a
``qiskit.QuantumCircuit``, a ``cirq.Circuit``, or a
``pennylane.tape.QuantumScript`` respectively — and measurement-bitstring
endianness follows that SDK's convention (Qiskit is little-endian; Cirq and
PennyLane put the lowest-index qubit leftmost). qtest's
:class:`~qtest.noise.NoiseModel` targets Qiskit Aer, so the Cirq and
PennyLane backends raise :class:`NotImplementedError` if a ``noise_model``
is supplied.

Install the optional SDKs with the matching extra::

   pip install 'qtest-quantum[cirq]'        # cirq-core
   pip install 'qtest-quantum[pennylane]'   # pennylane

.. currentmodule:: qtest.backends

Public API
----------

.. autosummary::
   :nosignatures:

   Backend
   CircuitResources
   QiskitBackend
   CirqBackend
   PennyLaneBackend
   get_backend
   get_default_backend
   set_default_backend
   register_backend
   list_available_backends

Reference
---------

.. automodule:: qtest.backends
   :members:
   :show-inheritance:
   :member-order: bysource
   :exclude-members: CircuitResources

Backend protocol
~~~~~~~~~~~~~~~~

.. automodule:: qtest.backends.base
   :members:
   :show-inheritance:
   :member-order: bysource

Registry
~~~~~~~~

.. automodule:: qtest.backends.registry
   :members:
   :show-inheritance:
   :member-order: bysource

Qiskit backend
~~~~~~~~~~~~~~

.. automodule:: qtest.backends.qiskit_backend
   :members:
   :show-inheritance:
   :member-order: bysource

Cirq backend
~~~~~~~~~~~~

.. automodule:: qtest.backends.cirq_backend
   :members:
   :show-inheritance:
   :member-order: bysource

PennyLane backend
~~~~~~~~~~~~~~~~~

.. automodule:: qtest.backends.pennylane_backend
   :members:
   :show-inheritance:
   :member-order: bysource
