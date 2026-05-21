Backends
========

The :mod:`qtest.backends` subpackage abstracts circuit execution behind
a single :class:`~qtest.backends.Backend` protocol. A Qiskit
implementation ships in the box; the protocol is designed so Cirq and
PennyLane adapters can be slotted in without touching user code.

.. currentmodule:: qtest.backends

Public API
----------

.. autosummary::
   :nosignatures:

   Backend
   QiskitBackend
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
