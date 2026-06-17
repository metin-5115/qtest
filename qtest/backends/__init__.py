"""Backend abstraction layer for qtest.

Public API:

* :class:`Backend` — abstract interface every backend implements.
* :class:`CircuitResources` — structural resource metrics of a circuit.
* :class:`QiskitBackend` — concrete Qiskit adapter (default).
* :class:`CirqBackend` — concrete Cirq adapter (registered as ``"cirq"``).
* :class:`PennyLaneBackend` — concrete PennyLane adapter (``"pennylane"``).
* :func:`get_backend`, :func:`get_default_backend`, :func:`set_default_backend`,
  :func:`register_backend`, :func:`list_available_backends` — registry helpers.

Importing this package does **not** import any quantum SDK; each SDK import
happens lazily, the first time that backend's execution method is called.
"""

from qtest.backends.base import Backend, CircuitResources
from qtest.backends.cirq_backend import CirqBackend
from qtest.backends.pennylane_backend import PennyLaneBackend
from qtest.backends.qiskit_backend import QiskitBackend
from qtest.backends.registry import (
    get_backend,
    get_default_backend,
    list_available_backends,
    register_backend,
    set_default_backend,
)

__all__ = [
    "Backend",
    "CircuitResources",
    "CirqBackend",
    "PennyLaneBackend",
    "QiskitBackend",
    "get_backend",
    "get_default_backend",
    "list_available_backends",
    "register_backend",
    "set_default_backend",
]
