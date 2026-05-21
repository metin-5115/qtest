"""Backend abstraction layer for qtest.

Public API:

* :class:`Backend` — abstract interface every backend implements.
* :class:`QiskitBackend` — concrete Qiskit adapter (default).
* :func:`get_backend`, :func:`get_default_backend`, :func:`set_default_backend`,
  :func:`register_backend`, :func:`list_available_backends` — registry helpers.

Importing this package does **not** import Qiskit; the actual Qiskit import
happens lazily, the first time a :class:`QiskitBackend` execution method is
called.
"""

from qtest.backends.base import Backend
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
    "QiskitBackend",
    "get_backend",
    "get_default_backend",
    "list_available_backends",
    "register_backend",
    "set_default_backend",
]
