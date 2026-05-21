"""Abstract backend interface for qtest.

A :class:`Backend` is a stateless adapter that translates qtest's public API
into a specific quantum SDK's primitives (Qiskit today; Cirq / PennyLane
later). It exposes three core operations — running a circuit with
measurements, extracting the post-circuit state vector, and extracting the
circuit's unitary — plus a couple of metadata properties.

Concrete backends MUST be free of cross-call mutable state: a given
configuration must produce the same output for the same input on every
call. Constructor arguments are immutable defaults; per-call overrides
(e.g. ``shots``) are supported where applicable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Backend(ABC):
    """Abstract base class for quantum circuit execution backends."""

    @abstractmethod
    def run_circuit(
        self,
        circuit: Any,
        shots: int | None = None,
        seed: int | None = None,
    ) -> dict[str, int]:
        """Run *circuit* and return measurement counts.

        Parameters
        ----------
        circuit
            A circuit object native to the backend (e.g. ``QuantumCircuit``).
        shots
            Number of shots. If ``None``, the backend's configured default
            is used.
        seed
            Optional seed for reproducible sampling.

        Returns
        -------
        dict[str, int]
            Mapping ``{bitstring: count}`` whose values sum to *shots*.
        """

    @abstractmethod
    def get_statevector(self, circuit: Any) -> np.ndarray:
        """Return the state vector produced by *circuit* (no measurement).

        The returned array has ``dtype=complex`` and length ``2**n_qubits``.
        """

    @abstractmethod
    def get_unitary(self, circuit: Any) -> np.ndarray:
        """Return the unitary matrix of *circuit*.

        The returned array has ``dtype=complex`` and shape
        ``(2**n_qubits, 2**n_qubits)``. Raises if *circuit* contains
        non-unitary instructions (measurement, reset, ...).
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend identifier."""

    @property
    @abstractmethod
    def supports_statevector(self) -> bool:
        """Whether :meth:`get_statevector` / :meth:`get_unitary` are supported."""
