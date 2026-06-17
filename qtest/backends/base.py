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
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CircuitResources:
    """Structural / resource metrics of a circuit, free of any execution.

    These describe how *expensive* a circuit is to run on hardware — the
    quantities transpilers and optimisers try to minimise — and are the basis
    for :mod:`qtest.assertions.resources`.
    """

    #: Number of qubits the circuit acts on.
    num_qubits: int
    #: Circuit depth (length of the critical path of gates).
    depth: int
    #: Total number of operations (gate instances).
    size: int
    #: Mapping ``{gate_name: count}``; gate names are backend-specific.
    gate_counts: dict[str, int] = field(default_factory=dict)
    #: Number of gates acting on exactly two qubits.
    two_qubit_count: int = 0
    #: Number of gates acting on two *or more* qubits.
    multi_qubit_count: int = 0

    @property
    def t_count(self) -> int:
        """Number of T / T-dagger gates (a key fault-tolerant cost metric)."""
        return self.gate_counts.get("t", 0) + self.gate_counts.get("tdg", 0)


class Backend(ABC):
    """Abstract base class for quantum circuit execution backends."""

    @abstractmethod
    def run_circuit(
        self,
        circuit: Any,
        shots: int | None = None,
        seed: int | None = None,
        noise_model: Any | None = None,
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
        noise_model
            Optional :class:`qtest.noise.NoiseModel` describing the noise to
            apply during simulation. ``None`` (the default) runs an ideal,
            noiseless simulation. Backends that cannot simulate noise should
            raise :class:`NotImplementedError` when a non-``None`` model is
            passed.

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

    def get_density_matrix(self, circuit: Any, noise_model: Any | None = None) -> np.ndarray:
        """Return the density matrix produced by *circuit*.

        With ``noise_model=None`` this is the pure-state projector
        :math:`|\\psi\\rangle\\langle\\psi|`; with a noise model it is the mixed
        state resulting from noisy evolution. The returned array has
        ``dtype=complex`` and shape ``(2**n_qubits, 2**n_qubits)``.

        The default implementation raises :class:`NotImplementedError`;
        backends capable of density-matrix simulation override it.
        """
        raise NotImplementedError(
            f"Backend {self.name!r} does not support density-matrix extraction."
        )

    def get_resources(self, circuit: Any) -> CircuitResources:
        """Return structural resource metrics for *circuit*.

        This is a static analysis of the circuit — no simulation is run, so
        it works on circuits of any width. The default implementation raises
        :class:`NotImplementedError`; backends able to introspect their native
        circuit type override it.
        """
        raise NotImplementedError(f"Backend {self.name!r} does not support resource extraction.")

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend identifier."""

    @property
    @abstractmethod
    def supports_statevector(self) -> bool:
        """Whether :meth:`get_statevector` / :meth:`get_unitary` are supported."""
