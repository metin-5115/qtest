"""Qiskit-based simulator backend for qtest.

This module is import-safe even when Qiskit is not installed: all
``import qiskit ...`` calls are deferred to method bodies. Constructing
:class:`QiskitBackend` does not require Qiskit; calling any of its execution
methods does.

Qiskit 1.0+ removed ``qiskit.execute`` and ``qiskit.BasicAer``; we use
``qiskit.transpile`` + ``backend.run(...)`` for sampling, and
``qiskit.quantum_info.{Statevector, Operator}`` for state / unitary
extraction (no Aer required for those).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qtest.backends.base import Backend

_QISKIT_MISSING_MSG = (
    "QiskitBackend requires Qiskit. Install with one of:\n"
    "  pip install 'qtest[aer]'   # qtest + qiskit + qiskit-aer (recommended)\n"
    "  pip install qiskit         # core only (slower BasicSimulator fallback)"
)


def _require_qiskit() -> Any:
    """Import and return the :mod:`qiskit` module, with a friendly error."""
    try:
        import qiskit
    except ImportError as exc:
        raise ImportError(_QISKIT_MISSING_MSG) from exc
    return qiskit


class QiskitBackend(Backend):
    """Qiskit simulator backend.

    Parameters
    ----------
    simulator_name
        Cosmetic identifier reported via :attr:`name`. The actual simulator
        chosen at run time is :class:`qiskit_aer.AerSimulator` if available,
        else :class:`qiskit.providers.basic_provider.BasicSimulator`.
    shots
        Default shot count used when :meth:`run_circuit` is called without
        an explicit ``shots`` argument.
    optimization_level
        Default ``transpile`` optimisation level (one of ``0, 1, 2, 3``).

    Notes
    -----
    The constructor arguments are *defaults*; per-call arguments to
    :meth:`run_circuit` override them. The backend keeps no other state
    between calls — two calls with identical arguments produce identical
    output (assuming the same ``seed``).
    """

    def __init__(
        self,
        simulator_name: str = "aer_simulator",
        shots: int = 1024,
        optimization_level: int = 1,
    ) -> None:
        if not isinstance(simulator_name, str) or not simulator_name:
            raise ValueError("simulator_name must be a non-empty string")
        if not isinstance(shots, int) or isinstance(shots, bool) or shots <= 0:
            raise ValueError(f"shots must be a positive integer, got {shots!r}")
        if optimization_level not in (0, 1, 2, 3):
            raise ValueError(
                f"optimization_level must be one of {{0, 1, 2, 3}}, " f"got {optimization_level!r}"
            )
        self._simulator_name = simulator_name
        self._default_shots = shots
        self._default_optimization_level = optimization_level

    # ------------------------------------------------------------------ #
    # Metadata                                                            #
    # ------------------------------------------------------------------ #

    @property
    def name(self) -> str:
        return f"qiskit:{self._simulator_name}"

    @property
    def supports_statevector(self) -> bool:
        return True

    # ------------------------------------------------------------------ #
    # Simulator selection                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_sampling_simulator() -> Any:
        """Return an instantiated simulator backend suitable for sampling.

        Prefers ``qiskit_aer.AerSimulator``; falls back to
        ``qiskit.providers.basic_provider.BasicSimulator``.
        """
        try:
            from qiskit_aer import AerSimulator

            return AerSimulator()
        except ImportError:
            from qiskit.providers.basic_provider import BasicSimulator

            return BasicSimulator()

    # ------------------------------------------------------------------ #
    # Execution                                                           #
    # ------------------------------------------------------------------ #

    def run_circuit(
        self,
        circuit: Any,
        shots: int | None = None,
        seed: int | None = None,
    ) -> dict[str, int]:
        qiskit = _require_qiskit()

        if circuit is None:
            raise ValueError("circuit must not be None")

        shots_val = self._default_shots if shots is None else shots
        if not isinstance(shots_val, int) or isinstance(shots_val, bool) or shots_val <= 0:
            raise ValueError(f"shots must be a positive integer, got {shots_val!r}")

        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise ValueError(f"seed must be an integer or None, got {seed!r}")

        simulator = self._get_sampling_simulator()
        transpiled = qiskit.transpile(
            circuit,
            simulator,
            optimization_level=self._default_optimization_level,
        )

        run_kwargs: dict[str, Any] = {"shots": shots_val}
        if seed is not None:
            run_kwargs["seed_simulator"] = seed

        try:
            job = simulator.run(transpiled, **run_kwargs)
        except TypeError:
            run_kwargs.pop("seed_simulator", None)
            job = simulator.run(transpiled, **run_kwargs)

        result = job.result()
        return dict(result.get_counts())

    def get_statevector(self, circuit: Any) -> np.ndarray:
        _require_qiskit()
        if circuit is None:
            raise ValueError("circuit must not be None")

        from qiskit.quantum_info import Statevector

        sv = Statevector.from_instruction(circuit)
        return np.asarray(sv.data, dtype=complex)

    def get_unitary(self, circuit: Any) -> np.ndarray:
        _require_qiskit()
        if circuit is None:
            raise ValueError("circuit must not be None")

        from qiskit.quantum_info import Operator

        op = Operator(circuit)
        return np.asarray(op.data, dtype=complex)
