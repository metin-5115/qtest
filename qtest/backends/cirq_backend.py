"""Cirq-based simulator backend for qtest.

Import-safe without Cirq: every ``import cirq`` is deferred to a method body,
so constructing :class:`CirqBackend` (and importing this module) never requires
Cirq. Calling an execution method does.

Native circuit type
--------------------
This backend operates on :class:`cirq.Circuit` objects over
:class:`cirq.LineQubit` (or any sortable qubit). Measurement bitstrings are
ordered with the **lowest-index qubit leftmost** (Cirq's big-endian
convention), which is the *opposite* of Qiskit's. Keep your ``expected`` dicts
consistent with whichever backend you select.

Noise
-----
qtest's :class:`~qtest.noise.NoiseModel` is built on Qiskit Aer, so passing a
``noise_model`` to this backend raises :class:`NotImplementedError`. Noiseless
simulation is fully supported.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qtest.backends.base import Backend, CircuitResources

_CIRQ_MISSING_MSG = "CirqBackend requires Cirq. Install it with:\n  pip install 'qtest[cirq]'"

_MEASURE_KEY = "qtest_all"


def _require_cirq() -> Any:
    try:
        import cirq
    except ImportError as exc:  # pragma: no cover - exercised only without cirq
        raise ImportError(_CIRQ_MISSING_MSG) from exc
    return cirq


class CirqBackend(Backend):
    """Cirq state-vector simulator backend.

    Parameters
    ----------
    shots
        Default shot count for :meth:`run_circuit`.
    """

    def __init__(self, shots: int = 1024) -> None:
        if not isinstance(shots, int) or isinstance(shots, bool) or shots <= 0:
            raise ValueError(f"shots must be a positive integer, got {shots!r}")
        self._default_shots = shots

    @property
    def name(self) -> str:
        return "cirq:simulator"

    @property
    def supports_statevector(self) -> bool:
        return True

    # ------------------------------------------------------------------ #
    # Execution                                                           #
    # ------------------------------------------------------------------ #

    def run_circuit(
        self,
        circuit: Any,
        shots: int | None = None,
        seed: int | None = None,
        noise_model: Any | None = None,
    ) -> dict[str, int]:
        cirq = _require_cirq()
        if circuit is None:
            raise ValueError("circuit must not be None")
        if noise_model is not None:
            raise NotImplementedError(
                "CirqBackend does not support qtest noise models (they target "
                "Qiskit Aer). Use the Qiskit backend for noisy simulation."
            )

        shots_val = self._default_shots if shots is None else shots
        if not isinstance(shots_val, int) or isinstance(shots_val, bool) or shots_val <= 0:
            raise ValueError(f"shots must be a positive integer, got {shots_val!r}")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise ValueError(f"seed must be an integer or None, got {seed!r}")

        qubits = sorted(circuit.all_qubits())
        if not qubits:
            raise ValueError("circuit has no qubits to measure")
        # Rebuild without any existing measurements, then measure all qubits.
        ops = [op for moment in circuit for op in moment if not cirq.is_measurement(op)]
        sim_circuit = cirq.Circuit(ops)
        sim_circuit.append(cirq.measure(*qubits, key=_MEASURE_KEY))

        result = cirq.Simulator(seed=seed).run(sim_circuit, repetitions=shots_val)
        histogram = result.histogram(key=_MEASURE_KEY)
        n = len(qubits)
        return {format(int(outcome), f"0{n}b"): int(count) for outcome, count in histogram.items()}

    def get_statevector(self, circuit: Any) -> np.ndarray:
        _require_cirq()
        if circuit is None:
            raise ValueError("circuit must not be None")
        import cirq

        final = cirq.Simulator().simulate(circuit).final_state_vector
        return np.asarray(final, dtype=complex)

    def get_unitary(self, circuit: Any) -> np.ndarray:
        cirq = _require_cirq()
        if circuit is None:
            raise ValueError("circuit must not be None")
        return np.asarray(cirq.unitary(circuit), dtype=complex)

    def get_resources(self, circuit: Any) -> CircuitResources:
        cirq = _require_cirq()
        if circuit is None:
            raise ValueError("circuit must not be None")

        gate_counts: dict[str, int] = {}
        size = 0
        two_qubit = 0
        multi_qubit = 0
        for moment in circuit:
            for op in moment:
                if cirq.is_measurement(op):
                    continue
                size += 1
                # Normalise to lowercase so names line up with Qiskit's
                # (e.g. cirq.T -> "t", cirq.H -> "h").
                name = str(op.gate).lower()
                gate_counts[name] = gate_counts.get(name, 0) + 1
                n = len(op.qubits)
                if n >= 2:
                    multi_qubit += 1
                    if n == 2:
                        two_qubit += 1

        return CircuitResources(
            num_qubits=len(circuit.all_qubits()),
            depth=len(circuit),
            size=size,
            gate_counts=gate_counts,
            two_qubit_count=two_qubit,
            multi_qubit_count=multi_qubit,
        )
