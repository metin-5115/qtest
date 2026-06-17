"""PennyLane-based simulator backend for qtest.

Import-safe without PennyLane: every ``import pennylane`` is deferred to a
method body, so constructing :class:`PennyLaneBackend` (and importing this
module) never requires PennyLane.

Native circuit type
--------------------
Unlike Qiskit/Cirq, PennyLane has no free-standing "circuit" object — programs
are expressed as quantum functions bound to a device. This backend therefore
operates on a :class:`pennylane.tape.QuantumScript` (a.k.a. ``QuantumTape``),
which is the closest device-independent container of operations. Build one as::

    import pennylane as qml
    tape = qml.tape.QuantumScript([qml.Hadamard(0), qml.CNOT([0, 1])])

Measurement bitstrings are ordered with the **lowest-index wire leftmost**.
Existing measurements on the tape are ignored; qtest supplies its own.

Noise
-----
qtest's :class:`~qtest.noise.NoiseModel` targets Qiskit Aer, so passing a
``noise_model`` here raises :class:`NotImplementedError`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qtest.backends.base import Backend

_PENNYLANE_MISSING_MSG = (
    "PennyLaneBackend requires PennyLane. Install it with:\n  pip install 'qtest[pennylane]'"
)


def _require_pennylane() -> Any:
    try:
        import pennylane as qml
    except ImportError as exc:  # pragma: no cover - exercised only without pennylane
        raise ImportError(_PENNYLANE_MISSING_MSG) from exc
    return qml


def _extract(circuit: Any) -> tuple[list[Any], list[Any]]:
    """Return ``(operations, wire_order)`` from a QuantumScript / QuantumTape."""
    if circuit is None:
        raise ValueError("circuit must not be None")
    if not hasattr(circuit, "operations") or not hasattr(circuit, "wires"):
        raise ValueError(
            "PennyLaneBackend expects a pennylane.tape.QuantumScript/QuantumTape; "
            f"got {type(circuit).__name__}."
        )
    operations = list(circuit.operations)
    wire_order = sorted(circuit.wires.tolist())
    if not wire_order:
        raise ValueError("circuit has no wires")
    return operations, wire_order


class PennyLaneBackend(Backend):
    """PennyLane ``default.qubit`` simulator backend.

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
        return "pennylane:default.qubit"

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
        qml = _require_pennylane()
        if noise_model is not None:
            raise NotImplementedError(
                "PennyLaneBackend does not support qtest noise models (they "
                "target Qiskit Aer). Use the Qiskit backend for noisy simulation."
            )
        shots_val = self._default_shots if shots is None else shots
        if not isinstance(shots_val, int) or isinstance(shots_val, bool) or shots_val <= 0:
            raise ValueError(f"shots must be a positive integer, got {shots_val!r}")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise ValueError(f"seed must be an integer or None, got {seed!r}")

        operations, wire_order = _extract(circuit)
        dev = qml.device("default.qubit", wires=wire_order, seed=seed)
        tape = qml.tape.QuantumScript(operations, [qml.counts(all_outcomes=True)], shots=shots_val)
        counts = qml.execute([tape], dev)[0]
        return {str(k): int(v) for k, v in counts.items() if int(v) > 0}

    def get_statevector(self, circuit: Any) -> np.ndarray:
        qml = _require_pennylane()
        operations, wire_order = _extract(circuit)
        dev = qml.device("default.qubit", wires=wire_order)
        tape = qml.tape.QuantumScript(operations, [qml.state()])
        state = qml.execute([tape], dev)[0]
        return np.asarray(state, dtype=complex)

    def get_unitary(self, circuit: Any) -> np.ndarray:
        qml = _require_pennylane()
        operations, wire_order = _extract(circuit)
        matrix = qml.matrix(qml.tape.QuantumScript(operations, []), wire_order=wire_order)
        return np.asarray(matrix, dtype=complex)
