"""``assert_entangled`` / ``assert_separable`` — entanglement-aware assertions.

Both operate on the *pure* state a circuit prepares (or a raw state vector).
Entanglement is judged via the entanglement entropy of a bipartition: the von
Neumann entropy of the reduced density matrix on a subsystem is ``0`` iff that
cut is a product state, and positive otherwise.

* :func:`assert_entangled` — the state is entangled across the chosen cut
  (or, with ``qubits=None``, is not a fully-product state).
* :func:`assert_separable` — the state is a product across the chosen cut
  (or, with ``qubits=None``, is fully separable on every qubit).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qtest.backends import Backend
from qtest.backends.registry import get_backend
from qtest.config import get_config
from qtest.metrics import entanglement_entropy

# Default entropy threshold (bits) separating "separable" from "entangled".
_DEFAULT_TOL = 1e-9


def assert_entangled(
    circuit: Any,
    qubits: list[int] | None = None,
    tolerance: float = _DEFAULT_TOL,
    backend: Backend | None = None,
    msg: str | None = None,
) -> None:
    """Assert that *circuit*'s state is entangled.

    Parameters
    ----------
    circuit
        A backend-native circuit (no measurements) or a 1-D state-vector array.
    qubits
        Subsystem defining the bipartition ``qubits | rest``. The assertion
        passes when the entanglement entropy across this cut exceeds
        *tolerance*. When ``None``, the state must be entangled *somewhere* —
        i.e. it is not a fully-product state (some single-qubit reduced state
        is mixed).
    tolerance
        Entropy threshold in bits. Entanglement entropy must be strictly
        greater than this to count as entangled.
    backend
        Backend used to extract the state vector (defaults to the configured
        backend).
    msg
        Optional prefix prepended to the assertion failure message.

    Raises
    ------
    AssertionError
        When the state is separable across the chosen cut (entropy ≤ tolerance).
    """
    state, n = _get_state(circuit, backend)
    entropy, label = _bipartition_entropy(state, n, qubits, want_max=True)
    if entropy > tolerance:
        return
    raise AssertionError(
        _format_failure(
            circuit=circuit,
            n=n,
            entropy=entropy,
            tolerance=tolerance,
            cut_label=label,
            expected="entangled (entropy > tolerance)",
            user_msg=msg,
        )
    )


def assert_separable(
    circuit: Any,
    qubits: list[int] | None = None,
    tolerance: float = 1e-9,
    backend: Backend | None = None,
    msg: str | None = None,
) -> None:
    """Assert that *circuit*'s state is separable (no entanglement).

    Parameters
    ----------
    circuit
        A backend-native circuit (no measurements) or a 1-D state-vector array.
    qubits
        Subsystem defining the bipartition ``qubits | rest``. The assertion
        passes when the entanglement entropy across this cut is at most
        *tolerance*. When ``None``, *every* qubit must be unentangled (the
        state is a full product state).
    tolerance
        Entropy threshold in bits. Entanglement entropy must be ≤ this.
    backend
        Backend used to extract the state vector.
    msg
        Optional prefix prepended to the assertion failure message.

    Raises
    ------
    AssertionError
        When the state is entangled across the chosen cut (entropy > tolerance).
    """
    state, n = _get_state(circuit, backend)
    entropy, label = _bipartition_entropy(state, n, qubits, want_max=True)
    if entropy <= tolerance:
        return
    raise AssertionError(
        _format_failure(
            circuit=circuit,
            n=n,
            entropy=entropy,
            tolerance=tolerance,
            cut_label=label,
            expected="separable (entropy <= tolerance)",
            user_msg=msg,
        )
    )


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _get_state(circuit: Any, backend: Backend | None) -> tuple[np.ndarray, int]:
    """Return ``(state_vector, num_qubits)`` from a circuit or raw array."""
    if isinstance(circuit, (np.ndarray, list, tuple)):
        vec = np.asarray(circuit, dtype=complex)
        if vec.ndim != 1 or vec.size < 2 or (vec.size & (vec.size - 1)):
            raise ValueError(
                f"state vector must be 1-D with a power-of-two length ≥ 2, got {vec.shape}"
            )
        return vec, vec.size.bit_length() - 1

    if backend is None:
        backend = get_backend(get_config().default_backend)
    if not backend.supports_statevector:
        raise ValueError(f"Backend {backend.name!r} does not support state-vector extraction.")
    vec = np.asarray(backend.get_statevector(circuit), dtype=complex)
    if vec.ndim != 1 or vec.size < 2 or (vec.size & (vec.size - 1)):
        raise ValueError(f"backend returned an invalid state vector of shape {vec.shape}")
    return vec, vec.size.bit_length() - 1


def _bipartition_entropy(
    state: np.ndarray, n: int, qubits: list[int] | None, want_max: bool
) -> tuple[float, str]:
    """Return ``(entropy, cut_label)`` for the chosen bipartition.

    With *qubits* given, the entropy is that of the ``qubits | rest`` cut. With
    ``qubits=None`` the worst-case single-qubit entropy is returned, which is
    positive iff the state is not a full product state.
    """
    if qubits is not None:
        if not qubits or any(q < 0 or q >= n for q in qubits):
            raise ValueError(f"qubits must be a non-empty subset of [0, {n}); got {qubits}")
        return entanglement_entropy(state, qubits, n), f"{sorted(set(qubits))} | rest"

    if n < 2:
        raise ValueError("entanglement requires at least 2 qubits; pass qubits for a 1-qubit cut")
    per_qubit = [entanglement_entropy(state, [q], n) for q in range(n)]
    idx = int(np.argmax(per_qubit)) if want_max else int(np.argmin(per_qubit))
    return per_qubit[idx], f"qubit {idx} | rest (worst of all single-qubit cuts)"


def _circuit_summary(circuit: Any) -> str:
    if isinstance(circuit, np.ndarray):
        return f"<state vector dim={circuit.size}>"
    name = getattr(circuit, "name", None)
    n_qubits = getattr(circuit, "num_qubits", None)
    if name and n_qubits is not None:
        return f"{name} ({n_qubits} qubits)"
    return str(name) if name else f"<{type(circuit).__name__}>"


def _format_failure(
    *,
    circuit: Any,
    n: int,
    entropy: float,
    tolerance: float,
    cut_label: str,
    expected: str,
    user_msg: str | None,
) -> str:
    lines: list[str] = []
    if user_msg:
        lines.extend([user_msg, ""])
    lines.append("Entanglement assertion failed")
    lines.append("")
    lines.append(f"  Circuit: {_circuit_summary(circuit)}")
    lines.append(f"  Qubits: {n}")
    lines.append(f"  Bipartition: {cut_label}")
    lines.append(f"  Entanglement entropy: {entropy:.6g} bits")
    lines.append(f"  Tolerance: {tolerance:g}")
    lines.append(f"  Expected: {expected}")
    return "\n".join(lines)
