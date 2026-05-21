"""``assert_state_close`` — state-vector level assertion.

Compares a circuit's resulting state vector against an expected target
(either a complex array or a named state from
:mod:`qtest.assertions._state_library`).

Two comparison modes
--------------------
* ``global_phase=True`` (default) — compare via **fidelity**
  :math:`F = |\\langle\\psi|\\varphi\\rangle|^{2}`. The assertion succeeds
  when :math:`F \\ge 1 - \\text{tolerance}`. This is **invariant under a
  global phase** :math:`e^{i\\theta}` and is almost always the right notion
  of "the same state".
* ``global_phase=False`` — compare via raw Euclidean distance
  :math:`\\lVert\\psi - \\varphi\\rVert_{2}`. The assertion succeeds when
  this distance is strictly less than ``tolerance``. **Phase-sensitive**:
  two vectors differing only by an overall :math:`e^{i\\theta}` will fail.
"""

from __future__ import annotations

from typing import Any, Union

import numpy as np

from qtest.assertions._state_library import get_state
from qtest.backends import Backend
from qtest.backends.registry import get_backend
from qtest.config import get_config
from qtest.metrics import fidelity

# Accepted types for the *expected_state* parameter.
ExpectedState = Union[str, "list[complex]", "tuple[complex, ...]", np.ndarray]

# Floating-point slack absorbed by the fidelity comparison so that
# bit-equal state vectors don't fail at tolerance=0 due to ULP-level
# round-off (~2.2e-16 per double-precision op).
_FP_SLACK = 1e-12


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #


def assert_state_close(
    circuit: Any,
    expected_state: ExpectedState,
    tolerance: float = 1e-6,
    global_phase: bool = True,
    backend: Backend | None = None,
    msg: str | None = None,
) -> None:
    """Assert that *circuit*'s state vector equals *expected_state*.

    Parameters
    ----------
    circuit
        Backend-native circuit object (e.g. ``qiskit.QuantumCircuit``).
        Should contain no measurement / reset / classical operations
        — otherwise the backend will reject it when extracting the state.
    expected_state
        Either a name from :mod:`qtest.assertions._state_library`
        (e.g. ``"bell"``, ``"ghz_3"``), or a 1-D complex array / list of
        amplitudes with unit Euclidean norm.
    tolerance
        Maximum permitted infidelity (when ``global_phase=True``) or L2
        distance (when ``global_phase=False``). Defaults to ``1e-6``.
    global_phase
        Whether to ignore a global phase factor when comparing. Defaults
        to ``True`` (recommended).
    backend
        Backend to extract the state vector with. Defaults to the
        configured default backend; must satisfy
        :attr:`Backend.supports_statevector`.
    msg
        Optional prefix prepended to the assertion failure message.

    Raises
    ------
    ValueError
        For malformed input (unknown state name, wrong shape, non-unit
        norm, backend without state-vector support, etc.).
    AssertionError
        When the circuit's state vector diverges from *expected_state*
        beyond ``tolerance``.

    Examples
    --------
    >>> from qiskit import QuantumCircuit
    >>> qc = QuantumCircuit(2); qc.h(0); qc.cx(0, 1)
    >>> assert_state_close(qc, "bell")  # doctest: +SKIP
    """
    if not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool):
        raise ValueError(f"tolerance must be a real number, got {tolerance!r}")
    if tolerance < 0.0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance}")

    expected = _coerce_expected_state(expected_state)

    if backend is None:
        backend = get_backend(get_config().default_backend)
    if not backend.supports_statevector:
        raise ValueError(
            f"Backend {backend.name!r} does not support state-vector "
            "extraction (supports_statevector is False). Use a state-vector-"
            "capable backend such as QiskitBackend."
        )

    actual = np.asarray(backend.get_statevector(circuit), dtype=complex)
    if actual.ndim != 1:
        raise ValueError(f"Backend returned a non-1-D state vector of shape {actual.shape}")
    if actual.shape != expected.shape:
        raise ValueError(
            f"State vector shape mismatch: circuit produced {actual.shape}, "
            f"expected {expected.shape}. Check qubit counts."
        )

    if global_phase:
        fid = float(np.clip(fidelity(actual, expected), 0.0, 1.0))
        passed = fid >= 1.0 - tolerance - _FP_SLACK
        measured_value = fid
        threshold_desc = f">= {1.0 - tolerance:.6g}"
        metric_label = "fidelity"
    else:
        diff = float(np.linalg.norm(actual - expected))
        passed = diff <= tolerance + _FP_SLACK
        measured_value = diff
        threshold_desc = f"<= {tolerance:.6g}"
        metric_label = "L2 distance"

    if not passed:
        raise AssertionError(
            _format_failure_message(
                circuit=circuit,
                tolerance=tolerance,
                metric_label=metric_label,
                measured_value=measured_value,
                threshold_desc=threshold_desc,
                expected=expected,
                actual=actual,
                global_phase=global_phase,
                user_msg=msg,
            )
        )


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _coerce_expected_state(expected_state: ExpectedState) -> np.ndarray:
    """Convert *expected_state* into a validated 1-D complex unit vector."""
    if isinstance(expected_state, str):
        vec = get_state(expected_state)
    elif isinstance(expected_state, np.ndarray):
        vec = np.asarray(expected_state, dtype=complex)
    elif isinstance(expected_state, (list, tuple)):
        if not expected_state:
            raise ValueError("expected_state list/tuple must be non-empty")
        vec = np.asarray(expected_state, dtype=complex)
    else:
        raise ValueError(
            f"expected_state must be str | list | tuple | np.ndarray, "
            f"got {type(expected_state).__name__}"
        )

    if vec.ndim != 1:
        raise ValueError(f"expected_state must be 1-D, got shape {vec.shape}")
    if vec.size == 0:
        raise ValueError("expected_state must be non-empty")

    # Dimension must be a power of two — every n-qubit state has 2**n amplitudes.
    if vec.size & (vec.size - 1):
        raise ValueError(f"expected_state length must be a power of 2, got {vec.size}")

    norm = float(np.linalg.norm(vec))
    if not np.isclose(norm, 1.0, atol=1e-6):
        raise ValueError(f"expected_state must have unit Euclidean norm, got {norm:.6g}")
    return vec


def _format_amplitude(z: complex) -> str:
    """Render a complex amplitude with consistent significant digits."""
    real = z.real
    imag = z.imag
    return f"{real:+.4f} {imag:+.4f}j"


def _format_state(state: np.ndarray, *, max_entries: int = 16) -> str:
    """Render a state vector as ``|basis>: amp`` lines.

    For large vectors only the top-``max_entries`` amplitudes (by magnitude)
    are shown, followed by an ellipsis line.
    """
    n = state.size
    n_qubits = int(np.log2(n))

    if n <= max_entries:
        indices: list[int] = list(range(n))
    else:
        ranked = sorted(range(n), key=lambda i: -abs(state[i]))[:max_entries]
        indices = sorted(ranked)

    lines = [f"    |{i:0{n_qubits}b}>: {_format_amplitude(complex(state[i]))}" for i in indices]
    if n > max_entries:
        lines.append(f"    ... ({n - max_entries} more amplitudes hidden)")
    return "\n".join(lines)


def _circuit_summary(circuit: Any) -> str:
    name = getattr(circuit, "name", None)
    n_qubits = getattr(circuit, "num_qubits", None)
    if name and n_qubits is not None:
        return f"{name} ({n_qubits} qubits)"
    if name:
        return str(name)
    if n_qubits is not None:
        return f"<{type(circuit).__name__} num_qubits={n_qubits}>"
    return f"<{type(circuit).__name__}>"


def _format_failure_message(
    *,
    circuit: Any,
    tolerance: float,
    metric_label: str,
    measured_value: float,
    threshold_desc: str,
    expected: np.ndarray,
    actual: np.ndarray,
    global_phase: bool,
    user_msg: str | None,
) -> str:
    """Build the multi-line ``AssertionError`` payload."""
    lines: list[str] = []

    if user_msg:
        lines.append(user_msg)
        lines.append("")

    lines.append("State vector mismatch")
    lines.append("")
    lines.append(f"  Circuit: {_circuit_summary(circuit)}")
    lines.append(f"  Tolerance: {tolerance:g}")
    lines.append(f"  Global-phase ignored: {global_phase}")
    lines.append(f"  {metric_label}: {measured_value:.6f} (expected {threshold_desc})")
    lines.append("")
    lines.append("  Expected state:")
    lines.append(_format_state(expected))
    lines.append("")
    lines.append("  Measured state:")
    lines.append(_format_state(actual))

    return "\n".join(lines)
