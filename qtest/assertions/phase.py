"""``assert_phase`` — relative-phase assertion between two basis amplitudes.

Global phase is physically irrelevant, but *relative* phases between basis
states are the whole point of many circuits (a ``Z``/``S``/``T`` rotation, a
phase-kickback, a QFT). :func:`assert_phase` checks that the phase of one
computational-basis amplitude relative to another matches a target angle,
modulo :math:`2\\pi`.
"""

from __future__ import annotations

import math
from typing import Any, Union

import numpy as np

from qtest.backends import Backend
from qtest.backends.registry import get_backend
from qtest.config import get_config

BasisState = Union[int, str]

# Amplitudes smaller than this have an ill-defined phase.
_AMP_FLOOR = 1e-9


def assert_phase(
    circuit: Any,
    index_a: BasisState,
    index_b: BasisState,
    expected_phase: float,
    tolerance: float = 1e-6,
    backend: Backend | None = None,
    msg: str | None = None,
) -> None:
    r"""Assert the relative phase ``arg(amp_b / amp_a)`` equals *expected_phase*.

    Parameters
    ----------
    circuit
        A backend-native circuit (no measurements) or a 1-D state-vector array.
    index_a, index_b
        The two computational basis states, given either as integer indices
        (``0`` … ``2**n - 1``) or as bitstrings (e.g. ``"01"``). ``amp_a`` is
        the reference.
    expected_phase
        Target relative phase in radians. Compared modulo :math:`2\pi`.
    tolerance
        Maximum permitted angular deviation in radians.
    backend
        Backend used to extract the state vector.
    msg
        Optional prefix prepended to the assertion failure message.

    Raises
    ------
    ValueError
        For malformed input, or when either reference amplitude is ~0 (phase
        undefined).
    AssertionError
        When the relative phase deviates from *expected_phase* beyond
        *tolerance*.
    """
    if not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool) or tolerance < 0.0:
        raise ValueError(f"tolerance must be a non-negative real number, got {tolerance!r}")

    state = _get_state_vector(circuit, backend)
    dim = state.size
    ia = _coerce_index(index_a, dim, "index_a")
    ib = _coerce_index(index_b, dim, "index_b")

    amp_a = complex(state[ia])
    amp_b = complex(state[ib])
    if abs(amp_a) < _AMP_FLOOR or abs(amp_b) < _AMP_FLOOR:
        raise ValueError(
            f"relative phase is undefined: |amp[{ia}]|={abs(amp_a):.3g}, "
            f"|amp[{ib}]|={abs(amp_b):.3g} (one is ~0)."
        )

    measured_phase = math.atan2((amp_b / amp_a).imag, (amp_b / amp_a).real)
    deviation = _angular_distance(measured_phase, float(expected_phase))
    if deviation <= tolerance:
        return

    raise AssertionError(
        _format_failure(
            circuit=circuit,
            ia=ia,
            ib=ib,
            measured=measured_phase,
            expected=float(expected_phase),
            deviation=deviation,
            tolerance=tolerance,
            user_msg=msg,
        )
    )


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _get_state_vector(circuit: Any, backend: Backend | None) -> np.ndarray:
    if isinstance(circuit, (np.ndarray, list, tuple)):
        vec = np.asarray(circuit, dtype=complex)
    else:
        if backend is None:
            backend = get_backend(get_config().default_backend)
        vec = np.asarray(backend.get_statevector(circuit), dtype=complex)
    if vec.ndim != 1 or vec.size < 2 or (vec.size & (vec.size - 1)):
        raise ValueError(
            f"state vector must be 1-D with a power-of-two length ≥ 2, got {vec.shape}"
        )
    return vec


def _coerce_index(index: BasisState, dim: int, name: str) -> int:
    """Convert an int index or bitstring into a validated integer index."""
    if isinstance(index, bool):
        raise ValueError(f"{name} must be an int index or bitstring, got bool")
    if isinstance(index, str):
        s = index.replace(" ", "")
        if not s or any(c not in "01" for c in s):
            raise ValueError(f"{name} bitstring must contain only 0/1, got {index!r}")
        value = int(s, 2)
    elif isinstance(index, int):
        value = index
    else:
        raise ValueError(f"{name} must be an int index or bitstring, got {type(index).__name__}")
    if not (0 <= value < dim):
        raise ValueError(f"{name}={value} is out of range for a dim-{dim} state.")
    return value


def _angular_distance(a: float, b: float) -> float:
    """Smallest absolute angular difference between *a* and *b*, modulo 2π."""
    diff = (a - b) % (2.0 * math.pi)
    return min(diff, 2.0 * math.pi - diff)


def _circuit_summary(circuit: Any) -> str:
    if isinstance(circuit, np.ndarray):
        return f"<state vector dim={circuit.size}>"
    name = getattr(circuit, "name", None)
    return str(name) if name else f"<{type(circuit).__name__}>"


def _format_failure(
    *,
    circuit: Any,
    ia: int,
    ib: int,
    measured: float,
    expected: float,
    deviation: float,
    tolerance: float,
    user_msg: str | None,
) -> str:
    lines: list[str] = []
    if user_msg:
        lines.extend([user_msg, ""])
    lines.append("Relative phase mismatch")
    lines.append("")
    lines.append(f"  Circuit: {_circuit_summary(circuit)}")
    lines.append(f"  Reference amplitude: index {ia}")
    lines.append(f"  Target amplitude:    index {ib}")
    lines.append(f"  Measured phase: {measured:+.6f} rad")
    lines.append(f"  Expected phase: {expected:+.6f} rad")
    lines.append(f"  Angular deviation: {deviation:.3e} rad (tolerance {tolerance:g})")
    return "\n".join(lines)
