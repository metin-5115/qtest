"""``assert_unitary`` — verify that an operation is unitary.

Accepts a matrix (``np.ndarray``), a Qiskit ``Gate`` (anything exposing
``to_matrix()``), or a quantum circuit (anything the configured backend can
extract a unitary from). Checks both :math:`U^{\\dagger} U \\approx I` and
:math:`U U^{\\dagger} \\approx I`, reporting the worst-case off-diagonal /
diagonal deviation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qtest.backends import Backend
from qtest.backends.registry import get_backend
from qtest.config import get_config

# Floating-point slack added to user-supplied tolerance to absorb 1-2 ULP
# round-off from the matrix multiplies (~size * machine epsilon).
_FP_SLACK = 1e-14


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #


def assert_unitary(
    operation: Any,
    tolerance: float = 1e-9,
    backend: Backend | None = None,
    msg: str | None = None,
) -> None:
    """Assert that *operation* represents a unitary operator.

    Parameters
    ----------
    operation
        One of:

        * ``np.ndarray`` — used directly as the candidate matrix.
        * Object with a ``to_matrix()`` method (e.g. ``qiskit.circuit.Gate``)
          — its matrix representation is used.
        * Anything else — passed to :meth:`Backend.get_unitary` (treated as
          a circuit).
    tolerance
        Maximum permitted entry-wise deviation
        :math:`\\max_{ij} |(U^\\dagger U)_{ij} - I_{ij}|`. Defaults to ``1e-9``.
    backend
        Backend used to extract a unitary from circuit-shaped inputs.
        Ignored for arrays and gate objects.
    msg
        Optional prefix prepended to the assertion failure message.

    Raises
    ------
    ValueError
        If the matrix is not 2-D, not square, or has zero size.
    AssertionError
        If ``operation`` deviates from unitarity beyond ``tolerance``.
    """
    if not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool):
        raise ValueError(f"tolerance must be a real number, got {tolerance!r}")
    if tolerance < 0.0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance}")

    matrix = _to_matrix(operation, backend)
    _validate_matrix_shape(matrix)

    d = matrix.shape[0]
    identity = np.eye(d, dtype=complex)
    left = matrix.conj().T @ matrix
    right = matrix @ matrix.conj().T

    left_dev = float(np.max(np.abs(left - identity)))
    right_dev = float(np.max(np.abs(right - identity)))
    deviation = max(left_dev, right_dev)

    if deviation <= tolerance + _FP_SLACK:
        return

    raise AssertionError(
        _format_failure(
            operation=operation,
            matrix=matrix,
            tolerance=tolerance,
            left_dev=left_dev,
            right_dev=right_dev,
            user_msg=msg,
        )
    )


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _to_matrix(operation: Any, backend: Backend | None) -> np.ndarray:
    """Coerce *operation* into a complex 2-D ``ndarray``."""
    if isinstance(operation, np.ndarray):
        return np.asarray(operation, dtype=complex)

    if hasattr(operation, "to_matrix") and callable(operation.to_matrix):
        mat = operation.to_matrix()
        return np.asarray(mat, dtype=complex)

    # Treat as a circuit and ask the backend for its unitary.
    if backend is None:
        backend = get_backend(get_config().default_backend)
    return np.asarray(backend.get_unitary(operation), dtype=complex)


def _validate_matrix_shape(matrix: np.ndarray) -> None:
    if matrix.ndim != 2:
        raise ValueError(f"unitary matrix must be 2-D, got array of shape {matrix.shape}")
    if matrix.size == 0:
        raise ValueError("unitary matrix must be non-empty")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"unitary matrix must be square, got shape {matrix.shape}")


def _operation_summary(operation: Any) -> str:
    name = getattr(operation, "name", None)
    if isinstance(operation, np.ndarray):
        return f"<ndarray shape={operation.shape}>"
    n_qubits = getattr(operation, "num_qubits", None)
    if name and n_qubits is not None:
        return f"{name} ({n_qubits} qubits)"
    if name:
        return str(name)
    return f"<{type(operation).__name__}>"


def _format_failure(
    *,
    operation: Any,
    matrix: np.ndarray,
    tolerance: float,
    left_dev: float,
    right_dev: float,
    user_msg: str | None,
) -> str:
    lines: list[str] = []
    if user_msg:
        lines.append(user_msg)
        lines.append("")

    lines.append("Operation is not unitary")
    lines.append("")
    lines.append(f"  Operation: {_operation_summary(operation)}")
    lines.append(f"  Matrix shape: {matrix.shape}")
    lines.append(f"  Tolerance: {tolerance:g}")
    lines.append(f"  max|U†U - I|: {left_dev:.3e}")
    lines.append(f"  max|UU† - I|: {right_dev:.3e}")
    lines.append("")
    lines.append(
        "  Frobenius norm of U†U - I: "
        f"{float(np.linalg.norm(matrix.conj().T @ matrix - np.eye(matrix.shape[0]))):.3e}"
    )
    return "\n".join(lines)
