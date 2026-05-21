"""Tests for :func:`qtest.assertions.assert_unitary`."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from qtest.assertions import assert_unitary
from qtest.assertions.unitary import (
    _operation_summary,
    _to_matrix,
    _validate_matrix_shape,
)
from qtest.backends.base import Backend

# --------------------------------------------------------------------------- #
# Mock backend                                                                #
# --------------------------------------------------------------------------- #


class _UnitaryBackend(Backend):
    """Mock backend returning a fixed unitary."""

    def __init__(self, unitary: np.ndarray, *, name: str = "mock_u") -> None:
        self._u = np.asarray(unitary, dtype=complex)
        self._name = name

    def run_circuit(
        self, circuit: Any, shots: int | None = None, seed: int | None = None
    ) -> dict[str, int]:
        raise NotImplementedError

    def get_statevector(self, circuit: Any) -> np.ndarray:
        raise NotImplementedError

    def get_unitary(self, circuit: Any) -> np.ndarray:
        return self._u.copy()

    @property
    def name(self) -> str:
        return self._name

    @property
    def supports_statevector(self) -> bool:
        return True


class _FakeCircuit:
    def __init__(self, name: str = "fake", num_qubits: int | None = None) -> None:
        self.name = name
        if num_qubits is not None:
            self.num_qubits = num_qubits


class _FakeGate:
    """Object exposing a ``to_matrix()`` method à la qiskit.circuit.Gate."""

    def __init__(self, matrix: np.ndarray, name: str = "fake_gate") -> None:
        self._m = np.asarray(matrix, dtype=complex)
        self.name = name

    def to_matrix(self) -> np.ndarray:
        return self._m.copy()


SQRT2 = math.sqrt(2.0)


# --------------------------------------------------------------------------- #
# Canonical unitaries (pass)                                                  #
# --------------------------------------------------------------------------- #


HADAMARD = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / SQRT2
PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
PAULI_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
PAULI_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
CNOT = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ],
    dtype=complex,
)


@pytest.mark.parametrize(
    "matrix",
    [HADAMARD, PAULI_X, PAULI_Y, PAULI_Z, CNOT, np.eye(8, dtype=complex)],
)
def test_known_unitaries_pass(matrix: np.ndarray) -> None:
    assert_unitary(matrix)


def test_random_unitary_passes() -> None:
    """A QR-decomposed Gaussian is Haar-random unitary."""
    rng = np.random.default_rng(seed=42)
    z = rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8))
    q, _ = np.linalg.qr(z)
    assert_unitary(q, tolerance=1e-10)


# --------------------------------------------------------------------------- #
# Non-unitary inputs (fail)                                                   #
# --------------------------------------------------------------------------- #


def test_zero_matrix_fails() -> None:
    with pytest.raises(AssertionError, match="not unitary"):
        assert_unitary(np.zeros((4, 4), dtype=complex))


def test_double_h_minus_identity_fails() -> None:
    """2*H is not unitary (norm too large)."""
    with pytest.raises(AssertionError):
        assert_unitary(2.0 * HADAMARD)


def test_partial_isometry_fails() -> None:
    """Projector |0><0| is not unitary."""
    proj = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
    with pytest.raises(AssertionError):
        assert_unitary(proj)


def test_failure_message_contains_diagnostics() -> None:
    with pytest.raises(AssertionError) as exc:
        assert_unitary(np.zeros((2, 2), dtype=complex), msg="X gate broken")
    text = str(exc.value)
    assert text.startswith("X gate broken")
    assert "not unitary" in text
    assert "max|U†U - I|" in text or "max|U" in text  # †
    assert "max|UU† - I|" in text or "max|UU" in text


# --------------------------------------------------------------------------- #
# Approximate unitarity (within / outside tolerance)                          #
# --------------------------------------------------------------------------- #


def test_near_unitary_within_tolerance_passes() -> None:
    perturbed = HADAMARD + 1e-10 * np.ones_like(HADAMARD)
    assert_unitary(perturbed, tolerance=1e-6)


def test_near_unitary_outside_tolerance_fails() -> None:
    perturbed = HADAMARD + 1e-3 * np.ones_like(HADAMARD)
    with pytest.raises(AssertionError):
        assert_unitary(perturbed, tolerance=1e-9)


# --------------------------------------------------------------------------- #
# Input types                                                                 #
# --------------------------------------------------------------------------- #


def test_ndarray_input() -> None:
    assert_unitary(HADAMARD)


def test_to_matrix_protocol() -> None:
    """Objects with a `.to_matrix()` method (Qiskit Gate-like) are accepted."""
    gate = _FakeGate(HADAMARD)
    assert_unitary(gate)


def test_circuit_input_uses_backend() -> None:
    backend = _UnitaryBackend(HADAMARD)
    assert_unitary(_FakeCircuit(), backend=backend)


def test_circuit_input_with_default_backend() -> None:
    """When backend=None, config.default_backend is looked up in the registry."""
    from qtest.backends import registry
    from qtest.config import configure, reset_config

    class _StubBackend(_UnitaryBackend):
        def __init__(self) -> None:
            super().__init__(HADAMARD)

    registry.register_backend("ut_mock", _StubBackend)
    try:
        configure(default_backend="ut_mock")
        assert_unitary(_FakeCircuit())
    finally:
        reset_config()
        registry._REGISTRY.pop("ut_mock", None)


# --------------------------------------------------------------------------- #
# Input validation                                                            #
# --------------------------------------------------------------------------- #


def test_non_square_matrix_raises() -> None:
    with pytest.raises(ValueError, match="square"):
        assert_unitary(np.zeros((2, 3), dtype=complex))


def test_one_dimensional_array_raises() -> None:
    with pytest.raises(ValueError, match="2-D"):
        assert_unitary(np.array([1.0, 0.0], dtype=complex))


def test_three_dimensional_array_raises() -> None:
    with pytest.raises(ValueError, match="2-D"):
        assert_unitary(np.zeros((2, 2, 2), dtype=complex))


def test_empty_array_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        assert_unitary(np.zeros((0, 0), dtype=complex))


def test_negative_tolerance_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        assert_unitary(HADAMARD, tolerance=-0.1)


def test_non_numeric_tolerance_raises() -> None:
    with pytest.raises(ValueError, match="real number"):
        assert_unitary(HADAMARD, tolerance="loose")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Integration with real Qiskit                                                #
# --------------------------------------------------------------------------- #


def test_qiskit_hadamard_circuit_passes() -> None:
    qiskit = pytest.importorskip("qiskit")
    qc = qiskit.QuantumCircuit(1)
    qc.h(0)
    assert_unitary(qc)


def test_qiskit_cnot_circuit_passes() -> None:
    qiskit = pytest.importorskip("qiskit")
    qc = qiskit.QuantumCircuit(2)
    qc.cx(0, 1)
    assert_unitary(qc)


def test_qiskit_gate_to_matrix_passes() -> None:
    pytest.importorskip("qiskit")
    from qiskit.circuit.library import HGate

    assert_unitary(HGate())


# --------------------------------------------------------------------------- #
# Helper unit tests                                                           #
# --------------------------------------------------------------------------- #


def test_to_matrix_ndarray() -> None:
    out = _to_matrix(HADAMARD, backend=None)
    assert np.allclose(out, HADAMARD)


def test_to_matrix_gate() -> None:
    out = _to_matrix(_FakeGate(PAULI_X), backend=None)
    assert np.allclose(out, PAULI_X)


def test_to_matrix_circuit_uses_backend() -> None:
    out = _to_matrix(_FakeCircuit(), backend=_UnitaryBackend(HADAMARD))
    assert np.allclose(out, HADAMARD)


def test_validate_shape_accepts_square() -> None:
    _validate_matrix_shape(HADAMARD)


def test_validate_shape_rejects_non_square() -> None:
    with pytest.raises(ValueError):
        _validate_matrix_shape(np.zeros((2, 3), dtype=complex))


def test_operation_summary_ndarray() -> None:
    assert "ndarray" in _operation_summary(HADAMARD)


def test_operation_summary_named_object() -> None:
    s = _operation_summary(_FakeCircuit(name="qc", num_qubits=3))
    assert "qc" in s
    assert "3" in s
