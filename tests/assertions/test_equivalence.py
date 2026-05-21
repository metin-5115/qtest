"""Tests for :func:`qtest.assertions.assert_circuit_equivalent`."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from qtest.assertions import assert_circuit_equivalent
from qtest.assertions.equivalence import (
    _auto_select_method,
    _generate_random_state,
    _qubit_count,
)
from qtest.backends.base import Backend

# --------------------------------------------------------------------------- #
# Mock infrastructure                                                         #
# --------------------------------------------------------------------------- #


SQRT2 = math.sqrt(2.0)
HADAMARD = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / SQRT2
PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
IDENTITY_2 = np.eye(2, dtype=complex)


class _UnitaryBackend(Backend):
    """Mock backend that returns a pre-set unitary keyed by id(circuit)."""

    def __init__(self, mapping: dict[int, np.ndarray]) -> None:
        self._mapping = mapping

    def run_circuit(
        self, circuit: Any, shots: int | None = None, seed: int | None = None
    ) -> dict[str, int]:
        raise NotImplementedError

    def get_statevector(self, circuit: Any) -> np.ndarray:
        raise NotImplementedError

    def get_unitary(self, circuit: Any) -> np.ndarray:
        return self._mapping[id(circuit)].copy()

    @property
    def name(self) -> str:
        return "mock_eq"

    @property
    def supports_statevector(self) -> bool:
        return True


class _FakeCircuit:
    def __init__(self, name: str = "fake", num_qubits: int = 1) -> None:
        self.name = name
        self.num_qubits = num_qubits


def _make_pair(
    u_a: np.ndarray, u_b: np.ndarray
) -> tuple[_FakeCircuit, _FakeCircuit, _UnitaryBackend]:
    """Build two fake circuits and a backend that maps each to a unitary."""
    dim = u_a.shape[0]
    n_qubits = int(round(math.log2(dim)))
    a = _FakeCircuit(name="A", num_qubits=n_qubits)
    b = _FakeCircuit(name="B", num_qubits=n_qubits)
    backend = _UnitaryBackend({id(a): u_a, id(b): u_b})
    return a, b, backend


# --------------------------------------------------------------------------- #
# Happy paths                                                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("method", ["unitary", "hilbert_schmidt", "random_sampling"])
def test_identical_circuits_pass_all_methods(method: str) -> None:
    a, b, backend = _make_pair(HADAMARD, HADAMARD)
    assert_circuit_equivalent(a, b, method=method, backend=backend, seed=0)


def test_h_squared_equivalent_to_identity() -> None:
    """H @ H = I — the canonical 'algebraically equivalent' example."""
    a, b, backend = _make_pair(HADAMARD @ HADAMARD, IDENTITY_2)
    assert_circuit_equivalent(a, b, backend=backend)


def test_global_phase_is_ignored_by_unitary_method() -> None:
    """Process fidelity is phase-insensitive: U vs e^{iθ}U should pass."""
    a, b, backend = _make_pair(HADAMARD, np.exp(1.234j) * HADAMARD)
    assert_circuit_equivalent(a, b, method="unitary", backend=backend)


def test_global_phase_rejected_by_hilbert_schmidt() -> None:
    """HS distance is phase-sensitive: a global phase makes the circuits differ."""
    a, b, backend = _make_pair(HADAMARD, np.exp(1.234j) * HADAMARD)
    with pytest.raises(AssertionError):
        assert_circuit_equivalent(a, b, method="hilbert_schmidt", backend=backend)


# --------------------------------------------------------------------------- #
# Failure paths                                                               #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("method", ["unitary", "hilbert_schmidt", "random_sampling"])
def test_different_circuits_fail(method: str) -> None:
    """Hadamard vs X — clearly different unitaries."""
    a, b, backend = _make_pair(HADAMARD, PAULI_X)
    with pytest.raises(AssertionError, match="not equivalent"):
        assert_circuit_equivalent(a, b, method=method, backend=backend, seed=0)


def test_failure_message_contains_diagnostics() -> None:
    a, b, backend = _make_pair(HADAMARD, PAULI_X)
    with pytest.raises(AssertionError) as exc:
        assert_circuit_equivalent(a, b, method="unitary", backend=backend, msg="H vs X should fail")
    text = str(exc.value)
    assert text.startswith("H vs X should fail")
    assert "Circuit A: A" in text
    assert "Circuit B: B" in text
    assert "Method: unitary" in text
    assert "Measured process infidelity" in text


def test_failure_message_for_random_sampling_includes_samples() -> None:
    a, b, backend = _make_pair(HADAMARD, PAULI_X)
    with pytest.raises(AssertionError) as exc:
        assert_circuit_equivalent(
            a, b, method="random_sampling", n_samples=25, backend=backend, seed=7
        )
    text = str(exc.value)
    assert "Samples: 25" in text
    assert "mean infidelity" in text


# --------------------------------------------------------------------------- #
# Method auto-selection                                                       #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "n_qubits, expected",
    [
        (1, "unitary"),
        (4, "unitary"),
        (8, "unitary"),
        (9, "hilbert_schmidt"),
        (12, "hilbert_schmidt"),
        (15, "hilbert_schmidt"),
        (16, "random_sampling"),
        (20, "random_sampling"),
    ],
)
def test_auto_select_method(n_qubits: int, expected: str) -> None:
    assert _auto_select_method(n_qubits) == expected


def test_failure_message_marks_auto_selection() -> None:
    a, b, backend = _make_pair(HADAMARD, PAULI_X)
    with pytest.raises(AssertionError) as exc:
        assert_circuit_equivalent(a, b, backend=backend)
    assert "(auto)" in str(exc.value)


def test_failure_message_no_auto_marker_for_explicit_method() -> None:
    a, b, backend = _make_pair(HADAMARD, PAULI_X)
    with pytest.raises(AssertionError) as exc:
        assert_circuit_equivalent(a, b, method="unitary", backend=backend)
    assert "(auto)" not in str(exc.value)


# --------------------------------------------------------------------------- #
# Random sampling                                                             #
# --------------------------------------------------------------------------- #


def test_random_sampling_seed_reproducible() -> None:
    """Same seed -> identical accept/reject decision across runs."""
    a, b, backend = _make_pair(HADAMARD, PAULI_X)
    err1 = err2 = None
    try:
        assert_circuit_equivalent(
            a, b, method="random_sampling", n_samples=20, backend=backend, seed=42
        )
    except AssertionError as e:
        err1 = str(e)
    try:
        assert_circuit_equivalent(
            a, b, method="random_sampling", n_samples=20, backend=backend, seed=42
        )
    except AssertionError as e:
        err2 = str(e)
    assert err1 is not None
    assert err1 == err2


def test_random_sampling_different_seeds_give_different_estimates() -> None:
    """Sanity: two seeds shouldn't produce the exact same numeric infidelity."""
    a, b, backend = _make_pair(HADAMARD, PAULI_X)
    msg_a = msg_b = ""
    try:
        assert_circuit_equivalent(
            a, b, method="random_sampling", n_samples=20, backend=backend, seed=1
        )
    except AssertionError as e:
        msg_a = str(e)
    try:
        assert_circuit_equivalent(
            a, b, method="random_sampling", n_samples=20, backend=backend, seed=2
        )
    except AssertionError as e:
        msg_b = str(e)
    # The mean-infidelity float embedded in the message should differ.
    assert msg_a and msg_b
    assert msg_a != msg_b


# --------------------------------------------------------------------------- #
# Input validation                                                            #
# --------------------------------------------------------------------------- #


def test_qubit_count_mismatch_raises() -> None:
    a = _FakeCircuit(num_qubits=1)
    b = _FakeCircuit(num_qubits=2)
    with pytest.raises(ValueError, match="Qubit count mismatch"):
        assert_circuit_equivalent(a, b)


def test_unknown_method_raises() -> None:
    a, b, backend = _make_pair(HADAMARD, HADAMARD)
    with pytest.raises(ValueError, match="Unknown method"):
        assert_circuit_equivalent(a, b, method="bogus", backend=backend)


def test_negative_tolerance_raises() -> None:
    a, b, backend = _make_pair(HADAMARD, HADAMARD)
    with pytest.raises(ValueError, match="non-negative"):
        assert_circuit_equivalent(a, b, tolerance=-0.1, backend=backend)


@pytest.mark.parametrize("bad", [0, -1, -100])
def test_non_positive_n_samples_raises(bad: int) -> None:
    a, b, backend = _make_pair(HADAMARD, HADAMARD)
    with pytest.raises(ValueError, match="n_samples"):
        assert_circuit_equivalent(a, b, method="random_sampling", n_samples=bad, backend=backend)


def test_circuit_without_num_qubits_raises() -> None:
    class _NoQubits:
        name = "nope"

    with pytest.raises(ValueError, match="num_qubits"):
        assert_circuit_equivalent(_NoQubits(), _NoQubits())


# --------------------------------------------------------------------------- #
# _generate_random_state                                                      #
# --------------------------------------------------------------------------- #


def test_random_state_is_unit_norm() -> None:
    for n in range(1, 5):
        psi = _generate_random_state(n, seed=0)
        assert math.isclose(float(np.linalg.norm(psi)), 1.0, abs_tol=1e-10), n


def test_random_state_has_correct_dimension() -> None:
    for n in range(1, 6):
        assert _generate_random_state(n, seed=1).size == 1 << n


def test_random_state_seed_reproducible() -> None:
    a = _generate_random_state(3, seed=123)
    b = _generate_random_state(3, seed=123)
    assert np.allclose(a, b)


def test_random_state_different_seeds_differ() -> None:
    a = _generate_random_state(3, seed=1)
    b = _generate_random_state(3, seed=2)
    assert not np.allclose(a, b)


def test_random_state_invalid_qubits_raises() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        _generate_random_state(0)


def test_qubit_count_helper() -> None:
    assert _qubit_count(_FakeCircuit(num_qubits=4)) == 4


def test_qubit_count_helper_missing_attr_raises() -> None:
    class _NoNumQubits:
        pass

    with pytest.raises(ValueError, match="num_qubits"):
        _qubit_count(_NoNumQubits())


# --------------------------------------------------------------------------- #
# Integration with real Qiskit                                                #
# --------------------------------------------------------------------------- #


def test_qiskit_identical_circuits() -> None:
    qiskit = pytest.importorskip("qiskit")
    qc1 = qiskit.QuantumCircuit(2)
    qc1.h(0)
    qc1.cx(0, 1)
    qc2 = qiskit.QuantumCircuit(2)
    qc2.h(0)
    qc2.cx(0, 1)
    assert_circuit_equivalent(qc1, qc2)


def test_qiskit_h_squared_equals_identity() -> None:
    qiskit = pytest.importorskip("qiskit")
    qc_hh = qiskit.QuantumCircuit(1)
    qc_hh.h(0)
    qc_hh.h(0)
    qc_id = qiskit.QuantumCircuit(1)
    assert_circuit_equivalent(qc_hh, qc_id)


def test_qiskit_different_circuits_fail() -> None:
    qiskit = pytest.importorskip("qiskit")
    qc_h = qiskit.QuantumCircuit(1)
    qc_h.h(0)
    qc_x = qiskit.QuantumCircuit(1)
    qc_x.x(0)
    with pytest.raises(AssertionError):
        assert_circuit_equivalent(qc_h, qc_x)
