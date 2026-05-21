"""Tests for :class:`qtest.backends.qiskit_backend.QiskitBackend`."""

from __future__ import annotations

import math

import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit")
from qiskit import QuantumCircuit  # noqa: E402

from qtest.backends.qiskit_backend import QiskitBackend  # noqa: E402

# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture
def backend() -> QiskitBackend:
    return QiskitBackend(shots=1024)


@pytest.fixture
def hadamard_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    return qc


@pytest.fixture
def bell_circuit() -> QuantumCircuit:
    """Bell state preparation (no measurement, suitable for statevector)."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


# --------------------------------------------------------------------------- #
# Metadata                                                                    #
# --------------------------------------------------------------------------- #


def test_name_contains_qiskit() -> None:
    assert "qiskit" in QiskitBackend().name.lower()


def test_supports_statevector_is_true() -> None:
    assert QiskitBackend().supports_statevector is True


def test_custom_simulator_name_in_name() -> None:
    b = QiskitBackend(simulator_name="my_sim")
    assert "my_sim" in b.name


# --------------------------------------------------------------------------- #
# Constructor validation                                                      #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad_shots", [0, -1, -1000])
def test_init_rejects_non_positive_shots(bad_shots: int) -> None:
    with pytest.raises(ValueError):
        QiskitBackend(shots=bad_shots)


def test_init_rejects_non_int_shots() -> None:
    with pytest.raises(ValueError):
        QiskitBackend(shots=1024.0)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_level", [-1, 4, 99])
def test_init_rejects_bad_optimization_level(bad_level: int) -> None:
    with pytest.raises(ValueError):
        QiskitBackend(optimization_level=bad_level)


def test_init_rejects_empty_simulator_name() -> None:
    with pytest.raises(ValueError):
        QiskitBackend(simulator_name="")


# --------------------------------------------------------------------------- #
# run_circuit                                                                 #
# --------------------------------------------------------------------------- #


def test_run_circuit_hadamard_roughly_50_50(
    backend: QiskitBackend, hadamard_circuit: QuantumCircuit
) -> None:
    counts = backend.run_circuit(hadamard_circuit, shots=4000, seed=42)
    assert isinstance(counts, dict)
    assert sum(counts.values()) == 4000
    assert set(counts.keys()) == {"0", "1"}
    for outcome in ("0", "1"):
        assert abs(counts[outcome] / 4000 - 0.5) < 0.05


@pytest.mark.parametrize("shots", [100, 500, 2000, 5000])
def test_run_circuit_shots_parameter_respected(
    backend: QiskitBackend, hadamard_circuit: QuantumCircuit, shots: int
) -> None:
    counts = backend.run_circuit(hadamard_circuit, shots=shots, seed=1)
    assert sum(counts.values()) == shots


def test_run_circuit_seed_is_reproducible(
    backend: QiskitBackend, hadamard_circuit: QuantumCircuit
) -> None:
    c1 = backend.run_circuit(hadamard_circuit, shots=1000, seed=12345)
    c2 = backend.run_circuit(hadamard_circuit, shots=1000, seed=12345)
    assert c1 == c2


def test_run_circuit_default_shots_used_when_omitted(
    hadamard_circuit: QuantumCircuit,
) -> None:
    b = QiskitBackend(shots=512)
    counts = b.run_circuit(hadamard_circuit, seed=0)
    assert sum(counts.values()) == 512


def test_run_circuit_single_qubit_x_gate() -> None:
    """X gate on |0> should yield bitstring '1' deterministically."""
    qc = QuantumCircuit(1, 1)
    qc.x(0)
    qc.measure(0, 0)
    counts = QiskitBackend(shots=300).run_circuit(qc, seed=0)
    assert counts == {"1": 300}


def test_run_circuit_none_raises(backend: QiskitBackend) -> None:
    with pytest.raises(ValueError):
        backend.run_circuit(None, shots=100)


@pytest.mark.parametrize("bad_shots", [0, -1, -100])
def test_run_circuit_non_positive_shots_raises(
    backend: QiskitBackend, hadamard_circuit: QuantumCircuit, bad_shots: int
) -> None:
    with pytest.raises(ValueError):
        backend.run_circuit(hadamard_circuit, shots=bad_shots)


def test_run_circuit_non_int_seed_raises(
    backend: QiskitBackend, hadamard_circuit: QuantumCircuit
) -> None:
    with pytest.raises(ValueError):
        backend.run_circuit(hadamard_circuit, shots=100, seed=3.14)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# get_statevector                                                             #
# --------------------------------------------------------------------------- #


def test_get_statevector_bell_state(backend: QiskitBackend, bell_circuit: QuantumCircuit) -> None:
    sv = backend.get_statevector(bell_circuit)
    # Qiskit little-endian: index 0 = |00>, index 3 = |11>
    expected = np.array([1.0, 0.0, 0.0, 1.0], dtype=complex) / math.sqrt(2.0)
    assert sv.dtype == complex
    assert sv.shape == (4,)
    assert np.allclose(sv, expected, atol=1e-10)


def test_get_statevector_zero_state(backend: QiskitBackend) -> None:
    qc = QuantumCircuit(1)  # no gates -> |0>
    sv = backend.get_statevector(qc)
    assert np.allclose(sv, np.array([1.0, 0.0], dtype=complex))


def test_get_statevector_plus_state(backend: QiskitBackend) -> None:
    qc = QuantumCircuit(1)
    qc.h(0)
    sv = backend.get_statevector(qc)
    expected = np.array([1.0, 1.0], dtype=complex) / math.sqrt(2.0)
    assert np.allclose(sv, expected, atol=1e-10)


def test_get_statevector_none_raises(backend: QiskitBackend) -> None:
    with pytest.raises(ValueError):
        backend.get_statevector(None)


# --------------------------------------------------------------------------- #
# get_unitary                                                                 #
# --------------------------------------------------------------------------- #


def test_get_unitary_hadamard(backend: QiskitBackend) -> None:
    qc = QuantumCircuit(1)
    qc.h(0)
    u = backend.get_unitary(qc)
    expected = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2.0)
    assert u.shape == (2, 2)
    assert np.allclose(u, expected, atol=1e-10)


def test_get_unitary_identity_for_empty_circuit(backend: QiskitBackend) -> None:
    qc = QuantumCircuit(2)  # 0 gates -> identity
    u = backend.get_unitary(qc)
    assert np.allclose(u, np.eye(4, dtype=complex))


def test_get_unitary_cnot(backend: QiskitBackend) -> None:
    """CNOT(control=0, target=1) in Qiskit's little-endian convention."""
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    u = backend.get_unitary(qc)
    expected = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
        ],
        dtype=complex,
    )
    assert np.allclose(u, expected, atol=1e-10)


def test_get_unitary_none_raises(backend: QiskitBackend) -> None:
    with pytest.raises(ValueError):
        backend.get_unitary(None)
