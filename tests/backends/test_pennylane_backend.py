"""Tests for :class:`qtest.backends.PennyLaneBackend` (require PennyLane)."""

from __future__ import annotations

import numpy as np
import pytest

import qtest
from qtest.backends import PennyLaneBackend, get_backend
from qtest.noise import depolarizing

qml = pytest.importorskip("pennylane")


def _bell_tape() -> qml.tape.QuantumScript:
    return qml.tape.QuantumScript([qml.Hadamard(0), qml.CNOT([0, 1])])


def test_metadata() -> None:
    b = PennyLaneBackend()
    assert b.name == "pennylane:default.qubit"
    assert b.supports_statevector is True


def test_registered_in_registry() -> None:
    assert isinstance(get_backend("pennylane"), PennyLaneBackend)


def test_constructor_rejects_bad_shots() -> None:
    with pytest.raises(ValueError):
        PennyLaneBackend(shots=-5)


def test_statevector_bell() -> None:
    sv = PennyLaneBackend().get_statevector(_bell_tape())
    expected = np.array([1, 0, 0, 1]) / np.sqrt(2)
    np.testing.assert_allclose(sv, expected, atol=1e-7)


def test_unitary_shape() -> None:
    assert PennyLaneBackend().get_unitary(_bell_tape()).shape == (4, 4)


def test_run_circuit_counts_sum_to_shots() -> None:
    counts = PennyLaneBackend().run_circuit(_bell_tape(), shots=2048, seed=1)
    assert sum(counts.values()) == 2048
    assert set(counts) <= {"00", "11"}


def test_run_circuit_seed_reproducible() -> None:
    b = PennyLaneBackend()
    a1 = b.run_circuit(_bell_tape(), shots=512, seed=7)
    a2 = b.run_circuit(_bell_tape(), shots=512, seed=7)
    assert a1 == a2


def test_noise_model_rejected() -> None:
    with pytest.raises(NotImplementedError):
        PennyLaneBackend().run_circuit(_bell_tape(), shots=10, noise_model=depolarizing(0.1))


def test_non_tape_input_raises() -> None:
    with pytest.raises(ValueError, match="QuantumScript"):
        PennyLaneBackend().get_statevector(object())


def test_integration_with_assertions() -> None:
    b = PennyLaneBackend()
    qtest.assert_state_close(_bell_tape(), "bell", backend=b)
    qtest.assert_distribution_close(
        _bell_tape(), {"00": 0.5, "11": 0.5}, shots=4096, tolerance=0.05, seed=1, backend=b
    )
