"""Tests for :class:`qtest.backends.CirqBackend` (require Cirq)."""

from __future__ import annotations

import numpy as np
import pytest

import qtest
from qtest.backends import CirqBackend, get_backend
from qtest.noise import depolarizing

cirq = pytest.importorskip("cirq")


def _bell() -> cirq.Circuit:
    q = cirq.LineQubit.range(2)
    return cirq.Circuit([cirq.H(q[0]), cirq.CNOT(q[0], q[1])])


def test_metadata() -> None:
    b = CirqBackend()
    assert b.name == "cirq:simulator"
    assert b.supports_statevector is True


def test_registered_in_registry() -> None:
    assert isinstance(get_backend("cirq"), CirqBackend)


def test_constructor_rejects_bad_shots() -> None:
    with pytest.raises(ValueError):
        CirqBackend(shots=0)


def test_statevector_bell() -> None:
    sv = CirqBackend().get_statevector(_bell())
    expected = np.array([1, 0, 0, 1]) / np.sqrt(2)
    np.testing.assert_allclose(sv, expected, atol=1e-7)


def test_unitary_shape() -> None:
    assert CirqBackend().get_unitary(_bell()).shape == (4, 4)


def test_run_circuit_counts_sum_to_shots() -> None:
    counts = CirqBackend().run_circuit(_bell(), shots=2048, seed=1)
    assert sum(counts.values()) == 2048
    assert set(counts) <= {"00", "11"}


def test_run_circuit_seed_reproducible() -> None:
    b = CirqBackend()
    assert b.run_circuit(_bell(), shots=512, seed=7) == b.run_circuit(_bell(), shots=512, seed=7)


def test_noise_model_rejected() -> None:
    with pytest.raises(NotImplementedError):
        CirqBackend().run_circuit(_bell(), shots=10, noise_model=depolarizing(0.1))


def test_none_circuit_raises() -> None:
    with pytest.raises(ValueError):
        CirqBackend().get_statevector(None)


def test_integration_with_assertions() -> None:
    b = CirqBackend()
    qtest.assert_state_close(_bell(), "bell", backend=b)
    qtest.assert_distribution_close(
        _bell(), {"00": 0.5, "11": 0.5}, shots=4096, tolerance=0.05, seed=1, backend=b
    )
