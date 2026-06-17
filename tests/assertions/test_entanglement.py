"""Tests for ``assert_entangled`` / ``assert_separable``."""

from __future__ import annotations

import numpy as np
import pytest

from qtest import assert_entangled, assert_separable

pytest.importorskip("qiskit")

from qiskit import QuantumCircuit  # noqa: E402

_INV_SQRT2 = 1.0 / np.sqrt(2.0)


def _bell() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


def _product() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.h(0)  # |+> ⊗ |0>
    return qc


def test_bell_is_entangled() -> None:
    assert_entangled(_bell())


def test_product_is_separable() -> None:
    assert_separable(_product())


def test_bell_not_separable_raises() -> None:
    with pytest.raises(AssertionError, match="Entanglement assertion failed"):
        assert_separable(_bell())


def test_product_not_entangled_raises() -> None:
    with pytest.raises(AssertionError):
        assert_entangled(_product())


def test_entangled_with_explicit_cut() -> None:
    assert_entangled(_bell(), qubits=[0])


def test_ghz_entangled_each_qubit() -> None:
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    assert_entangled(qc, qubits=[0])
    assert_entangled(qc, qubits=[0, 1])


def test_accepts_raw_state_vector() -> None:
    bell = np.array([_INV_SQRT2, 0.0, 0.0, _INV_SQRT2])
    assert_entangled(bell)
    assert_separable(np.array([1.0, 0.0, 0.0, 0.0]))


def test_invalid_qubits_raises() -> None:
    with pytest.raises(ValueError):
        assert_entangled(_bell(), qubits=[5])


def test_single_qubit_default_cut_raises() -> None:
    qc = QuantumCircuit(1)
    qc.h(0)
    with pytest.raises(ValueError, match="at least 2 qubits"):
        assert_entangled(qc)


def test_invalid_raw_state_raises() -> None:
    with pytest.raises(ValueError, match="power-of-two"):
        assert_entangled(np.array([1.0, 0.0, 0.0]))


def test_failure_message_includes_user_msg() -> None:
    with pytest.raises(AssertionError, match="my custom note"):
        assert_separable(_bell(), msg="my custom note")
