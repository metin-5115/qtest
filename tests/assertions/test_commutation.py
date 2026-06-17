"""Tests for ``assert_commutes``."""

from __future__ import annotations

import numpy as np
import pytest

from qtest import assert_commutes

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
ID = np.eye(2, dtype=complex)


def test_identity_commutes_with_everything() -> None:
    assert_commutes(ID, X)
    assert_commutes(X, ID)


def test_pauli_commutes_with_itself() -> None:
    assert_commutes(X, X)
    assert_commutes(Z, Z)


def test_distinct_paulis_anticommute() -> None:
    assert_commutes(X, Z, anti=True)
    assert_commutes(X, Y, anti=True)
    assert_commutes(Y, Z, anti=True)


def test_distinct_paulis_do_not_commute() -> None:
    with pytest.raises(AssertionError, match="do not commute"):
        assert_commutes(X, Z)


def test_anticommute_failure_message() -> None:
    with pytest.raises(AssertionError, match="do not anticommute"):
        assert_commutes(X, X, anti=True)


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same shape"):
        assert_commutes(X, np.eye(4, dtype=complex))


def test_non_square_raises() -> None:
    with pytest.raises(ValueError):
        assert_commutes(np.zeros((2, 3)), np.zeros((2, 3)))


def test_negative_tolerance_raises() -> None:
    with pytest.raises(ValueError):
        assert_commutes(X, X, tolerance=-1.0)


def test_accepts_gate_with_to_matrix() -> None:
    pytest.importorskip("qiskit")
    from qiskit.circuit.library import XGate, ZGate

    assert_commutes(XGate(), XGate())
    assert_commutes(XGate(), ZGate(), anti=True)
