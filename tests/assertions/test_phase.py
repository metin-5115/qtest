"""Tests for ``assert_phase``."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qtest import assert_phase

pytest.importorskip("qiskit")

from qiskit import QuantumCircuit  # noqa: E402


def test_s_gate_imparts_pi_over_2() -> None:
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.s(0)  # |+> -> (|0> + i|1>)/sqrt(2)
    assert_phase(qc, 0, 1, math.pi / 2, tolerance=1e-6)


def test_t_gate_imparts_pi_over_4() -> None:
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.t(0)
    assert_phase(qc, 0, 1, math.pi / 4, tolerance=1e-6)


def test_phase_accepts_bitstring_indices() -> None:
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.s(0)
    assert_phase(qc, "0", "1", math.pi / 2, tolerance=1e-6)


def test_phase_modulo_2pi() -> None:
    # expected given as pi/2 + 2pi must still match.
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.s(0)
    assert_phase(qc, 0, 1, math.pi / 2 + 2 * math.pi, tolerance=1e-6)


def test_wrong_phase_raises() -> None:
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.s(0)
    with pytest.raises(AssertionError, match="Relative phase mismatch"):
        assert_phase(qc, 0, 1, 0.0, tolerance=1e-6)


def test_zero_amplitude_raises() -> None:
    qc = QuantumCircuit(1)  # |0>: amplitude of |1> is 0
    with pytest.raises(ValueError, match="undefined"):
        assert_phase(qc, 0, 1, 0.0)


def test_raw_state_vector() -> None:
    state = np.array([1.0, 1j]) / np.sqrt(2)
    assert_phase(state, 0, 1, math.pi / 2, tolerance=1e-9)


def test_out_of_range_index_raises() -> None:
    with pytest.raises(ValueError, match="out of range"):
        assert_phase(np.array([1.0, 0.0]), 0, 5, 0.0)


def test_invalid_bitstring_raises() -> None:
    state = np.array([1.0, 1j]) / np.sqrt(2)
    with pytest.raises(ValueError, match="only 0/1"):
        assert_phase(state, "0", "2", 0.0)


def test_negative_tolerance_raises() -> None:
    state = np.array([1.0, 1j]) / np.sqrt(2)
    with pytest.raises(ValueError, match="non-negative"):
        assert_phase(state, 0, 1, 0.0, tolerance=-1.0)


def test_non_power_of_two_state_raises() -> None:
    with pytest.raises(ValueError, match="power-of-two"):
        assert_phase(np.array([1.0, 0.0, 0.0]), 0, 1, 0.0)
