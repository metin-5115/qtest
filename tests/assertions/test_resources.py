"""Tests for the resource / cost assertions."""

from __future__ import annotations

import pytest
from qiskit import QuantumCircuit

from qtest import (
    assert_max_depth,
    assert_max_gate_count,
    assert_max_t_count,
    assert_max_two_qubit_count,
)
from qtest.backends import QiskitBackend


def _sample_circuit() -> QuantumCircuit:
    """A 3-qubit circuit: depth 3, 5 gates, 2 two-qubit gates, 1 T gate."""
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.t(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.h(2)
    return qc


# --------------------------------------------------------------------------- #
# get_resources                                                               #
# --------------------------------------------------------------------------- #


def test_resources_metrics_are_correct() -> None:
    res = QiskitBackend().get_resources(_sample_circuit())
    assert res.num_qubits == 3
    assert res.size == 5
    assert res.two_qubit_count == 2
    assert res.multi_qubit_count == 2
    assert res.t_count == 1
    assert res.gate_counts["cx"] == 2
    assert res.gate_counts["h"] == 2


def test_barrier_and_measure_excluded_from_two_qubit_count() -> None:
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.barrier()  # spans 2 qubits but is not a gate
    qc.measure_all()
    res = QiskitBackend().get_resources(qc)
    assert res.two_qubit_count == 1


# --------------------------------------------------------------------------- #
# assert_max_depth                                                            #
# --------------------------------------------------------------------------- #


def test_max_depth_passes_at_and_above_limit() -> None:
    qc = _sample_circuit()
    assert_max_depth(qc, qc.depth())
    assert_max_depth(qc, qc.depth() + 1)


def test_max_depth_fails_when_exceeded() -> None:
    with pytest.raises(AssertionError, match="depth exceeds"):
        assert_max_depth(_sample_circuit(), 1)


# --------------------------------------------------------------------------- #
# assert_max_gate_count                                                       #
# --------------------------------------------------------------------------- #


def test_max_gate_count_total() -> None:
    assert_max_gate_count(_sample_circuit(), 5)
    with pytest.raises(AssertionError, match="gate count exceeds"):
        assert_max_gate_count(_sample_circuit(), 4)


def test_max_gate_count_filtered_by_name() -> None:
    assert_max_gate_count(_sample_circuit(), 2, gate="cx")
    assert_max_gate_count(_sample_circuit(), 0, gate="rz")  # absent gate -> 0
    with pytest.raises(AssertionError, match="'cx' gate count"):
        assert_max_gate_count(_sample_circuit(), 1, gate="cx")


def test_max_gate_count_gate_name_is_case_insensitive() -> None:
    assert_max_gate_count(_sample_circuit(), 2, gate="CX")


# --------------------------------------------------------------------------- #
# assert_max_two_qubit_count                                                  #
# --------------------------------------------------------------------------- #


def test_max_two_qubit_count() -> None:
    assert_max_two_qubit_count(_sample_circuit(), 2)
    with pytest.raises(AssertionError, match="two-qubit-gate count exceeds"):
        assert_max_two_qubit_count(_sample_circuit(), 1)


# --------------------------------------------------------------------------- #
# assert_max_t_count                                                          #
# --------------------------------------------------------------------------- #


def test_max_t_count() -> None:
    assert_max_t_count(_sample_circuit(), 1)
    with pytest.raises(AssertionError, match="T-count exceeds"):
        assert_max_t_count(_sample_circuit(), 0)


def test_t_count_includes_tdg() -> None:
    qc = QuantumCircuit(1)
    qc.t(0)
    qc.tdg(0)
    assert_max_t_count(qc, 2)
    with pytest.raises(AssertionError):
        assert_max_t_count(qc, 1)


# --------------------------------------------------------------------------- #
# Validation                                                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "func",
    [assert_max_depth, assert_max_gate_count, assert_max_two_qubit_count, assert_max_t_count],
)
def test_negative_limit_raises(func) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        func(_sample_circuit(), -1)


@pytest.mark.parametrize(
    "func",
    [assert_max_depth, assert_max_gate_count, assert_max_two_qubit_count, assert_max_t_count],
)
def test_bool_limit_rejected(func) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        func(_sample_circuit(), True)


def test_none_circuit_raises() -> None:
    with pytest.raises(ValueError, match="circuit must not be None"):
        assert_max_depth(None, 5)


def test_empty_gate_name_raises() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        assert_max_gate_count(_sample_circuit(), 5, gate="")


def test_failure_message_includes_resource_breakdown() -> None:
    with pytest.raises(AssertionError) as exc:
        assert_max_depth(_sample_circuit(), 1, msg="optimiser regressed")
    text = str(exc.value)
    assert "optimiser regressed" in text
    assert "Circuit resources:" in text
    assert "two-qubit gates:" in text
