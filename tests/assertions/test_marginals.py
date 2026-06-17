"""Tests for ``assert_measurement_probabilities``."""

from __future__ import annotations

import pytest

from qtest import assert_measurement_probabilities

pytest.importorskip("qiskit")

from qiskit import QuantumCircuit  # noqa: E402


def _bell_measured() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    return qc


def test_marginal_single_bit_is_balanced() -> None:
    assert_measurement_probabilities(
        _bell_measured(),
        {"0": 0.5, "1": 0.5},
        bit_positions=[0],
        shots=4096,
        tolerance=0.05,
        seed=1,
    )


def test_marginal_other_bit_is_balanced() -> None:
    assert_measurement_probabilities(
        _bell_measured(),
        {"0": 0.5, "1": 0.5},
        bit_positions=[1],
        shots=4096,
        tolerance=0.05,
        seed=1,
    )


def test_marginal_hellinger_metric() -> None:
    assert_measurement_probabilities(
        _bell_measured(),
        {"0": 0.5, "1": 0.5},
        bit_positions=[0],
        shots=4096,
        tolerance=0.05,
        metric="hellinger",
        seed=1,
    )


def test_marginal_chi_square_metric() -> None:
    assert_measurement_probabilities(
        _bell_measured(),
        {"0": 0.5, "1": 0.5},
        bit_positions=[0],
        shots=4096,
        tolerance=0.01,  # significance level
        metric="chi_square",
        seed=1,
    )


def test_marginal_chi_square_failure() -> None:
    with pytest.raises(AssertionError, match="Marginal distribution mismatch"):
        assert_measurement_probabilities(
            _bell_measured(),
            {"0": 0.9, "1": 0.1},
            bit_positions=[0],
            shots=4096,
            tolerance=0.05,
            metric="chi_square",
            seed=1,
        )


def test_marginal_out_of_range_position_raises() -> None:
    with pytest.raises(ValueError, match="out of range"):
        assert_measurement_probabilities(
            _bell_measured(), {"0": 0.5, "1": 0.5}, bit_positions=[9], shots=64, seed=1
        )


def test_marginal_wrong_expected_raises() -> None:
    with pytest.raises(AssertionError, match="Marginal distribution mismatch"):
        assert_measurement_probabilities(
            _bell_measured(),
            {"0": 1.0, "1": 0.0},
            bit_positions=[0],
            shots=4096,
            tolerance=0.02,
            seed=1,
        )


def test_empty_bit_positions_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        assert_measurement_probabilities(_bell_measured(), {"0": 1.0}, bit_positions=[])


def test_key_width_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length"):
        assert_measurement_probabilities(
            _bell_measured(), {"00": 0.5, "11": 0.5}, bit_positions=[0]
        )


def test_unknown_metric_raises() -> None:
    with pytest.raises(ValueError, match="Unknown metric"):
        assert_measurement_probabilities(
            _bell_measured(), {"0": 0.5, "1": 0.5}, bit_positions=[0], metric="bogus"
        )
