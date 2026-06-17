"""Tests for :mod:`qtest._report` and assertion-driven distance recording."""

from __future__ import annotations

from typing import Any

import pytest

from qtest._report import _LOG, record_distance


@pytest.fixture(autouse=True)
def _reset_log() -> Any:
    _LOG.reset()
    yield
    _LOG.reset()


def test_record_and_values() -> None:
    record_distance(0.1)
    record_distance(0.2, metric="tv")
    assert _LOG.values == [0.1, 0.2]
    assert len(_LOG) == 2


def test_reset_clears() -> None:
    record_distance(0.5)
    _LOG.reset()
    assert len(_LOG) == 0


def test_distribution_assertion_records_distance() -> None:
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit

    import qtest

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    qtest.assert_distribution_close(qc, {"00": 0.5, "11": 0.5}, shots=4096, tolerance=0.1, seed=1)
    assert len(_LOG) == 1
    assert _LOG.values[0] >= 0.0


def test_chi_square_does_not_record() -> None:
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit

    import qtest

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    qtest.assert_distribution_close(
        qc, {"00": 0.5, "11": 0.5}, shots=4096, tolerance=0.01, metric="chi_square", seed=1
    )
    # chi-square reports a p-value, not a distance — nothing is recorded.
    assert len(_LOG) == 0
