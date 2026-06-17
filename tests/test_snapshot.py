"""Tests for :mod:`qtest.snapshot` and the ``qtest_snapshot`` fixture."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from qtest.snapshot import Snapshot, _sanitize

pytest_plugins = ["pytester"]

pytest.importorskip("qiskit")

from qiskit import QuantumCircuit  # noqa: E402


def _bell_measured() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    return qc


# --------------------------------------------------------------------------- #
# Snapshot class                                                              #
# --------------------------------------------------------------------------- #


def test_sanitize() -> None:
    assert _sanitize("test_foo[1-2]") == "test_foo_1-2"
    assert _sanitize("///") == "snapshot"


def test_first_run_creates_baseline(tmp_path: Path) -> None:
    snap = Snapshot(tmp_path, "bell", update=False)
    assert not snap.path_for().exists()
    snap.assert_distribution_close(_bell_measured(), shots=4096, tolerance=0.1, seed=1)
    assert snap.path_for().exists()


def test_second_run_compares_and_passes(tmp_path: Path) -> None:
    snap = Snapshot(tmp_path, "bell")
    snap.assert_distribution_close(_bell_measured(), shots=4096, tolerance=0.1, seed=1)
    # Re-sample with a different seed; statistically close, should pass.
    snap.assert_distribution_close(_bell_measured(), shots=4096, tolerance=0.15, seed=2)


def test_mismatch_fails(tmp_path: Path) -> None:
    snap = Snapshot(tmp_path, "bell")
    snap.assert_distribution_close(_bell_measured(), shots=4096, tolerance=0.1, seed=1)

    all_zero = QuantumCircuit(2)
    all_zero.measure_all()
    with pytest.raises(AssertionError, match="Snapshot mismatch"):
        snap.assert_distribution_close(all_zero, shots=4096, tolerance=0.05, seed=1)


def test_update_overwrites(tmp_path: Path) -> None:
    Snapshot(tmp_path, "bell").assert_distribution_close(
        _bell_measured(), shots=4096, tolerance=0.1, seed=1
    )
    all_zero = QuantumCircuit(2)
    all_zero.measure_all()
    # update=True rewrites the baseline to the new circuit's distribution.
    Snapshot(tmp_path, "bell", update=True).assert_distribution_close(
        all_zero, shots=2048, tolerance=0.1, seed=1
    )
    # The rewritten baseline now matches the all-zero circuit.
    Snapshot(tmp_path, "bell").assert_distribution_close(
        all_zero, shots=2048, tolerance=0.05, seed=2
    )


def test_named_snapshots_are_independent(tmp_path: Path) -> None:
    snap = Snapshot(tmp_path, "default")
    snap.assert_distribution_close(_bell_measured(), name="bell", shots=2048, tolerance=0.1, seed=1)
    assert snap.path_for("bell").exists()
    assert not snap.path_for("other").exists()


def test_no_measurements_raises(tmp_path: Path) -> None:
    snap = Snapshot(tmp_path, "x")
    qc = QuantumCircuit(2)
    qc.h(0)
    with pytest.raises(ValueError, match="measurement"):
        snap.assert_distribution_close(qc, shots=128, seed=1)


def test_malformed_snapshot_raises(tmp_path: Path) -> None:
    snap = Snapshot(tmp_path, "bad")
    snap.path_for().write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed"):
        snap.assert_distribution_close(_bell_measured(), shots=128, seed=1)


# --------------------------------------------------------------------------- #
# qtest_snapshot fixture (via pytester)                                       #
# --------------------------------------------------------------------------- #


def test_fixture_create_then_compare(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_snap=textwrap.dedent("""
            from qiskit import QuantumCircuit

            def _bell():
                qc = QuantumCircuit(2)
                qc.h(0); qc.cx(0, 1); qc.measure_all()
                return qc

            def test_snapshot(qtest_snapshot):
                qtest_snapshot.assert_distribution_close(_bell(), shots=4096, tolerance=0.15, seed=1)
            """))
    # First run creates the golden file and passes.
    result = pytester.runpytest()
    result.assert_outcomes(passed=1)
    assert (pytester.path / "__qtest_snapshots__").is_dir()
    # Second run compares against it and passes.
    result2 = pytester.runpytest()
    result2.assert_outcomes(passed=1)
