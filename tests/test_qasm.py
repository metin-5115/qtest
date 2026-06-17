"""Tests for :mod:`qtest.qasm` (OpenQASM loading)."""

from __future__ import annotations

from pathlib import Path

import pytest

from qtest.qasm import _detect_version, load_qasm, load_qasm_file

pytest.importorskip("qiskit")

from qiskit import QuantumCircuit, qasm2, qasm3  # noqa: E402


def _bell() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


# --------------------------------------------------------------------------- #
# Version detection                                                           #
# --------------------------------------------------------------------------- #


def test_detect_version() -> None:
    assert _detect_version("OPENQASM 2.0;\nqreg q[1];") == "2"
    assert _detect_version("OPENQASM 3.0;\nqubit q;") == "3"


def test_detect_version_missing_header_raises() -> None:
    with pytest.raises(ValueError, match="OPENQASM"):
        _detect_version("h q[0];")


def test_detect_version_unsupported_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        _detect_version("OPENQASM 4.0;")


# --------------------------------------------------------------------------- #
# QASM 2.0                                                                    #
# --------------------------------------------------------------------------- #


def test_load_qasm2_auto() -> None:
    qc = load_qasm(qasm2.dumps(_bell()))
    assert qc.num_qubits == 2


def test_load_qasm2_explicit_version() -> None:
    qc = load_qasm(qasm2.dumps(_bell()), version="2")
    assert qc.num_qubits == 2


def test_loaded_qasm2_circuit_works_with_assertions() -> None:
    import qtest

    qc = load_qasm(qasm2.dumps(_bell()))
    qtest.assert_state_close(qc, "bell")


# --------------------------------------------------------------------------- #
# QASM 3.0 (requires qiskit-qasm3-import)                                     #
# --------------------------------------------------------------------------- #


def test_load_qasm3_auto() -> None:
    pytest.importorskip("qiskit_qasm3_import")
    qc = load_qasm(qasm3.dumps(_bell()))
    assert qc.num_qubits == 2


def test_load_qasm3_circuit_works_with_assertions() -> None:
    pytest.importorskip("qiskit_qasm3_import")
    import qtest

    qc = load_qasm(qasm3.dumps(_bell()), version="3")
    qtest.assert_state_close(qc, "bell")


# --------------------------------------------------------------------------- #
# Validation & files                                                          #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad", ["", "   ", "\n"])
def test_empty_program_raises(bad: str) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        load_qasm(bad)


def test_unknown_version_raises() -> None:
    with pytest.raises(ValueError, match="version must be"):
        load_qasm(qasm2.dumps(_bell()), version="9")


def test_load_qasm_file(tmp_path: Path) -> None:
    path = tmp_path / "bell.qasm"
    path.write_text(qasm2.dumps(_bell()), encoding="utf-8")
    qc = load_qasm_file(path)
    assert qc.num_qubits == 2


def test_load_qasm_file_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_qasm_file(tmp_path / "nope.qasm")
