"""Load circuits from OpenQASM so they can be tested directly.

A common workflow is to receive a circuit as an OpenQASM program — emitted by a
compiler, exported from another tool, or checked into a fixtures directory — and
want to assert something about it. These helpers parse OpenQASM 2.0 / 3.0 into a
``qiskit.QuantumCircuit`` that plugs straight into qtest's assertions::

    from qtest import load_qasm, assert_state_close

    qc = load_qasm(qasm_program)          # version auto-detected from the header
    assert_state_close(qc, "bell")

OpenQASM 2.0 loading uses Qiskit's built-in parser. OpenQASM 3.0 loading
additionally requires the ``qiskit-qasm3-import`` package (the optional
``qtest[qasm3]`` extra); a clear error is raised if it is missing.

Imports are lazy, so importing :mod:`qtest.qasm` (or :mod:`qtest`) does not
require Qiskit.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_QASM3_MISSING_MSG = (
    "Loading OpenQASM 3.0 requires the 'qiskit-qasm3-import' package. "
    "Install it with:\n  pip install 'qtest[qasm3]'"
)

# Matches the version on the mandatory ``OPENQASM <major>.<minor>;`` header line.
_HEADER_RE = re.compile(r"OPENQASM\s+(\d+)(?:\.\d+)?\s*;", re.IGNORECASE)

_VALID_VERSIONS = frozenset({"auto", "2", "3"})


def _detect_version(program: str) -> str:
    """Return ``"2"`` or ``"3"`` by inspecting the ``OPENQASM`` header."""
    match = _HEADER_RE.search(program)
    if match is None:
        raise ValueError(
            "Could not find an 'OPENQASM <version>;' header; pass version='2' or "
            "version='3' explicitly."
        )
    major = match.group(1)
    if major not in ("2", "3"):
        raise ValueError(f"Unsupported OpenQASM major version {major!r}; expected 2 or 3.")
    return major


def load_qasm(program: str, version: str = "auto") -> Any:
    """Parse an OpenQASM *program* string into a ``qiskit.QuantumCircuit``.

    Parameters
    ----------
    program
        The OpenQASM source as a string.
    version
        ``"auto"`` (default) detects the version from the ``OPENQASM`` header;
        pass ``"2"`` or ``"3"`` to force a parser.

    Returns
    -------
    qiskit.QuantumCircuit

    Raises
    ------
    ValueError
        If *program* is empty, the version cannot be determined, or it is
        unsupported.
    ImportError
        If OpenQASM 3.0 loading is requested but ``qiskit-qasm3-import`` is not
        installed.
    """
    if not isinstance(program, str) or not program.strip():
        raise ValueError("program must be a non-empty OpenQASM string")
    if version not in _VALID_VERSIONS:
        raise ValueError(f"version must be one of {sorted(_VALID_VERSIONS)}, got {version!r}")

    resolved = _detect_version(program) if version == "auto" else version

    if resolved == "2":
        from qiskit import qasm2

        return qasm2.loads(program)

    from qiskit import qasm3

    try:
        return qasm3.loads(program)
    except Exception as exc:  # qiskit raises MissingOptionalLibraryError (an ImportError)
        if isinstance(exc, ImportError):
            raise ImportError(_QASM3_MISSING_MSG) from exc
        raise


def load_qasm_file(path: str | Path, version: str = "auto") -> Any:
    """Read an OpenQASM file and parse it into a ``qiskit.QuantumCircuit``.

    Parameters
    ----------
    path
        Path to a ``.qasm`` file.
    version
        See :func:`load_qasm`.

    Returns
    -------
    qiskit.QuantumCircuit
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"OpenQASM file not found: {file_path}")
    return load_qasm(file_path.read_text(encoding="utf-8"), version=version)


__all__ = ["load_qasm", "load_qasm_file"]
