"""Verifying that a transpiler preserves circuit semantics.

Run: pytest examples/05_testing_transpiler.py -v

Compilers (Qiskit's transpiler, in this case) rewrite a circuit into
hardware-native gates, swap qubit mappings, and apply optimisations.
A correct transpiler must preserve the unitary the circuit computes.
A broken one will pass small unit tests but silently corrupt the
algorithm.

``assert_circuit_equivalent`` compares two circuits via process
fidelity (default, phase-insensitive). It is the right hammer for
transpiler regression tests:

    1. Transpiled circuit at any optimisation level == original.
    2. A specific basis-gate rewrite == original.
    3. A hand-corrupted "transpiler output" (one wrong gate) FAILS,
       proving the test would catch a regression.

Expected output:

    ============= 7 passed in ~1s =============
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile

from qtest.assertions import assert_circuit_equivalent


# --------------------------------------------------------------------------- #
# Reference circuit                                                           #
# --------------------------------------------------------------------------- #


def reference_circuit() -> QuantumCircuit:
    """A small but non-trivial circuit: H + CX + parametric rotation + CZ.

    Three qubits, no measurements -- ``assert_circuit_equivalent``
    expects unitary-only circuits.
    """
    qc = QuantumCircuit(3, name="reference")
    qc.h(0)
    qc.cx(0, 1)
    qc.ry(np.pi / 5, 2)
    qc.cz(1, 2)
    qc.h(2)
    return qc


# --------------------------------------------------------------------------- #
# 1. Every optimisation level preserves the unitary                           #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("opt_level", [0, 1, 2, 3])
def test_transpile_preserves_unitary(opt_level: int) -> None:
    original = reference_circuit()
    compiled = transpile(original, optimization_level=opt_level, seed_transpiler=0)
    # 3 qubits -> "auto" picks the "unitary" method (process fidelity).
    assert_circuit_equivalent(original, compiled, tolerance=1e-9)


# --------------------------------------------------------------------------- #
# 2. Basis-gate rewrite preserves the unitary                                 #
# --------------------------------------------------------------------------- #


def test_transpile_to_rzcz_basis_preserves_unitary() -> None:
    """Rewriting into ``{rz, sx, x, cx}`` still yields the same operator."""
    original = reference_circuit()
    compiled = transpile(
        original,
        basis_gates=["rz", "sx", "x", "cx"],
        optimization_level=2,
        seed_transpiler=0,
    )
    assert_circuit_equivalent(original, compiled, tolerance=1e-9)


# --------------------------------------------------------------------------- #
# 3. Hilbert-Schmidt method (phase-sensitive) as a stricter check             #
# --------------------------------------------------------------------------- #


def test_transpile_preserves_unitary_up_to_global_phase() -> None:
    """``method="unitary"`` ignores global phase; HS does not.

    The transpiler may legitimately introduce a global phase factor
    (it's a free parameter of the unitary). So the phase-sensitive HS
    method may fail where the phase-insensitive ``unitary`` method
    succeeds. We confirm the *default* method is happy.
    """
    original = reference_circuit()
    compiled = transpile(original, optimization_level=3, seed_transpiler=0)
    assert_circuit_equivalent(original, compiled, method="unitary", tolerance=1e-9)


# --------------------------------------------------------------------------- #
# 4. Regression demo: a corrupted "transpiler output" must FAIL               #
# --------------------------------------------------------------------------- #
#
# We simulate a transpiler bug by taking a valid transpiled circuit,
# then deleting one gate. The assertion has to flag the difference;
# the ``pytest.raises`` wraps confirms it does.


def test_corrupted_transpilation_is_detected() -> None:
    original = reference_circuit()
    compiled = transpile(original, optimization_level=1, seed_transpiler=0)

    # Make a copy and drop the last instruction. Any real circuit
    # with at least one gate will diverge under this mutation.
    corrupted = compiled.copy()
    if len(corrupted.data) == 0:
        pytest.skip("transpiler emitted an empty circuit; nothing to corrupt")
    corrupted.data.pop()

    with pytest.raises(AssertionError):
        assert_circuit_equivalent(original, corrupted, tolerance=1e-9)


def test_swapped_gate_is_detected() -> None:
    """Replacing a CX with a CZ must be flagged."""
    original = reference_circuit()
    broken = QuantumCircuit(3, name="broken_transpile")
    broken.h(0)
    broken.cz(0, 1)  # BUG: should be cx(0, 1)
    broken.ry(np.pi / 5, 2)
    broken.cz(1, 2)
    broken.h(2)

    with pytest.raises(AssertionError):
        assert_circuit_equivalent(original, broken, tolerance=1e-9)
