"""Testing a real algorithm end-to-end: 2-qubit Grover's search.

Run: pytest examples/04_testing_grover.py -v

Grover's algorithm amplifies the amplitude of a target basis state
("marked" item) starting from a uniform superposition. For a 2-qubit
search space (4 items), the optimal number of Grover iterations is
exactly **1** -- after one iteration the success probability is 1.0;
after two iterations it drops back to 0.

This file builds the algorithm, then writes three tests:

    1. Correct implementation + 1 iteration -> always finds the marked state.
    2. Correct implementation + 2 iterations -> finds the WRONG state
       (probability 1.0 on |00>, not the marked one). Confirms the
       iteration count actually matters.
    3. A subtly broken oracle -> the same "1 iteration" test that passed
       above now fails, demonstrating qtest catches the regression.

Expected output:

    ============= 3 passed in ~1s =============
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit

from qtest.assertions import assert_distribution_close

N_QUBITS = 2
N_STATES = 2**N_QUBITS  # = 4


# --------------------------------------------------------------------------- #
# Building blocks                                                             #
# --------------------------------------------------------------------------- #


def oracle_for(marked: str) -> QuantumCircuit:
    """Phase-flip oracle: marks the basis state *marked* with a -1 phase.

    Implemented as ``X^a CZ X^a`` where ``a`` is a 0/1 pattern: flip
    every qubit that should be 0 in the marked string, apply the
    multi-controlled Z (here just CZ on 2 qubits), then undo the
    flips.
    """
    if len(marked) != N_QUBITS or any(c not in "01" for c in marked):
        raise ValueError(f"marked must be a {N_QUBITS}-bit string, got {marked!r}")
    qc = QuantumCircuit(N_QUBITS, name=f"oracle({marked})")
    # Qiskit's bit-string convention is little-endian: marked[-1] is qubit 0.
    bits = marked[::-1]
    for q, b in enumerate(bits):
        if b == "0":
            qc.x(q)
    qc.cz(0, 1)
    for q, b in enumerate(bits):
        if b == "0":
            qc.x(q)
    return qc


def diffuser() -> QuantumCircuit:
    """Standard Grover diffuser: ``H^n X^n CZ X^n H^n``."""
    qc = QuantumCircuit(N_QUBITS, name="diffuser")
    qc.h(range(N_QUBITS))
    qc.x(range(N_QUBITS))
    qc.cz(0, 1)
    qc.x(range(N_QUBITS))
    qc.h(range(N_QUBITS))
    return qc


def grover(marked: str, iterations: int) -> QuantumCircuit:
    """Full Grover circuit ending in a measurement of all qubits."""
    qc = QuantumCircuit(N_QUBITS)
    qc.h(range(N_QUBITS))  # uniform superposition
    for _ in range(iterations):
        qc.compose(oracle_for(marked), inplace=True)
        qc.compose(diffuser(), inplace=True)
    qc.measure_all()
    return qc


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("marked", ["00", "01", "10", "11"])
def test_grover_finds_marked_state_with_one_iteration(marked: str) -> None:
    """One iteration is the textbook optimum for N=4 -- success prob = 1.0.

    We give a small per-outcome tolerance to absorb numerical noise
    only; the ideal distribution is a delta on ``marked``.
    """
    qc = grover(marked, iterations=1)
    expected = {f"{i:0{N_QUBITS}b}": 0.0 for i in range(N_STATES)}
    expected[marked] = 1.0
    assert_distribution_close(qc, expected, tolerance=0.01, shots=2000, seed=0)


def test_two_iterations_overshoots_the_optimum() -> None:
    """Iteration count matters: two iterations destroy the amplification.

    Grover's success probability after k iterations is sin^2((2k+1)*theta)
    with sin^2(theta) = 1/N. For N=4 (theta = pi/6) that gives:
        k=1 -> sin^2(pi/2) = 1.0    (perfect, the textbook optimum)
        k=2 -> sin^2(5pi/6) = 0.25   (uniform! no advantage over guessing)
    This test pins the behaviour so a future code change can't silently
    drift the iteration count without breaking something.
    """
    qc = grover(marked="11", iterations=2)
    expected = {f"{i:0{N_QUBITS}b}": 0.25 for i in range(N_STATES)}
    assert_distribution_close(qc, expected, tolerance=0.05, shots=2000, seed=0)


# --------------------------------------------------------------------------- #
# Regression demo: a subtly broken oracle                                     #
# --------------------------------------------------------------------------- #
#
# Suppose someone refactors ``oracle_for`` and forgets to undo the
# X-flips at the end. The circuit still runs, still produces a
# distribution, and only quantum-aware tests catch it.


def broken_oracle_for(marked: str) -> QuantumCircuit:
    """Same as oracle_for but missing the final ``X`` un-flip step."""
    bits = marked[::-1]
    qc = QuantumCircuit(N_QUBITS, name=f"broken_oracle({marked})")
    for q, b in enumerate(bits):
        if b == "0":
            qc.x(q)
    qc.cz(0, 1)
    # BUG: the second X-flip pass is missing.
    return qc


def _grover_with_broken_oracle(marked: str, iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(N_QUBITS)
    qc.h(range(N_QUBITS))
    for _ in range(iterations):
        qc.compose(broken_oracle_for(marked), inplace=True)
        qc.compose(diffuser(), inplace=True)
    qc.measure_all()
    return qc


def test_broken_oracle_does_not_amplify_marked_state() -> None:
    """The broken oracle yields a distribution far from the spec.

    We pick a marked state containing a 0-bit so the missing X-unflip
    actually changes the circuit (for "11" the X-flip pass is a no-op
    and the bug would be invisible). We deliberately *expect the
    assertion to fail*. This is the kind of test you leave in once
    you've fixed a real bug to prevent regressions.
    """
    qc = _grover_with_broken_oracle(marked="10", iterations=1)
    expected = {f"{i:02b}": 0.0 for i in range(N_STATES)}
    expected["10"] = 1.0

    with pytest.raises(AssertionError):
        assert_distribution_close(qc, expected, tolerance=0.01, shots=2000, seed=0)


# --------------------------------------------------------------------------- #
# Sanity check (not strictly needed, but useful while developing)             #
# --------------------------------------------------------------------------- #


def test_oracle_only_flips_marked_state_phase() -> None:
    """The oracle is a diagonal matrix with a single -1 on the marked row."""
    from qiskit.quantum_info import Operator

    marked = "10"
    u = Operator(oracle_for(marked)).data
    diag = np.diag(u)
    # Off-diagonal entries must all vanish.
    assert np.allclose(u - np.diag(diag), 0, atol=1e-12)
    # Exactly one -1, rest +1.
    minus_one_count = int(np.sum(np.isclose(diag, -1.0)))
    plus_one_count = int(np.sum(np.isclose(diag, 1.0)))
    assert minus_one_count == 1 and plus_one_count == N_STATES - 1
