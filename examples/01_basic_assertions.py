"""Basic qtest assertions in action.

Run: pytest examples/01_basic_assertions.py -v

This file walks through the four headline assertion functions qtest
ships with. The tests are deliberately tiny -- the goal is to *see*
what each assertion is for and what its arguments mean, not to cover
any real algorithm.

What you will see (when the suite passes):

    ============= 4 passed in 0.4s =============

If you intentionally break one of the circuits (suggestion at the
bottom of the file), the failure message tells you exactly what went
wrong -- shot counts, measured distribution, distance metric used,
etc.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit

from qtest.assertions import (
    assert_circuit_equivalent,
    assert_distribution_close,
    assert_state_close,
    assert_unitary,
)


# --------------------------------------------------------------------------- #
# 1. Bell state distribution                                                  #
# --------------------------------------------------------------------------- #
#
# A Bell pair is the simplest entangled state: starting from |00>, apply H
# on qubit 0, then CNOT(0 -> 1). Measuring in the computational basis
# should give 50/50 odds of "00" vs "11", and never "01" or "10".


def test_bell_state_has_uniform_50_50_distribution() -> None:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    # Tolerance = 0.05 means each probability may drift by up to 5
    # percentage points from the expected 50/50. With shots=4000 the
    # standard deviation of each empirical frequency is ~0.008.
    assert_distribution_close(
        qc,
        expected={"00": 0.5, "11": 0.5},
        tolerance=0.05,
        shots=4000,
        seed=42,
    )


# --------------------------------------------------------------------------- #
# 2. Single-qubit |+> state                                                   #
# --------------------------------------------------------------------------- #
#
# Hadamard on |0> produces |+> = (|0> + |1>) / sqrt(2). assert_state_close
# uses fidelity by default, so the comparison ignores any global phase
# the simulator might introduce.


def test_hadamard_produces_plus_state() -> None:
    qc = QuantumCircuit(1)
    qc.h(0)

    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    assert_state_close(qc, expected_state=plus, tolerance=1e-9)

    # qtest also ships a named state library; the line below is
    # equivalent to the array form above.
    assert_state_close(qc, expected_state="plus", tolerance=1e-9)


# --------------------------------------------------------------------------- #
# 3. Custom gate is unitary                                                   #
# --------------------------------------------------------------------------- #
#
# assert_unitary works on raw matrices, on objects with .to_matrix()
# (e.g. Qiskit Gates), and on whole circuits. Here we build a custom
# 2x2 matrix and assert that U^dag U = I to within 1e-12.


def test_custom_gate_matrix_is_unitary() -> None:
    theta = 1.2345
    # A Y-rotation matrix.
    u = np.array(
        [
            [np.cos(theta / 2), -np.sin(theta / 2)],
            [np.sin(theta / 2), np.cos(theta / 2)],
        ],
        dtype=complex,
    )
    assert_unitary(u, tolerance=1e-12)


# --------------------------------------------------------------------------- #
# 4. Two circuits implement the same unitary                                  #
# --------------------------------------------------------------------------- #
#
# A standard identity: H = RY(pi/2) * Z. In Qiskit, gates are added
# left-to-right but the unitary product is right-to-left -- so the
# circuit ``qc.z(0); qc.ry(pi/2, 0)`` evaluates to ``RY(pi/2) @ Z``,
# which equals H exactly.


def test_h_equals_z_then_ry() -> None:
    direct = QuantumCircuit(1)
    direct.h(0)

    decomposed = QuantumCircuit(1)
    decomposed.z(0)
    decomposed.ry(np.pi / 2, 0)

    # Default method "unitary" computes process fidelity, which is
    # phase-insensitive -- exactly what we want for refactor tests.
    assert_circuit_equivalent(direct, decomposed, tolerance=1e-9)


# --------------------------------------------------------------------------- #
# Try breaking it!                                                            #
# --------------------------------------------------------------------------- #
#
# To see what a failure looks like, change the Bell-state expectation
# to {"00": 0.6, "11": 0.4} or swap the Z for an X in
# test_h_equals_z_then_ry's `decomposed` circuit. The error message
# will print the measured vs. expected distribution and the metric
# used, so you can diagnose the regression without re-running by hand.
