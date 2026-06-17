"""Advanced algorithms — the full qtest toolkit on real workloads.

Run: pytest examples/08_advanced_algorithms.py -v
     pytest examples/08_advanced_algorithms.py -v --qtest-summary

Three non-trivial workloads, each verified from several angles:

  1. Quantum teleportation with REAL mid-circuit measurement + classical
     feed-forward corrections (Qiskit `if_test`), checked for arbitrary inputs.
  2. A VQE-style hardware-efficient ansatz: always unitary (property-based),
     and entangling or separable depending on its parameters.
  3. Quantum phase estimation: exact in the ideal case, and characterised
     under realistic depolarizing noise (hardware-readiness).
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

import qtest
from qtest.strategies import quantum_circuits

# =========================================================================== #
# Circuit library                                                             #
# =========================================================================== #


def teleport_real(theta: float) -> QuantumCircuit:
    """Teleport Ry(theta)|0> from q0 to q2 with real measurement + feed-forward.

    Unlike a deferred-measurement toy, this measures q0/q1 mid-circuit and
    applies X/Z corrections to q2 conditioned on those classical bits — exactly
    how teleportation runs on hardware. The target is then uncomputed and
    measured, so a correct protocol reads ``0`` deterministically regardless of
    the random mid-circuit outcomes.
    """
    q = QuantumRegister(3, "q")
    c = ClassicalRegister(3, "c")
    qc = QuantumCircuit(q, c, name="teleport_real")
    qc.ry(theta, 0)  # unknown state |psi> on q0
    qc.h(1)
    qc.cx(1, 2)  # Bell pair on q1, q2
    qc.cx(0, 1)
    qc.h(0)
    qc.measure(0, 0)  # mid-circuit measurements
    qc.measure(1, 1)
    with qc.if_test((c[1], 1)):  # classical feed-forward corrections
        qc.x(2)
    with qc.if_test((c[0], 1)):
        qc.z(2)
    qc.ry(-theta, 2)  # uncompute the target -> q2 == |0>
    qc.measure(2, 2)
    return qc


def ansatz(params: list[float]) -> QuantumCircuit:
    """A 2-qubit hardware-efficient ansatz: Ry rotations + a CX entangler."""
    qc = QuantumCircuit(2, name="ansatz")
    qc.ry(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    return qc


def qpe_phase_gate(phase: float, t: int = 3) -> QuantumCircuit:
    """QPE of a phase-gate eigenvalue e^{2*pi*i*phase} with t counting qubits."""
    qc = QuantumCircuit(t + 1, name=f"qpe_{phase}")
    qc.x(t)
    for q in range(t):
        qc.h(q)
    for j in range(t):
        qc.cp(2 * np.pi * phase * (2**j), j, t)
    for i in range(t // 2):
        qc.swap(i, t - 1 - i)
    for j in range(t):
        for m in range(j):
            qc.cp(-np.pi / 2 ** (j - m), m, j)
        qc.h(j)
    qc.measure_all()
    return qc


# =========================================================================== #
# 1. Teleportation with real mid-circuit measurement                          #
# =========================================================================== #


@pytest.mark.parametrize("theta", [0.0, np.pi / 4, np.pi / 2, 0.9, 2.5, np.pi])
def test_real_teleportation_transfers_the_state(theta: float):
    qc = teleport_real(theta)
    # c2 (the target's measurement) is the leftmost bit; must be 0 every shot.
    qtest.assert_measurement_probabilities(
        qc, {"0": 1.0}, bit_positions=[0], shots=2048, tolerance=0.001, seed=1
    )


@settings(max_examples=10, deadline=None)
@given(theta=st.floats(min_value=0.0, max_value=2 * np.pi))
def test_real_teleportation_works_for_any_state(theta: float):
    """Property: feed-forward teleportation succeeds for an arbitrary state."""
    qc = teleport_real(theta)
    qtest.assert_measurement_probabilities(
        qc, {"0": 1.0}, bit_positions=[0], shots=1024, tolerance=0.001, seed=1
    )


# =========================================================================== #
# 2. VQE-style parametric ansatz                                              #
# =========================================================================== #


@settings(max_examples=25, deadline=None)
@given(
    a=st.floats(min_value=-2 * np.pi, max_value=2 * np.pi),
    b=st.floats(min_value=-2 * np.pi, max_value=2 * np.pi),
)
def test_ansatz_is_always_unitary(a: float, b: float):
    """Property: the ansatz is a valid unitary for every parameter setting."""
    qtest.assert_unitary(ansatz([a, b]), tolerance=1e-9)


def test_ansatz_prepares_bell_at_known_parameters():
    qtest.assert_state_close(ansatz([np.pi / 2, 0.0]), "bell", tolerance=1e-9)


def test_ansatz_at_zero_is_the_ground_state():
    qtest.assert_state_close(ansatz([0.0, 0.0]), [1.0, 0.0, 0.0, 0.0], tolerance=1e-9)


def test_ansatz_entanglement_is_parameter_dependent():
    # Control qubit in |+> -> CX entangles.
    qtest.assert_entangled(ansatz([np.pi / 2, 0.0]), qubits=[0])
    # Control qubit in |0> -> CX is inert -> product state.
    qtest.assert_separable(ansatz([0.0, np.pi / 2]), qubits=[0])


# =========================================================================== #
# 3. Quantum phase estimation: ideal + under noise                            #
# =========================================================================== #


def test_qpe_estimates_quarter_phase_exactly():
    # phase = 1/4 -> counting register reads binary "010" (== 2 == 2/8).
    qc = qpe_phase_gate(0.25, t=3)
    qtest.assert_measurement_probabilities(
        qc, {"010": 1.0}, bit_positions=[1, 2, 3], shots=8192, tolerance=0.02, seed=1
    )


def test_qpe_tolerates_small_hardware_noise():
    # Deeper circuits are noise-sensitive: QPE stays within budget only for
    # modest noise. Beyond this, real hardware needs error mitigation.
    qc = qpe_phase_gate(0.25, t=3)
    qtest.assert_robust_to_noise(
        qc,
        {"1010": 1.0},  # full reading: eigenstate qubit |1> + register "010"
        noise_levels=[0.002, 0.005, 0.01],
        max_distance=0.15,
        noise_type="depolarizing",
        shots=8192,
        seed=1,
    )


# =========================================================================== #
# 4. Property-based: any generated circuit is unitary                         #
# =========================================================================== #


@settings(max_examples=25, deadline=None)
@given(circuit=quantum_circuits(n_qubits=st.integers(1, 3), depth=st.integers(1, 6)))
def test_any_generated_circuit_is_unitary(circuit):
    qtest.assert_unitary(circuit, tolerance=1e-9)
