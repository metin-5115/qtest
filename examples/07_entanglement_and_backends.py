"""New assertions and multi-backend testing with qtest.

Run: pytest examples/07_entanglement_and_backends.py -v

Covers the entanglement / phase / commutation / marginal assertions and shows
the same Bell test running on the Qiskit, Cirq, and PennyLane backends.

The Cirq and PennyLane sections are skipped automatically if those optional
SDKs are not installed (`pip install 'qtest[cirq]'` / `'qtest[pennylane]'`).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from qiskit import QuantumCircuit

from qtest import (
    assert_commutes,
    assert_distribution_close,
    assert_entangled,
    assert_measurement_probabilities,
    assert_phase,
    assert_separable,
)

# --------------------------------------------------------------------------- #
# Entanglement                                                                #
# --------------------------------------------------------------------------- #


def test_bell_is_entangled_and_product_is_not() -> None:
    bell = QuantumCircuit(2)
    bell.h(0)
    bell.cx(0, 1)
    assert_entangled(bell)

    product = QuantumCircuit(2)
    product.h(0)  # |+> ⊗ |0>
    assert_separable(product)


# --------------------------------------------------------------------------- #
# Relative phase                                                              #
# --------------------------------------------------------------------------- #


def test_t_gate_imparts_pi_over_4_phase() -> None:
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.t(0)
    # |+> -> (|0> + e^{iπ/4}|1>)/√2
    assert_phase(qc, 0, 1, math.pi / 4, tolerance=1e-6)


# --------------------------------------------------------------------------- #
# Commutation                                                                 #
# --------------------------------------------------------------------------- #


def test_pauli_commutation_relations() -> None:
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    z = np.array([[1, 0], [0, -1]], dtype=complex)
    assert_commutes(x, x)  # [X, X] = 0
    assert_commutes(x, z, anti=True)  # {X, Z} = 0


# --------------------------------------------------------------------------- #
# Marginal distribution                                                       #
# --------------------------------------------------------------------------- #


def test_one_qubit_marginal_of_bell_is_balanced() -> None:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    assert_measurement_probabilities(
        qc, {"0": 0.5, "1": 0.5}, bit_positions=[0], shots=4096, tolerance=0.05, seed=42
    )


# --------------------------------------------------------------------------- #
# Same test, three backends                                                   #
# --------------------------------------------------------------------------- #


def test_bell_on_qiskit_backend() -> None:
    from qtest.backends import QiskitBackend

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    assert_distribution_close(
        qc, {"00": 0.5, "11": 0.5}, shots=4096, tolerance=0.05, seed=42, backend=QiskitBackend()
    )


def test_bell_on_cirq_backend() -> None:
    cirq = pytest.importorskip("cirq")
    from qtest.backends import CirqBackend

    q = cirq.LineQubit.range(2)
    circuit = cirq.Circuit([cirq.H(q[0]), cirq.CNOT(q[0], q[1])])
    assert_distribution_close(
        circuit, {"00": 0.5, "11": 0.5}, shots=4096, tolerance=0.05, seed=42, backend=CirqBackend()
    )


def test_bell_on_pennylane_backend() -> None:
    qml = pytest.importorskip("pennylane")
    from qtest.backends import PennyLaneBackend

    tape = qml.tape.QuantumScript([qml.Hadamard(0), qml.CNOT([0, 1])])
    assert_distribution_close(
        tape,
        {"00": 0.5, "11": 0.5},
        shots=4096,
        tolerance=0.05,
        seed=42,
        backend=PennyLaneBackend(),
    )
