"""Property-based testing of quantum code with Hypothesis + qtest.

Run: pytest examples/02_property_based_testing.py -v

A *property* is an assertion that should hold for **every** input from
some domain, not just hand-picked examples. Hypothesis explores that
domain (random circuits, random Pauli strings, ...) on each run and,
when it finds a counter-example, shrinks it down to the smallest input
that still triggers the failure.

This file demonstrates five canonical quantum properties:

    1. U * U^-1 = identity   (every circuit and its inverse cancel)
    2. P * P    = identity   (every Pauli is its own inverse)
    3. Clifford circuits stay Clifford under composition
    4. Product states stay separable (Schmidt rank = 1)
    5. Density matrices are Hermitian, PSD, and unit-trace

Expected output:

    ============= 5 passed in ~3s =============
"""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford

from qtest.assertions import assert_state_close, assert_unitary
from qtest.strategies import (
    pauli_strings,
    product_states,
    quantum_circuits,
    random_density_matrices,
)

# Quantum work (state-vector simulation, Clifford construction, ...) is
# slow enough that Hypothesis would flag every example with its
# default 200ms deadline. Disable it and the "too_slow" health check.
_QUANTUM = settings(
    max_examples=15,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)

# Map Pauli-string characters to single-qubit gate names so we can
# apply a "XIZY"-style operator to a circuit by name.
_PAULI_GATES = {"I": "id", "X": "x", "Y": "y", "Z": "z"}


# --------------------------------------------------------------------------- #
# 1. Every circuit composed with its inverse acts as the identity            #
# --------------------------------------------------------------------------- #


@_QUANTUM
@given(circuit=quantum_circuits(n_qubits=3, depth=8))
def test_circuit_compose_inverse_is_identity(circuit: QuantumCircuit) -> None:
    combined = circuit.compose(circuit.inverse())
    zero = np.zeros(2**combined.num_qubits, dtype=complex)
    zero[0] = 1.0
    assert_state_close(combined, expected_state=zero, tolerance=1e-6)


# --------------------------------------------------------------------------- #
# 2. Every Pauli string is its own inverse                                    #
# --------------------------------------------------------------------------- #
#
# P^2 = I for every P in {I, X, Y, Z}, so applying a Pauli-string
# circuit twice should return to the start.


def _pauli_circuit(pauli: str) -> QuantumCircuit:
    qc = QuantumCircuit(len(pauli))
    for q, p in enumerate(pauli):
        getattr(qc, _PAULI_GATES[p])(q)
    return qc


@_QUANTUM
@given(pauli=pauli_strings(n_qubits=3))
def test_pauli_strings_are_self_inverse(pauli: str) -> None:
    qc = _pauli_circuit(pauli)
    twice = qc.compose(qc)
    zero = np.zeros(2**3, dtype=complex)
    zero[0] = 1.0
    assert_state_close(twice, expected_state=zero, tolerance=1e-9)


# --------------------------------------------------------------------------- #
# 3. Random Clifford circuits stay Clifford under composition                 #
# --------------------------------------------------------------------------- #
#
# Cliffords are a group: composing any two Cliffords gives another
# Clifford. We draw two random circuits from the Clifford gate set
# {H, S, X, Y, Z, CX} and confirm that their composition is still a
# valid Clifford (the Clifford constructor would raise otherwise).

_CLIFFORD_SET = ["h", "s", "x", "y", "z", "cx"]


@_QUANTUM
@given(
    a=quantum_circuits(n_qubits=2, depth=6, gate_set=_CLIFFORD_SET),
    b=quantum_circuits(n_qubits=2, depth=6, gate_set=_CLIFFORD_SET),
)
def test_clifford_composition_stays_clifford(
    a: QuantumCircuit, b: QuantumCircuit
) -> None:
    composed = a.compose(b)
    # Will raise QiskitError if `composed` is not a Clifford operator.
    clifford = Clifford(composed)
    # Sanity-check: the underlying symplectic matrix is well-formed.
    assert clifford.num_qubits == 2


# --------------------------------------------------------------------------- #
# 4. Product states remain separable across every bipartition                 #
# --------------------------------------------------------------------------- #


@_QUANTUM
@given(psi=product_states(n_qubits=3))
def test_product_states_have_schmidt_rank_one(psi: np.ndarray) -> None:
    for cut in (1, 2):
        matrix = psi.reshape(2**cut, 2 ** (3 - cut))
        s = np.linalg.svd(matrix, compute_uv=False)
        rank = int(np.sum(s > 1e-10))
        assert rank == 1, f"rank={rank} across cut {cut}; expected 1"


# --------------------------------------------------------------------------- #
# 5. Random density matrices are valid quantum states                         #
# --------------------------------------------------------------------------- #
#
# A density matrix must be Hermitian, positive semi-definite, and have
# unit trace. assert_unitary doesn't help here, but a few NumPy lines
# do.


@_QUANTUM
@given(rho=random_density_matrices(n_qubits=2, rank=st.integers(1, 4)))
def test_density_matrix_is_a_valid_state(rho: np.ndarray) -> None:
    assert np.allclose(rho, rho.conj().T, atol=1e-12), "not Hermitian"
    assert np.isclose(np.trace(rho).real, 1.0, atol=1e-12), "trace != 1"
    eigvals = np.linalg.eigvalsh(rho)
    assert np.all(eigvals > -1e-10), f"negative eigenvalue: {eigvals.min()}"


# --------------------------------------------------------------------------- #
# Bonus: assert_unitary across random circuits                                #
# --------------------------------------------------------------------------- #
#
# A property test that doesn't need any of qtest's specialised
# strategies -- every gate-only circuit is unitary by construction.


@_QUANTUM
@given(qc=quantum_circuits(n_qubits=2, depth=5))
def test_every_gate_only_circuit_is_unitary(qc: QuantumCircuit) -> None:
    assert_unitary(qc, tolerance=1e-9)
