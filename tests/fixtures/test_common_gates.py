"""Tests for :mod:`qtest.fixtures.common_gates`."""

from __future__ import annotations

from typing import Any, Callable

import pytest

pytest.importorskip("qiskit")

from qtest.assertions import assert_state_close, assert_unitary  # noqa: E402
from qtest.fixtures.common_gates import (  # noqa: E402
    _CLIFFORD_SINGLES,
    hadamards,
    random_clifford_circuit,
)

# --------------------------------------------------------------------------- #
# hadamards                                                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_hadamards_qubit_count(n: int) -> None:
    assert hadamards(n).num_qubits == n


def test_hadamards_default_n_is_one() -> None:
    assert hadamards().num_qubits == 1


def test_hadamards_n1_produces_plus_state() -> None:
    assert_state_close(hadamards(1), "plus")


def test_hadamards_n2_produces_uniform_state() -> None:
    """H ⊗ H |00> = uniform 2-qubit superposition (all amplitudes 0.5)."""
    import numpy as np

    expected = np.full(4, 0.5, dtype=complex)
    assert_state_close(hadamards(2), expected)


def test_hadamards_is_unitary() -> None:
    assert_unitary(hadamards(3))


@pytest.mark.parametrize("bad", [0, -1, 2.5, True])
def test_hadamards_rejects_invalid_n(bad: Any) -> None:
    with pytest.raises(ValueError):
        hadamards(bad)


# --------------------------------------------------------------------------- #
# random_clifford_circuit                                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_random_clifford_qubit_count(n: int) -> None:
    assert random_clifford_circuit(n=n, depth=10, seed=0).num_qubits == n


def test_random_clifford_default_arguments() -> None:
    qc = random_clifford_circuit()
    assert qc.num_qubits == 1
    # depth=10 -> at most 10 gates (could be fewer if duplicates collapse,
    # but raw count_ops should show ~10 instructions).
    total_ops = sum(qc.count_ops().values())
    assert total_ops == 10


@pytest.mark.parametrize("depth", [0, 1, 5, 25])
def test_random_clifford_depth_respected(depth: int) -> None:
    qc = random_clifford_circuit(n=2, depth=depth, seed=42)
    assert sum(qc.count_ops().values()) == depth


def test_random_clifford_seed_reproducibility() -> None:
    a = random_clifford_circuit(n=3, depth=15, seed=42)
    b = random_clifford_circuit(n=3, depth=15, seed=42)
    # Same seed -> structurally identical (same qasm)
    from qiskit.qasm2 import dumps

    assert dumps(a) == dumps(b)


def test_random_clifford_different_seeds_differ() -> None:
    """Two seeds should produce different gate sequences."""
    from qiskit.qasm2 import dumps

    a = random_clifford_circuit(n=3, depth=20, seed=1)
    b = random_clifford_circuit(n=3, depth=20, seed=2)
    assert dumps(a) != dumps(b)


def test_random_clifford_uses_only_clifford_gates() -> None:
    """Every gate must come from the Clifford generator set."""
    qc = random_clifford_circuit(n=3, depth=50, seed=7)
    allowed = set(_CLIFFORD_SINGLES) | {"cx"}
    for name in qc.count_ops():
        assert name in allowed, f"non-Clifford gate {name!r} in random circuit"


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_random_clifford_is_always_unitary(seed: int) -> None:
    """Sanity: any Clifford circuit must be unitary."""
    assert_unitary(random_clifford_circuit(n=3, depth=20, seed=seed))


def test_random_clifford_single_qubit_avoids_cx() -> None:
    """For n=1 the random_clifford should never emit a CX gate."""
    qc = random_clifford_circuit(n=1, depth=30, seed=0)
    assert "cx" not in qc.count_ops()


@pytest.mark.parametrize("bad_n", [0, -1, 2.5, True])
def test_random_clifford_rejects_invalid_n(bad_n: Any) -> None:
    with pytest.raises(ValueError):
        random_clifford_circuit(n=bad_n, depth=5)


@pytest.mark.parametrize("bad_depth", [-1, -10, 2.5, True])
def test_random_clifford_rejects_invalid_depth(bad_depth: Any) -> None:
    with pytest.raises(ValueError):
        random_clifford_circuit(n=2, depth=bad_depth)


def test_random_clifford_depth_zero_returns_identity_circuit() -> None:
    qc = random_clifford_circuit(n=2, depth=0)
    assert sum(qc.count_ops().values()) == 0


# --------------------------------------------------------------------------- #
# Pytest fixtures                                                             #
# --------------------------------------------------------------------------- #


def test_hadamard_circuit_factory_fixture(
    hadamard_circuit: Callable[..., Any],
) -> None:
    qc = hadamard_circuit(2)
    assert qc.num_qubits == 2
    assert_unitary(qc)


def test_random_clifford_factory_fixture(
    random_clifford: Callable[..., Any],
) -> None:
    qc = random_clifford(n=2, depth=15, seed=99)
    assert qc.num_qubits == 2
    assert sum(qc.count_ops().values()) == 15
    assert_unitary(qc)


def test_random_clifford_fixture_uses_default_args(
    random_clifford: Callable[..., Any],
) -> None:
    """When called without args, factory should use depth=10, n=1."""
    qc = random_clifford()
    assert qc.num_qubits == 1
    assert sum(qc.count_ops().values()) == 10
