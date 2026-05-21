"""Pytest fixtures (and plain factory functions) for common gate-sequence circuits.

Provides:

* ``hadamard_circuit(n)`` — n-qubit Hadamard layer (one ``H`` per qubit).
* ``random_clifford_circuit(n, depth, seed)`` — depth-controlled random
  Clifford built from the gate set ``{H, S, S†, X, Y, Z}`` plus occasional
  ``CX`` for ``n >= 2``.

Examples
--------
>>> # In a pytest test:
>>> def test_uniform(hadamard_circuit):  # doctest: +SKIP
...     assert_distribution_close(
...         hadamard_circuit(3), expected={"000": 1/8, ..., "111": 1/8}
...     )

>>> # As plain functions:
>>> from qtest.fixtures.common_gates import random_clifford_circuit
>>> qc = random_clifford_circuit(n=2, depth=20, seed=42)  # doctest: +SKIP
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pytest

_CLIFFORD_SINGLES: tuple[str, ...] = ("h", "s", "sdg", "x", "y", "z")


# --------------------------------------------------------------------------- #
# Plain factory functions                                                     #
# --------------------------------------------------------------------------- #


def hadamards(n: int = 1) -> Any:
    """Return an n-qubit circuit that applies :math:`H` to every qubit.

    The resulting state from :math:`|0\\rangle^{\\otimes n}` is the uniform
    superposition :math:`H^{\\otimes n}|0\\rangle^{\\otimes n}`.
    """
    from qiskit import QuantumCircuit

    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    qc = QuantumCircuit(n, name=f"hadamards_{n}")
    for q in range(n):
        qc.h(q)
    return qc


def random_clifford_circuit(
    n: int = 1,
    depth: int = 10,
    seed: int | None = None,
) -> Any:
    """Return a random Clifford circuit on *n* qubits with *depth* gates.

    Each "layer" is a single gate: with probability ~0.3 (when ``n >= 2``)
    a random ``CX``, otherwise a random gate drawn from
    ``{H, S, S†, X, Y, Z}`` on a random qubit. Reproducible when
    ``seed`` is set.

    Parameters
    ----------
    n
        Qubit count (``n >= 1``).
    depth
        Number of gates (``depth >= 0``). ``depth == 0`` returns the identity
        circuit.
    seed
        Seed for the RNG; ``None`` means non-deterministic.
    """
    from qiskit import QuantumCircuit

    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    if not isinstance(depth, int) or isinstance(depth, bool) or depth < 0:
        raise ValueError(f"depth must be a non-negative integer, got {depth!r}")

    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n, name=f"random_clifford_d{depth}_n{n}")
    for _ in range(depth):
        if n > 1 and rng.random() < 0.3:
            ctrl, tgt = rng.choice(n, size=2, replace=False).tolist()
            qc.cx(int(ctrl), int(tgt))
        else:
            gate_name = _CLIFFORD_SINGLES[int(rng.integers(0, len(_CLIFFORD_SINGLES)))]
            qubit = int(rng.integers(0, n))
            getattr(qc, gate_name)(qubit)
    return qc


# --------------------------------------------------------------------------- #
# Pytest fixtures                                                             #
# --------------------------------------------------------------------------- #


@pytest.fixture
def hadamard_circuit() -> Callable[..., Any]:
    """Factory fixture for n-qubit Hadamard-layer circuits.

    Example::

        def test_uniform(hadamard_circuit):
            qc = hadamard_circuit(3)         # 3 qubits, all in |+>
            qc.measure_all()
            assert_distribution_close(
                qc, expected={f"{i:03b}": 1/8 for i in range(8)}
            )
    """
    return hadamards


@pytest.fixture
def random_clifford() -> Callable[..., Any]:
    """Factory fixture for random Clifford circuits.

    Example::

        def test_clifford_unitarity(random_clifford):
            for seed in range(5):
                assert_unitary(random_clifford(n=3, depth=20, seed=seed))
    """
    return random_clifford_circuit
