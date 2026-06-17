"""Hypothesis strategy for random :class:`qiskit.QuantumCircuit` instances.

The single public strategy here, :func:`quantum_circuits`, draws a
circuit by composing a depth-many gate "layers" (one gate per layer).
Each gate is sampled independently from a configurable vocabulary, so
the strategy is suitable for property-based tests that need a small
space of legal, well-formed circuits.

Determinism / shrinking
-----------------------
Every random choice (qubit count, depth, gate name, qubit indices,
rotation angle) is performed via :func:`draw`. Hypothesis therefore
controls all entropy, which means:

* the same ``hypothesis.seed`` reproduces the same example;
* shrinking can reduce ``n_qubits``, ``depth``, simplify gate names
  (sampled-from shrinks to the first element), drop two-qubit gates in
  favor of single-qubit ones, and shrink rotation angles toward 0.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Union

from hypothesis import strategies as st

from qtest.strategies.gates import (
    DEFAULT_GATE_SET,
    PARAMETRIC_GATES,
    TWO_QUBIT_GATES,
)

# Strategies accept either an ``int`` (fixed value) or a Hypothesis
# strategy (drawn per example). This alias documents that.
IntLike = Union[int, st.SearchStrategy[int]]


def _resolve_int(draw: st.DrawFn, value: IntLike, *, name: str, minimum: int = 1) -> int:
    """Draw an ``int`` from *value* if it is a strategy, else return as-is.

    Centralised so the dual-mode parameters (``n_qubits``, ``depth``)
    behave identically and validate uniformly.
    """
    resolved = draw(value) if isinstance(value, st.SearchStrategy) else value
    if not isinstance(resolved, int) or isinstance(resolved, bool) or resolved < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {resolved!r}")
    return resolved


# Module-level default strategies — defined once as shared singletons rather
# than inline in the signature (flake8-bugbear B008 forbids calls in defaults).
_DEFAULT_N_QUBITS = st.integers(min_value=1, max_value=5)
_DEFAULT_DEPTH = st.integers(min_value=1, max_value=20)


@st.composite
def quantum_circuits(
    draw: st.DrawFn,
    n_qubits: IntLike = _DEFAULT_N_QUBITS,
    depth: IntLike = _DEFAULT_DEPTH,
    gate_set: Sequence[str] | None = None,
    include_measurements: bool = False,
) -> Any:
    """Strategy yielding random :class:`qiskit.QuantumCircuit` instances.

    The circuit is constructed layer-by-layer, one gate per layer, by
    sampling a gate name from *gate_set* and then sampling whichever
    parameters that gate needs (qubit indices, rotation angle).

    Parameters
    ----------
    n_qubits
        Either an integer or a Hypothesis strategy producing positive
        integers. Defaults to ``integers(1, 5)``.
    depth
        Either an integer or a Hypothesis strategy producing positive
        integers (the number of gates placed in the circuit). Defaults
        to ``integers(1, 20)``.
    gate_set
        Gate vocabulary. ``None`` selects
        :data:`qtest.strategies.gates.DEFAULT_GATE_SET`. Two-qubit gates
        are automatically filtered out when the drawn ``n_qubits == 1``;
        if that filtering leaves the vocabulary empty (e.g. ``["cx"]``
        with ``n_qubits=1``), the strategy raises :class:`ValueError`.
    include_measurements
        When ``True`` the circuit is terminated with
        :meth:`QuantumCircuit.measure_all`, adding a barrier plus one
        measurement per qubit. Useful when the circuit will be fed to
        a sampling backend rather than a state-vector simulator.

    Returns
    -------
    QuantumCircuit
        A freshly-constructed Qiskit circuit.

    Notes
    -----
    Importing Qiskit is deferred to the first draw so that merely
    importing :mod:`qtest.strategies` does not force a Qiskit import
    (matches the lazy-import convention used elsewhere in qtest).

    Examples
    --------
    >>> from hypothesis import given, settings
    >>> from qtest.strategies import quantum_circuits
    >>> @given(qc=quantum_circuits(n_qubits=2, depth=5))
    ... @settings(max_examples=10)
    ... def test_no_classical_bits(qc):
    ...     assert qc.num_clbits == 0
    """
    from qiskit import QuantumCircuit

    n = _resolve_int(draw, n_qubits, name="n_qubits", minimum=1)
    d = _resolve_int(draw, depth, name="depth", minimum=1)

    vocabulary: tuple[str, ...] = DEFAULT_GATE_SET if gate_set is None else tuple(gate_set)
    if not vocabulary:
        raise ValueError("gate_set must be non-empty")

    if n == 1:
        vocabulary = tuple(g for g in vocabulary if g not in TWO_QUBIT_GATES)
        if not vocabulary:
            raise ValueError("gate_set contains only two-qubit gates but n_qubits == 1")

    qc = QuantumCircuit(n)
    gate_strategy = st.sampled_from(vocabulary)
    angle_strategy = st.floats(
        min_value=0.0,
        max_value=2.0 * math.pi,
        allow_nan=False,
        allow_infinity=False,
    )

    for _ in range(d):
        gate = draw(gate_strategy)
        if gate in TWO_QUBIT_GATES:
            # Two distinct qubit indices in [0, n). ``unique=True`` plus a
            # fixed size of 2 gives the desired pair, and shrinks toward
            # the (0, 1) pair.
            pair = draw(
                st.lists(
                    st.integers(min_value=0, max_value=n - 1),
                    min_size=2,
                    max_size=2,
                    unique=True,
                )
            )
            getattr(qc, gate)(pair[0], pair[1])
        elif gate in PARAMETRIC_GATES:
            angle = draw(angle_strategy)
            qubit = draw(st.integers(min_value=0, max_value=n - 1))
            getattr(qc, gate)(angle, qubit)
        else:
            qubit = draw(st.integers(min_value=0, max_value=n - 1))
            getattr(qc, gate)(qubit)

    if include_measurements:
        qc.measure_all()

    return qc


__all__ = ["quantum_circuits"]
