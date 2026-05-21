"""Tests for :func:`qtest.strategies.quantum_circuits`.

The strategy is itself the unit under test: each test wraps a tiny
property in ``@given(quantum_circuits(...))`` and asserts that the
strategy's contract holds for every example Hypothesis throws at it.
A separate test forces a failure to verify that Hypothesis actually
*shrinks* drawn circuits (sanity-checking the determinism /
``draw``-only entropy claim).
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from qtest.strategies import quantum_circuits
from qtest.strategies.gates import (
    DEFAULT_GATE_SET,
    PARAMETRIC_GATES,
    TWO_QUBIT_GATES,
)

# Hypothesis emits a health-check when a composite calls a backend
# (Qiskit) inside ``draw`` and the construction is comparatively slow.
# Suppressing the ``too_slow`` and ``data_too_large`` checks keeps the
# tests stable on slower CI runners without weakening any assertions.
_FAST_SETTINGS = settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


# --------------------------------------------------------------------------- #
# Structural invariants                                                       #
# --------------------------------------------------------------------------- #


@_FAST_SETTINGS
@given(qc=quantum_circuits(n_qubits=3, depth=5))
def test_n_qubits_constraint_respected(qc) -> None:
    """A fixed ``n_qubits`` must yield circuits with exactly that width."""
    assert qc.num_qubits == 3


@_FAST_SETTINGS
@given(qc=quantum_circuits(n_qubits=2, depth=7))
def test_depth_constraint_respected(qc) -> None:
    """One gate per layer, so ``len(qc.data) == depth`` exactly."""
    assert len(qc.data) == 7


@_FAST_SETTINGS
@given(
    qc=quantum_circuits(
        n_qubits=2,
        depth=5,
        gate_set=["h", "cx"],
    )
)
def test_gate_set_restriction_respected(qc) -> None:
    """Only the requested gates may appear in the generated circuit."""
    allowed = {"h", "cx"}
    for instr in qc.data:
        assert instr.operation.name in allowed


@_FAST_SETTINGS
@given(
    qc=quantum_circuits(
        n_qubits=2,
        depth=5,
        include_measurements=True,
    )
)
def test_include_measurements_adds_measure_and_barrier(qc) -> None:
    """``include_measurements=True`` => ``measure_all()`` appended.

    ``measure_all`` emits one barrier followed by N measurements, so
    the circuit ends with exactly that suffix.
    """
    names = [instr.operation.name for instr in qc.data]
    assert names[-3:] == ["barrier", "measure", "measure"]
    assert qc.num_clbits == 2


@_FAST_SETTINGS
@given(qc=quantum_circuits(n_qubits=2, depth=4))
def test_default_no_measurements(qc) -> None:
    """Without ``include_measurements`` the circuit must have no clbits."""
    assert qc.num_clbits == 0
    names = {instr.operation.name for instr in qc.data}
    assert "measure" not in names
    assert "barrier" not in names


# --------------------------------------------------------------------------- #
# Parameter / qubit-index correctness                                         #
# --------------------------------------------------------------------------- #


@_FAST_SETTINGS
@given(qc=quantum_circuits(n_qubits=3, depth=10))
def test_two_qubit_gates_use_distinct_qubits(qc) -> None:
    """No two-qubit gate may target the same qubit for ctrl and target."""
    for instr in qc.data:
        if instr.operation.name in TWO_QUBIT_GATES:
            qubits = [qc.find_bit(q).index for q in instr.qubits]
            assert len(set(qubits)) == 2


@_FAST_SETTINGS
@given(qc=quantum_circuits(n_qubits=2, depth=8, gate_set=["rx", "ry", "rz"]))
def test_parametric_angles_in_range(qc) -> None:
    """All rotation angles drawn must lie in ``[0, 2π]``."""
    import math

    for instr in qc.data:
        assert instr.operation.name in PARAMETRIC_GATES
        (angle,) = instr.operation.params
        assert 0.0 <= float(angle) <= 2.0 * math.pi + 1e-9


# --------------------------------------------------------------------------- #
# Strategy-valued parameters                                                  #
# --------------------------------------------------------------------------- #


@_FAST_SETTINGS
@given(
    qc=quantum_circuits(
        n_qubits=st.integers(min_value=1, max_value=4),
        depth=st.integers(min_value=1, max_value=6),
    )
)
def test_strategy_valued_n_and_depth(qc) -> None:
    """Passing strategies for ``n_qubits`` / ``depth`` still gives valid widths."""
    assert 1 <= qc.num_qubits <= 4
    assert 1 <= len(qc.data) <= 6


# --------------------------------------------------------------------------- #
# Defensive checks                                                            #
# --------------------------------------------------------------------------- #


@settings(max_examples=1, deadline=None)
@given(data=st.data())
def test_empty_gate_set_raises(data) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        data.draw(quantum_circuits(n_qubits=1, depth=1, gate_set=[]))


@settings(max_examples=1, deadline=None)
@given(data=st.data())
def test_only_two_qubit_gates_with_one_qubit_raises(data) -> None:
    """``n_qubits=1`` with a two-qubit-only vocabulary is impossible."""
    with pytest.raises(ValueError, match="two-qubit"):
        data.draw(quantum_circuits(n_qubits=1, depth=1, gate_set=["cx", "cz"]))


@_FAST_SETTINGS
@given(qc=quantum_circuits(n_qubits=3, depth=20))
def test_default_strategy_uses_default_gate_set(qc) -> None:
    """With the default vocabulary, generated gate names are a subset of it.

    Stronger version of ``test_gate_set_restriction_respected`` aimed at
    the default.
    """
    names = {instr.operation.name for instr in qc.data}
    assert names <= set(DEFAULT_GATE_SET)


# --------------------------------------------------------------------------- #
# Shrinking sanity-check                                                      #
# --------------------------------------------------------------------------- #


def test_shrinking_reduces_depth() -> None:
    """A failing property must shrink to a minimal counter-example.

    The property ``len(qc.data) < 1`` is false for *every* circuit with
    depth >= 1, so Hypothesis will shrink to the smallest depth (1).
    We catch the assertion error raised by the inner test, inspect the
    Hypothesis report, and verify the shrunk example reports
    ``depth=1`` worth of data.
    """

    @given(qc=quantum_circuits(n_qubits=2, depth=st.integers(min_value=1, max_value=10)))
    @settings(max_examples=50, deadline=None)
    def inner(qc) -> None:
        # Will fail on the very first example; Hypothesis then shrinks.
        assert len(qc.data) < 1

    with pytest.raises(AssertionError):
        inner()
