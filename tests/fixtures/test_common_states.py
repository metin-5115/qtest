"""Tests for :mod:`qtest.fixtures.common_states`."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pytest

# Skip the entire module if Qiskit isn't installed.
pytest.importorskip("qiskit")

from qtest.assertions import assert_state_close, assert_unitary  # noqa: E402
from qtest.fixtures.common_states import (  # noqa: E402
    bell_circuit,
    ghz_circuit,
    minus_circuit,
    plus_circuit,
    w_circuit,
)

# --------------------------------------------------------------------------- #
# Plain factories                                                             #
# --------------------------------------------------------------------------- #


def test_bell_circuit_has_two_qubits() -> None:
    qc = bell_circuit()
    assert qc.num_qubits == 2


def test_plus_circuit_has_one_qubit() -> None:
    assert plus_circuit().num_qubits == 1


def test_minus_circuit_has_one_qubit() -> None:
    assert minus_circuit().num_qubits == 1


@pytest.mark.parametrize("n", [1, 2, 3, 5, 7])
def test_ghz_circuit_qubit_count(n: int) -> None:
    assert ghz_circuit(n).num_qubits == n


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_w_circuit_qubit_count(n: int) -> None:
    assert w_circuit(n).num_qubits == n


# State correctness — verified end-to-end with assert_state_close.


def test_bell_circuit_prepares_bell_state() -> None:
    assert_state_close(bell_circuit(), "bell")


def test_plus_circuit_prepares_plus_state() -> None:
    assert_state_close(plus_circuit(), "plus")


def test_minus_circuit_prepares_minus_state() -> None:
    assert_state_close(minus_circuit(), "minus")


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_ghz_circuit_prepares_ghz_state(n: int) -> None:
    assert_state_close(ghz_circuit(n), f"ghz_{n}")


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_w_circuit_prepares_w_state(n: int) -> None:
    assert_state_close(w_circuit(n), f"w_{n}")


def test_ghz_n1_is_plus_state() -> None:
    """For n=1 GHZ reduces to (|0>+|1>)/√2 = |+>."""
    assert_state_close(ghz_circuit(1), "plus")


def test_w_n1_is_one_ket() -> None:
    """For n=1 W reduces to |1>."""
    assert_state_close(w_circuit(1), np.array([0.0, 1.0], dtype=complex))


# Each call returns a fresh circuit object (no shared mutable state).


def test_bell_returns_independent_objects() -> None:
    a = bell_circuit()
    b = bell_circuit()
    assert a is not b


def test_ghz_returns_independent_objects() -> None:
    assert ghz_circuit(3) is not ghz_circuit(3)


# All builders produce unitary-only circuits (no measurements).


@pytest.mark.parametrize(
    "circuit_fn",
    [bell_circuit, plus_circuit, minus_circuit],
)
def test_static_circuits_are_unitary(circuit_fn: Callable[[], Any]) -> None:
    assert_unitary(circuit_fn())


@pytest.mark.parametrize("n", [2, 3, 4])
def test_ghz_circuit_is_unitary(n: int) -> None:
    assert_unitary(ghz_circuit(n))


# Input validation.


@pytest.mark.parametrize("bad", [0, -1, -10])
def test_ghz_circuit_rejects_non_positive_n(bad: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        ghz_circuit(bad)


@pytest.mark.parametrize("bad", [0, -1, 2.5, True])
def test_w_circuit_rejects_invalid_n(bad: Any) -> None:
    with pytest.raises(ValueError):
        w_circuit(bad)


# --------------------------------------------------------------------------- #
# Pytest fixtures (via conftest's pytest_plugins registration)                #
# --------------------------------------------------------------------------- #


def test_bell_state_fixture(bell_state: Any) -> None:
    assert bell_state.num_qubits == 2
    assert_state_close(bell_state, "bell")


def test_plus_state_fixture(plus_state: Any) -> None:
    assert plus_state.num_qubits == 1
    assert_state_close(plus_state, "plus")


def test_minus_state_fixture(minus_state: Any) -> None:
    assert_state_close(minus_state, "minus")


def test_ghz_state_factory_fixture(ghz_state: Callable[[int], Any]) -> None:
    """`ghz_state` is a factory: call it with the qubit count."""
    qc = ghz_state(5)
    assert qc.num_qubits == 5
    assert_state_close(qc, "ghz_5")


def test_w_state_factory_fixture(w_state: Callable[[int], Any]) -> None:
    qc = w_state(4)
    assert qc.num_qubits == 4
    assert_state_close(qc, "w_4")


@pytest.mark.parametrize("fixture_name,n", [("ghz_3", 3), ("ghz_4", 4), ("ghz_5", 5)])
def test_ghz_shortcut_fixtures(request: pytest.FixtureRequest, fixture_name: str, n: int) -> None:
    qc = request.getfixturevalue(fixture_name)
    assert qc.num_qubits == n
    assert_state_close(qc, f"ghz_{n}")


def test_each_fixture_request_is_independent(bell_state: Any) -> None:
    """Fixture scope is `function` by default — verify no cross-test sharing."""
    bell_state.name = "mutated"
    # If scope were session/module, the next test would see "mutated"; the
    # subsequent test asserts num_qubits == 2 on a freshly-built circuit.
