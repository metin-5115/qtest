"""Tests for :mod:`qtest.strategies.gates`."""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings

from qtest.strategies import pauli_strings, random_gates
from qtest.strategies.gates import DEFAULT_GATE_SET

_FAST_SETTINGS = settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


# --------------------------------------------------------------------------- #
# random_gates                                                                #
# --------------------------------------------------------------------------- #


@_FAST_SETTINGS
@given(name=random_gates())
def test_random_gates_default_returns_from_default_set(name: str) -> None:
    assert name in DEFAULT_GATE_SET


@_FAST_SETTINGS
@given(name=random_gates(["h", "x", "rx"]))
def test_random_gates_custom_set(name: str) -> None:
    assert name in {"h", "x", "rx"}


def test_random_gates_rejects_empty_set() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        random_gates([])


def test_random_gates_rejects_non_string_entries() -> None:
    with pytest.raises(ValueError, match="non-empty strings"):
        random_gates(["h", 1, "x"])  # type: ignore[list-item]


def test_random_gates_rejects_empty_string_entries() -> None:
    with pytest.raises(ValueError, match="non-empty strings"):
        random_gates(["h", "", "x"])


# --------------------------------------------------------------------------- #
# pauli_strings                                                               #
# --------------------------------------------------------------------------- #


@_FAST_SETTINGS
@given(p=pauli_strings(n_qubits=4))
def test_pauli_strings_length(p: str) -> None:
    assert len(p) == 4


@_FAST_SETTINGS
@given(p=pauli_strings(n_qubits=5))
def test_pauli_strings_alphabet(p: str) -> None:
    assert set(p) <= {"I", "X", "Y", "Z"}


@pytest.mark.parametrize("bad", [0, -1, 1.5, "3", True])
def test_pauli_strings_rejects_bad_n_qubits(bad: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        pauli_strings(bad)  # type: ignore[arg-type]


def test_pauli_strings_reachability() -> None:
    """Sampling many examples should explore the full 4-character alphabet."""
    seen: set[str] = set()

    @settings(max_examples=200, deadline=None)
    @given(p=pauli_strings(n_qubits=1))
    def collect(p: str) -> None:
        seen.add(p)

    collect()
    # Probability that any one of the four single-qubit Paulis is
    # *missing* after 200 uniform draws is ~ 4*(3/4)**200 < 1e-23.
    assert seen == {"I", "X", "Y", "Z"}
