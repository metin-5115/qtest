"""Hypothesis strategies for individual quantum gates and Pauli strings.

The strategies here are the small building blocks used by
:mod:`qtest.strategies.circuits`. They are also useful on their own when
a property test only needs gate *names* or a Pauli operator description
rather than a full circuit.

Design notes
------------
* :func:`random_gates` returns the *name* of a gate as a string. Callers
  remain in control of which qubits the gate acts on and what arguments
  it takes. Returning names (rather than fully-bound gate operations)
  keeps the strategy independent of any particular circuit library and
  preserves Hypothesis's shrinking — the example list shrinks to early
  alphabetical entries.
* :func:`pauli_strings` returns a plain ``str`` so it can be used as a
  dictionary key, printed in failure messages, etc. The string reads
  little-endian by qubit index: ``"XIZY"`` means qubit 0 -> X, qubit 1 ->
  I, qubit 2 -> Z, qubit 3 -> Y.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

from hypothesis import strategies as st

# --------------------------------------------------------------------------- #
# Default gate set                                                            #
# --------------------------------------------------------------------------- #

#: Default gate vocabulary used by :func:`random_gates` and
#: :func:`qtest.strategies.quantum_circuits` when the caller passes no
#: explicit ``gate_set``. Covers a mix of Clifford, T, and parametric
#: rotations plus the two most common two-qubit entanglers.
DEFAULT_GATE_SET: tuple[str, ...] = (
    "h",
    "x",
    "y",
    "z",
    "s",
    "t",
    "cx",
    "cz",
    "rx",
    "ry",
    "rz",
)

#: Gates that take a single real angle parameter.
PARAMETRIC_GATES: frozenset[str] = frozenset({"rx", "ry", "rz", "p"})

#: Gates that act on two qubits.
TWO_QUBIT_GATES: frozenset[str] = frozenset({"cx", "cz", "swap", "iswap"})

#: Pauli alphabet for :func:`pauli_strings`.
_PAULI_ALPHABET: tuple[str, ...] = ("I", "X", "Y", "Z")


# --------------------------------------------------------------------------- #
# Public strategies                                                           #
# --------------------------------------------------------------------------- #


def random_gates(gate_set: Optional[Sequence[str]] = None) -> st.SearchStrategy[str]:
    """Strategy that draws a single gate **name** from *gate_set*.

    Parameters
    ----------
    gate_set
        Iterable of gate names to choose from. ``None`` uses
        :data:`DEFAULT_GATE_SET`.

    Returns
    -------
    SearchStrategy[str]
        Strategy yielding one of the names in *gate_set*.

    Examples
    --------
    >>> from hypothesis import given
    >>> from qtest.strategies import random_gates
    >>> @given(name=random_gates(["h", "x", "cx"]))
    ... def test_known_gate(name):
    ...     assert name in {"h", "x", "cx"}
    """
    if gate_set is None:
        choices: tuple[str, ...] = DEFAULT_GATE_SET
    else:
        choices = tuple(gate_set)
        if not choices:
            raise ValueError("gate_set must be non-empty")
        if not all(isinstance(g, str) and g for g in choices):
            raise ValueError("gate_set must contain non-empty strings")
    return st.sampled_from(choices)


def pauli_strings(n_qubits: int) -> st.SearchStrategy[str]:
    """Strategy that draws an *n_qubits*-character Pauli string.

    The alphabet is ``{"I", "X", "Y", "Z"}``. Each position is drawn
    independently, so all :math:`4^{n}` strings are reachable with
    uniform probability under Hypothesis's default sampling.

    Parameters
    ----------
    n_qubits
        Length of the returned string. Must be a positive integer.

    Returns
    -------
    SearchStrategy[str]
        Strategy yielding strings of length ``n_qubits`` over
        ``{"I","X","Y","Z"}``.

    Examples
    --------
    >>> from hypothesis import given
    >>> from qtest.strategies import pauli_strings
    >>> @given(p=pauli_strings(3))
    ... def test_len(p):
    ...     assert len(p) == 3 and set(p) <= {"I","X","Y","Z"}
    """
    if not isinstance(n_qubits, int) or isinstance(n_qubits, bool) or n_qubits < 1:
        raise ValueError(f"n_qubits must be a positive integer, got {n_qubits!r}")
    return st.lists(
        st.sampled_from(_PAULI_ALPHABET),
        min_size=n_qubits,
        max_size=n_qubits,
    ).map("".join)


__all__ = [
    "DEFAULT_GATE_SET",
    "PARAMETRIC_GATES",
    "TWO_QUBIT_GATES",
    "pauli_strings",
    "random_gates",
]
