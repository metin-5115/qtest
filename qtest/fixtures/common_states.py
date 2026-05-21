"""Pytest fixtures (and plain factory functions) for common quantum states.

Each fixture / factory returns a :class:`qiskit.QuantumCircuit` that prepares
the named state starting from :math:`|0\\rangle^{\\otimes n}`. Circuits are
unitary-only (no measurement / reset); add ``qc.measure_all()`` in your test
when sampling.

The same builders are exposed twice:

* As **pytest fixtures** (``bell_state``, ``ghz_state``, ``plus_state``,
  ``ghz_3`` …) — request them in test signatures.
* As **plain functions** (``bell_circuit``, ``ghz_circuit(n)``,
  ``plus_circuit`` …) — call them directly outside of pytest tests.

For pytest discovery, list this module in the consumer's
``pytest_plugins`` or import individual fixtures into your ``conftest.py``::

    # tests/conftest.py
    pytest_plugins = ["qtest.fixtures.common_states"]

Examples
--------
>>> # In a pytest test:
>>> def test_bell(bell_state, qiskit_backend):  # doctest: +SKIP
...     assert_state_close(bell_state, "bell")

>>> # As a plain function:
>>> from qtest.fixtures.common_states import bell_circuit  # doctest: +SKIP
>>> qc = bell_circuit()
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pytest

# --------------------------------------------------------------------------- #
# Plain factory functions                                                     #
# --------------------------------------------------------------------------- #


def bell_circuit() -> Any:
    """Return a circuit preparing the Bell state :math:`(|00\\rangle + |11\\rangle)/\\sqrt{2}`."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2, name="bell")
    qc.h(0)
    qc.cx(0, 1)
    return qc


def plus_circuit() -> Any:
    """Return a 1-qubit circuit preparing :math:`|+\\rangle = (|0\\rangle + |1\\rangle)/\\sqrt{2}`."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(1, name="plus")
    qc.h(0)
    return qc


def minus_circuit() -> Any:
    """Return a 1-qubit circuit preparing :math:`|-\\rangle = (|0\\rangle - |1\\rangle)/\\sqrt{2}`."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(1, name="minus")
    qc.x(0)
    qc.h(0)
    return qc


def ghz_circuit(n: int) -> Any:
    """Return an n-qubit GHZ-state preparation circuit.

    Builds :math:`(|0\\rangle^{\\otimes n} + |1\\rangle^{\\otimes n}) / \\sqrt{2}`
    via the standard ``H + (n-1) × CX`` recipe.

    Parameters
    ----------
    n
        Qubit count (``n >= 1``).
    """
    from qiskit import QuantumCircuit

    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    qc = QuantumCircuit(n, name=f"ghz_{n}")
    qc.h(0)
    for k in range(1, n):
        qc.cx(0, k)
    return qc


def w_circuit(n: int) -> Any:
    """Return an n-qubit W-state preparation circuit.

    For ``n == 1`` returns ``X|0> = |1>``; for ``n == 2`` uses the
    closed-form ``H + CX + X`` recipe; for ``n >= 3`` it appends a unitary
    :class:`qiskit.circuit.library.StatePreparation` block with the
    target amplitudes
    :math:`(|10\\dots0\\rangle + |01\\dots0\\rangle + \\dots) / \\sqrt{n}`.

    Parameters
    ----------
    n
        Qubit count (``n >= 1``).
    """
    from qiskit import QuantumCircuit

    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}")

    if n == 1:
        qc = QuantumCircuit(1, name="w_1")
        qc.x(0)
        return qc

    if n == 2:
        # Hand-coded: |00> -H(0)-> (|00>+|01>)/√2 -CX(0,1)-> (|00>+|11>)/√2
        # -X(0)-> (|01>+|10>)/√2 = |W_2>.
        qc = QuantumCircuit(2, name="w_2")
        qc.h(0)
        qc.cx(0, 1)
        qc.x(0)
        return qc

    # n >= 3: build the target amplitude vector and use a unitary
    # StatePreparation block.
    from qiskit.circuit.library import StatePreparation

    dim = 1 << n
    vec = np.zeros(dim, dtype=complex)
    for k in range(n):
        vec[1 << k] = 1.0
    vec /= np.sqrt(n)

    qc = QuantumCircuit(n, name=f"w_{n}")
    qc.append(StatePreparation(vec), range(n))
    return qc


# --------------------------------------------------------------------------- #
# Pytest fixtures                                                             #
# --------------------------------------------------------------------------- #


@pytest.fixture
def bell_state() -> Any:
    """Yield a fresh Bell-state preparation circuit per test.

    Example::

        def test_bell(bell_state):
            assert_state_close(bell_state, "bell")
    """
    return bell_circuit()


@pytest.fixture
def plus_state() -> Any:
    """Yield a fresh :math:`|+\\rangle` preparation circuit per test."""
    return plus_circuit()


@pytest.fixture
def minus_state() -> Any:
    """Yield a fresh :math:`|-\\rangle` preparation circuit per test."""
    return minus_circuit()


@pytest.fixture
def ghz_state() -> Callable[[int], Any]:
    """Factory fixture yielding ``n -> QuantumCircuit`` for GHZ-n states.

    Example::

        def test_ghz_5(ghz_state):
            assert_state_close(ghz_state(5), "ghz_5")
    """
    return ghz_circuit


@pytest.fixture
def w_state() -> Callable[[int], Any]:
    """Factory fixture yielding ``n -> QuantumCircuit`` for W-n states.

    Example::

        def test_w_4(w_state):
            assert_state_close(w_state(4), "w_4")
    """
    return w_circuit


@pytest.fixture
def ghz_3() -> Any:
    """Shortcut for a 3-qubit GHZ preparation circuit."""
    return ghz_circuit(3)


@pytest.fixture
def ghz_4() -> Any:
    """Shortcut for a 4-qubit GHZ preparation circuit."""
    return ghz_circuit(4)


@pytest.fixture
def ghz_5() -> Any:
    """Shortcut for a 5-qubit GHZ preparation circuit."""
    return ghz_circuit(5)
