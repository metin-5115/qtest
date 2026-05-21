"""Built-in pytest fixtures (and plain factories) for common quantum circuits.

To use the **fixtures** in your tests, add the relevant module(s) to
``pytest_plugins`` in your top-level ``conftest.py``::

    # tests/conftest.py
    pytest_plugins = [
        "qtest.fixtures.common_states",
        "qtest.fixtures.common_gates",
    ]

The same builders are also exposed as **plain functions** here, usable
outside of pytest::

    from qtest.fixtures import bell_circuit, ghz_circuit
    qc = bell_circuit()
    ghz5 = ghz_circuit(5)

Imports are **lazy** (PEP 562 ``__getattr__``): nothing is loaded until the
attribute is actually requested. This lets the pytest plugin loader register
``common_states`` and ``common_gates`` as plugins independently, without one
being imported as a side effect of the other (which would suppress assert
rewriting in the second).
"""

from typing import Any

__all__ = [
    "bell_circuit",
    "ghz_circuit",
    "hadamards",
    "minus_circuit",
    "plus_circuit",
    "random_clifford_circuit",
    "w_circuit",
]


_STATE_FACTORIES = frozenset(
    {"bell_circuit", "ghz_circuit", "minus_circuit", "plus_circuit", "w_circuit"}
)
_GATE_FACTORIES = frozenset({"hadamards", "random_clifford_circuit"})


def __getattr__(name: str) -> Any:
    if name in _STATE_FACTORIES:
        from qtest.fixtures import common_states

        return getattr(common_states, name)
    if name in _GATE_FACTORIES:
        from qtest.fixtures import common_gates

        return getattr(common_gates, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
