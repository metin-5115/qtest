"""Resource / cost assertions — guard a circuit's *structure*, not its output.

Where :func:`qtest.assert_circuit_equivalent` answers "does the transpiled
circuit still compute the right thing?", these assertions answer the
complementary question: "did it get *cheaper*?". They check the quantities an
optimiser or transpiler is supposed to reduce — circuit depth, total gate
count, two-qubit-gate count, and T-count — and are the natural way to lock in
an optimisation and catch regressions.

All checks are pure static analysis (no simulation), so they are fast and work
on circuits of any width. Metrics are extracted via
:meth:`Backend.get_resources`; backends that cannot introspect their native
circuit type raise :class:`NotImplementedError`.

Gate-name-based checks (:func:`assert_max_gate_count` with a ``gate`` filter,
and :func:`assert_max_t_count`) rely on backend gate naming, which qtest
normalises to lowercase (e.g. ``"t"``, ``"cx"``, ``"h"``).
"""

from __future__ import annotations

from typing import Any

from qtest.backends import Backend
from qtest.backends.base import CircuitResources
from qtest.backends.registry import get_backend
from qtest.config import get_config

__all__ = [
    "assert_max_depth",
    "assert_max_gate_count",
    "assert_max_t_count",
    "assert_max_two_qubit_count",
]


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #


def assert_max_depth(
    circuit: Any,
    max_depth: int,
    backend: Backend | None = None,
    msg: str | None = None,
) -> None:
    """Assert that *circuit* has depth at most *max_depth*.

    Parameters
    ----------
    circuit
        A circuit object native to the chosen backend.
    max_depth
        Maximum permitted circuit depth (non-negative integer).
    backend
        Backend used to analyse the circuit. Defaults to the configured one.
    msg
        Optional prefix prepended to the assertion failure message.

    Raises
    ------
    ValueError
        If ``max_depth`` is not a non-negative integer.
    AssertionError
        If the circuit's depth exceeds ``max_depth``.
    """
    _validate_limit(max_depth, "max_depth")
    res = _resources(circuit, backend)
    if res.depth <= max_depth:
        return
    raise AssertionError(
        _format_failure(
            "Circuit depth exceeds the allowed maximum",
            metric="depth",
            actual=res.depth,
            limit=max_depth,
            res=res,
            user_msg=msg,
        )
    )


def assert_max_gate_count(
    circuit: Any,
    max_count: int,
    gate: str | None = None,
    backend: Backend | None = None,
    msg: str | None = None,
) -> None:
    """Assert that *circuit* contains at most *max_count* gates.

    Parameters
    ----------
    circuit
        A circuit object native to the chosen backend.
    max_count
        Maximum permitted gate count (non-negative integer).
    gate
        If given, count only gates with this (lowercased) name, e.g.
        ``"cx"`` or ``"h"``. If ``None`` (the default), count every operation.
    backend
        Backend used to analyse the circuit. Defaults to the configured one.
    msg
        Optional prefix prepended to the assertion failure message.

    Raises
    ------
    ValueError
        If ``max_count`` is not a non-negative integer, or ``gate`` is not a
        non-empty string / ``None``.
    AssertionError
        If the (optionally filtered) gate count exceeds ``max_count``.
    """
    _validate_limit(max_count, "max_count")
    if gate is not None and (not isinstance(gate, str) or not gate):
        raise ValueError(f"gate must be a non-empty string or None, got {gate!r}")

    res = _resources(circuit, backend)
    if gate is None:
        actual = res.size
        metric = "total gate count"
    else:
        key = gate.lower()
        actual = res.gate_counts.get(key, 0)
        metric = f"{key!r} gate count"

    if actual <= max_count:
        return
    raise AssertionError(
        _format_failure(
            "Circuit gate count exceeds the allowed maximum",
            metric=metric,
            actual=actual,
            limit=max_count,
            res=res,
            user_msg=msg,
        )
    )


def assert_max_two_qubit_count(
    circuit: Any,
    max_count: int,
    backend: Backend | None = None,
    msg: str | None = None,
) -> None:
    """Assert that *circuit* contains at most *max_count* two-qubit gates.

    Two-qubit gates dominate error rates on real hardware, so this is one of
    the most useful regression guards after a transpilation pass.

    Raises
    ------
    ValueError
        If ``max_count`` is not a non-negative integer.
    AssertionError
        If the two-qubit-gate count exceeds ``max_count``.
    """
    _validate_limit(max_count, "max_count")
    res = _resources(circuit, backend)
    if res.two_qubit_count <= max_count:
        return
    raise AssertionError(
        _format_failure(
            "Circuit two-qubit-gate count exceeds the allowed maximum",
            metric="two-qubit-gate count",
            actual=res.two_qubit_count,
            limit=max_count,
            res=res,
            user_msg=msg,
        )
    )


def assert_max_t_count(
    circuit: Any,
    max_count: int,
    backend: Backend | None = None,
    msg: str | None = None,
) -> None:
    """Assert that *circuit* uses at most *max_count* T / T-dagger gates.

    T-count is the dominant cost metric for fault-tolerant (surface-code)
    execution, where T gates require expensive magic-state distillation.

    Raises
    ------
    ValueError
        If ``max_count`` is not a non-negative integer.
    AssertionError
        If the T-count exceeds ``max_count``.
    """
    _validate_limit(max_count, "max_count")
    res = _resources(circuit, backend)
    if res.t_count <= max_count:
        return
    raise AssertionError(
        _format_failure(
            "Circuit T-count exceeds the allowed maximum",
            metric="T-count",
            actual=res.t_count,
            limit=max_count,
            res=res,
            user_msg=msg,
        )
    )


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _validate_limit(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")


def _resources(circuit: Any, backend: Backend | None) -> CircuitResources:
    if circuit is None:
        raise ValueError("circuit must not be None")
    if backend is None:
        backend = get_backend(get_config().default_backend)
    return backend.get_resources(circuit)


def _format_failure(
    headline: str,
    *,
    metric: str,
    actual: int,
    limit: int,
    res: CircuitResources,
    user_msg: str | None,
) -> str:
    lines: list[str] = []
    if user_msg:
        lines.append(user_msg)
        lines.append("")

    lines.append(headline)
    lines.append("")
    lines.append(f"  {metric}: {actual}  (limit: {limit}, over by {actual - limit})")
    lines.append("")
    lines.append("  Circuit resources:")
    lines.append(f"    qubits:           {res.num_qubits}")
    lines.append(f"    depth:            {res.depth}")
    lines.append(f"    total gates:      {res.size}")
    lines.append(f"    two-qubit gates:  {res.two_qubit_count}")
    lines.append(f"    T-count:          {res.t_count}")
    if res.gate_counts:
        breakdown = ", ".join(f"{name}={count}" for name, count in sorted(res.gate_counts.items()))
        lines.append(f"    by gate:          {breakdown}")
    return "\n".join(lines)
