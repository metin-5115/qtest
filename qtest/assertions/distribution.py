"""``assert_distribution_close`` — qtest's primary statistical assertion.

Compares the measurement distribution produced by a quantum circuit against
an expected probability distribution and raises :class:`AssertionError` with
a human-readable diagnostic when they diverge beyond the configured
tolerance.

Three statistical metrics are supported:

* ``"tv"`` — total variation distance. Tolerance is the maximum permitted
  distance. Lower tolerance = stricter test.
* ``"hellinger"`` — Hellinger distance. Same semantics as ``"tv"``.
* ``"chi_square"`` — Pearson :math:`\\chi^{2}` goodness-of-fit test. Here
  ``tolerance`` is interpreted as the **significance level** :math:`\\alpha`:
  the assertion fails when ``p_value < tolerance``. Lower tolerance = stricter.
"""

from __future__ import annotations

import math
from typing import Any

from qtest.backends import Backend
from qtest.backends.registry import get_backend
from qtest.config import _resolve_value, get_config
from qtest.metrics import (
    auto_tolerance,
    chi_square_test,
    hellinger_distance,
    total_variation_distance,
)

_VALID_METRICS = frozenset({"tv", "chi_square", "hellinger"})
_PROB_SUM_ATOL = 1e-6


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #


def assert_distribution_close(
    circuit: Any,
    expected: dict[str, float],
    shots: int | None = None,
    tolerance: float | None = None,
    metric: str | None = None,
    backend: Backend | None = None,
    seed: int | None = None,
    msg: str | None = None,
) -> None:
    """Assert that *circuit*'s measurement distribution is close to *expected*.

    Parameters
    ----------
    circuit
        A backend-native circuit object (e.g. ``qiskit.QuantumCircuit``).
        Must contain at least one measurement instruction.
    expected
        Reference probability distribution as ``{bitstring: probability}``.
        Must be non-empty, with values in ``[0, 1]`` summing to ``1``.
        Bitstrings absent from *expected* are taken to have probability 0.
    shots, tolerance, metric, backend, seed
        Per-call overrides for the corresponding global config defaults.
        ``None`` means "use config" — see :class:`qtest.config.QtestConfig`.
    msg
        Optional prefix prepended to the assertion failure message.

    Raises
    ------
    ValueError
        For malformed input (empty distribution, inconsistent bitstring
        lengths, probabilities outside ``[0, 1]``, no measurement in
        circuit, non-positive shots, etc.).
    AssertionError
        When the measured distribution diverges from *expected* beyond the
        configured tolerance.

    Examples
    --------
    >>> from qiskit import QuantumCircuit
    >>> qc = QuantumCircuit(2, 2)
    >>> qc.h(0); qc.cx(0, 1); qc.measure([0, 1], [0, 1])
    >>> assert_distribution_close(  # doctest: +SKIP
    ...     qc, {"00": 0.5, "11": 0.5}, shots=4000, seed=42
    ... )
    """
    shots = _resolve_value("default_shots", shots)
    metric = _resolve_value("statistical_metric", metric)
    seed = _resolve_value("default_seed", seed)

    cfg = get_config()
    if tolerance is None:
        if cfg.auto_tolerance and metric != "chi_square":
            tolerance = auto_tolerance(shots=shots)
        else:
            tolerance = cfg.default_tolerance

    if metric not in _VALID_METRICS:
        raise ValueError(f"Unknown metric {metric!r}. Must be one of {sorted(_VALID_METRICS)}.")
    if not isinstance(shots, int) or isinstance(shots, bool) or shots <= 0:
        raise ValueError(f"shots must be a positive integer, got {shots!r}")
    if not isinstance(tolerance, (int, float)) or not (0.0 <= float(tolerance) <= 1.0):
        raise ValueError(f"tolerance must be in [0, 1], got {tolerance!r}")

    expected_norm = _validate_expected(expected)

    if not _has_measurements(circuit):
        raise ValueError(
            "circuit has no measurement instructions; assert_distribution_close "
            "requires a circuit that produces classical bitstrings. Add e.g. "
            "circuit.measure_all() or specific measure() calls."
        )

    if backend is None:
        backend = get_backend(cfg.default_backend)

    raw_counts = backend.run_circuit(circuit, shots=shots, seed=seed)
    measured_counts = {k.replace(" ", ""): v for k, v in raw_counts.items()}
    measured_probs = _normalize_counts(measured_counts)

    _validate_bitstring_lengths(expected_norm, measured_probs)

    if metric == "chi_square":
        try:
            _stat, p_value = chi_square_test(measured_counts, expected_norm, shots=shots)
        except ValueError as exc:
            raise AssertionError(
                _format_failure_message(
                    circuit=circuit,
                    backend=backend,
                    shots=shots,
                    metric=metric,
                    tolerance=tolerance,
                    measured_distance=0.0,
                    distance_label="p-value",
                    expected=expected_norm,
                    measured=measured_probs,
                    user_msg=msg,
                    extra_note=f"chi-square rejected hypothesis outright: {exc}",
                )
            ) from None
        failed = p_value < tolerance
        reported_value = p_value
        distance_label = "p-value"
    else:
        all_keys = set(expected_norm) | set(measured_probs)
        p_padded = {k: expected_norm.get(k, 0.0) for k in all_keys}
        q_padded = {k: measured_probs.get(k, 0.0) for k in all_keys}
        distance_fn = total_variation_distance if metric == "tv" else hellinger_distance
        distance = distance_fn(p_padded, q_padded)
        failed = distance > tolerance
        reported_value = distance
        distance_label = f"{metric} distance"

    if failed:
        raise AssertionError(
            _format_failure_message(
                circuit=circuit,
                backend=backend,
                shots=shots,
                metric=metric,
                tolerance=tolerance,
                measured_distance=reported_value,
                distance_label=distance_label,
                expected=expected_norm,
                measured=measured_probs,
                user_msg=msg,
            )
        )


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _normalize_counts(counts: dict[str, int]) -> dict[str, float]:
    """Convert raw integer counts to a probability distribution."""
    total = sum(counts.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in counts.items()}


def _validate_expected(expected: dict[str, float]) -> dict[str, float]:
    """Validate *expected* and return a normalised copy (spaces stripped)."""
    if not isinstance(expected, dict):
        raise ValueError(f"expected must be a dict[str, float], got {type(expected).__name__}")
    if not expected:
        raise ValueError("expected must be a non-empty probability distribution")

    cleaned: dict[str, float] = {}
    for key, value in expected.items():
        if not isinstance(key, str):
            raise ValueError(f"expected keys must be strings, got key {key!r}")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"expected[{key!r}] = {value!r} is not a real number")
        if math.isnan(value) or math.isinf(value):
            raise ValueError(f"expected[{key!r}] = {value!r} is not finite")
        if value < 0.0 or value > 1.0:
            raise ValueError(f"expected[{key!r}] = {value} is outside [0, 1]")
        cleaned[key.replace(" ", "")] = float(value)

    total = sum(cleaned.values())
    if not math.isclose(total, 1.0, abs_tol=_PROB_SUM_ATOL):
        raise ValueError(
            f"expected probabilities must sum to 1 (within {_PROB_SUM_ATOL}); " f"got {total}"
        )
    return cleaned


def _has_measurements(circuit: Any) -> bool:
    """Return whether *circuit* contains at least one measurement instruction.

    Best-effort: uses Qiskit's ``count_ops()`` when available, returns ``True``
    (i.e. assume the caller knows what they're doing) for unknown circuit
    types so non-Qiskit / mock backends are not broken.
    """
    if hasattr(circuit, "count_ops"):
        try:
            ops = circuit.count_ops()
        except Exception:
            return True
        if isinstance(ops, dict):
            return int(ops.get("measure", 0)) > 0
    return True


def _validate_bitstring_lengths(expected: dict[str, float], measured: dict[str, float]) -> None:
    """Ensure all bitstrings in *expected* and *measured* have the same length."""
    lengths = {len(k) for k in expected}
    lengths.update(len(k) for k in measured)
    if len(lengths) > 1:
        raise ValueError(
            f"Inconsistent bitstring lengths in expected/measured: "
            f"{sorted(lengths)} — all bitstrings must have the same width."
        )


def _circuit_summary(circuit: Any) -> str:
    """Human-readable one-line description of *circuit* for error messages."""
    name = getattr(circuit, "name", None)
    n_qubits = getattr(circuit, "num_qubits", None)
    if name and n_qubits is not None:
        return f"{name} ({n_qubits} qubits)"
    if name:
        return str(name)
    if n_qubits is not None:
        return f"<{type(circuit).__name__} num_qubits={n_qubits}>"
    return f"<{type(circuit).__name__}>"


def _format_diff_table(expected: dict[str, float], measured: dict[str, float]) -> str:
    """Render an ``expected vs measured`` per-bitstring diff table."""
    keys = sorted(set(expected) | set(measured))
    lines: list[str] = []
    for key in keys:
        e = expected.get(key, 0.0)
        m = measured.get(key, 0.0)
        diff = m - e
        sign = "+" if diff >= 0 else "-"
        lines.append(
            f'    "{key}": expected {e:.3f}, measured {m:.3f}  ' f"(diff: {sign}{abs(diff):.3f})"
        )
    return "\n".join(lines)


def _suggest_diagnosis(expected: dict[str, float], measured: dict[str, float]) -> str:
    """Return a short, rule-based hint pointing at likely causes."""
    if not measured:
        return "Measured distribution is empty — backend returned no counts."

    extra = {k for k in measured if k not in expected and measured[k] > 0.0}
    if extra:
        sample = sorted(extra)[:3]
        return (
            f"Measured distribution contains unexpected outcomes (e.g. {sample}). "
            "Possible causes: wrong measurement basis, noise/decoherence, "
            "or an incorrect `expected` dict."
        )

    missing = {k for k in expected if expected[k] > 0.0 and measured.get(k, 0.0) == 0.0}
    if missing:
        sample = sorted(missing)[:3]
        return (
            f"Expected outcomes never observed (e.g. {sample}). "
            "Try increasing `shots` or check the circuit's gate sequence."
        )

    expected_mode = max(expected, key=lambda k: expected[k])
    measured_mode = max(measured, key=lambda k: measured[k])
    if expected_mode != measured_mode:
        return (
            f"Peak shifted: expected mode '{expected_mode}', measured mode "
            f"'{measured_mode}'. Check qubit ordering or gate parameters."
        )

    return (
        "Distributions have the right shape but differ in magnitude. "
        "Try increasing `shots` to reduce sampling noise, or relax `tolerance`."
    )


def _format_failure_message(
    *,
    circuit: Any,
    backend: Backend,
    shots: int,
    metric: str,
    tolerance: float,
    measured_distance: float,
    distance_label: str,
    expected: dict[str, float],
    measured: dict[str, float],
    user_msg: str | None,
    extra_note: str | None = None,
) -> str:
    """Build the multi-line ``AssertionError`` payload."""
    cfg = get_config()
    lines: list[str] = []

    if user_msg:
        lines.append(user_msg)
        lines.append("")

    lines.append("Distribution mismatch beyond tolerance")
    lines.append("")
    lines.append(f"  Circuit: {_circuit_summary(circuit)}")
    lines.append(f"  Backend: {backend.name}")
    lines.append(f"  Shots: {shots}")
    lines.append(f"  Metric: {metric}")
    lines.append(f"  Tolerance: {tolerance}")
    lines.append(f"  Measured {distance_label}: {measured_distance:.6f}")
    lines.append("")
    lines.append("  Expected vs Measured:")
    lines.append(_format_diff_table(expected, measured))

    if extra_note:
        lines.append("")
        lines.append(f"  Note: {extra_note}")

    if cfg.verbose_failures:
        suggestion = _suggest_diagnosis(expected, measured)
        if suggestion:
            lines.append("")
            lines.append(f"  Suggestion: {suggestion}")

    return "\n".join(lines)
