"""``assert_measurement_probabilities`` — marginal-distribution assertion.

Tests the marginal measurement distribution over a *subset* of the measured
bits, rather than the full joint distribution. Useful when only some qubits
carry the result of interest (e.g. an ancilla-assisted computation where you
only care about the data register).

Bit selection is by **position in the measured bitstring** as returned by the
backend (index ``0`` = leftmost character). This keeps the marginalisation
backend-agnostic and free of endianness guesswork — the positions line up with
the keys you already see from :func:`assert_distribution_close` diagnostics.
"""

from __future__ import annotations

from typing import Any

from qtest._report import record_distance
from qtest.assertions.distribution import _normalize_counts, _validate_expected
from qtest.backends import Backend
from qtest.backends.registry import get_backend
from qtest.config import _resolve_value, get_config
from qtest.metrics import chi_square_test, hellinger_distance, total_variation_distance

_VALID_METRICS = frozenset({"tv", "chi_square", "hellinger"})


def assert_measurement_probabilities(
    circuit: Any,
    expected: dict[str, float],
    bit_positions: list[int],
    shots: int | None = None,
    tolerance: float | None = None,
    metric: str | None = None,
    backend: Backend | None = None,
    seed: int | None = None,
    noise_model: Any | None = None,
    msg: str | None = None,
) -> None:
    """Assert the marginal distribution over *bit_positions* matches *expected*.

    Parameters
    ----------
    circuit
        A backend-native circuit with measurements.
    expected
        Reference marginal distribution ``{bitstring: probability}`` whose keys
        have length ``len(bit_positions)``.
    bit_positions
        Positions (0-based, from the left) within the measured bitstring to
        keep. Order matters: it defines the order of characters in the marginal
        keys.
    shots, tolerance, metric, backend, seed, noise_model
        Per-call overrides for the corresponding global config defaults.
    msg
        Optional prefix prepended to the assertion failure message.

    Raises
    ------
    ValueError
        For malformed input (empty positions, out-of-range index, marginal-key
        width mismatch, unknown metric, ...).
    AssertionError
        When the measured marginal diverges from *expected* beyond tolerance.
    """
    shots = _resolve_value("default_shots", shots)
    metric = _resolve_value("statistical_metric", metric)
    seed = _resolve_value("default_seed", seed)
    cfg = get_config()
    tolerance = cfg.default_tolerance if tolerance is None else tolerance

    if metric not in _VALID_METRICS:
        raise ValueError(f"Unknown metric {metric!r}. Must be one of {sorted(_VALID_METRICS)}.")
    if not bit_positions:
        raise ValueError("bit_positions must be a non-empty list of bit indices.")

    expected_norm = _validate_expected(expected)
    width = len(bit_positions)
    if any(len(k) != width for k in expected_norm):
        raise ValueError(f"expected keys must all have length {width} (one char per bit position).")

    if backend is None:
        backend = get_backend(cfg.default_backend)

    run_kwargs: dict[str, Any] = {"shots": shots, "seed": seed}
    if noise_model is not None:
        run_kwargs["noise_model"] = noise_model
    raw_counts = backend.run_circuit(circuit, **run_kwargs)
    full_counts = {k.replace(" ", ""): v for k, v in raw_counts.items()}

    marginal_counts = _marginalize(full_counts, bit_positions)
    measured = _normalize_counts(marginal_counts)

    if metric == "chi_square":
        try:
            _stat, p_value = chi_square_test(marginal_counts, expected_norm, shots=shots)
        except ValueError as exc:
            raise AssertionError(
                _format_failure(
                    circuit,
                    backend,
                    bit_positions,
                    metric,
                    tolerance,
                    0.0,
                    "p-value",
                    expected_norm,
                    measured,
                    msg,
                    extra_note=f"chi-square rejected hypothesis outright: {exc}",
                )
            ) from None
        failed = p_value < tolerance
        reported, label = p_value, "p-value"
    else:
        all_keys = set(expected_norm) | set(measured)
        p_padded = {k: expected_norm.get(k, 0.0) for k in all_keys}
        q_padded = {k: measured.get(k, 0.0) for k in all_keys}
        distance_fn = total_variation_distance if metric == "tv" else hellinger_distance
        reported = distance_fn(p_padded, q_padded)
        failed = reported > tolerance
        label = f"{metric} distance"
        record_distance(reported, metric, shots=shots)

    if failed:
        raise AssertionError(
            _format_failure(
                circuit,
                backend,
                bit_positions,
                metric,
                tolerance,
                reported,
                label,
                expected_norm,
                measured,
                msg,
            )
        )


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _marginalize(counts: dict[str, int], positions: list[int]) -> dict[str, int]:
    """Sum *counts* into the marginal over the given bitstring *positions*."""
    marginal: dict[str, int] = {}
    for outcome, c in counts.items():
        if any(p < 0 or p >= len(outcome) for p in positions):
            raise ValueError(
                f"bit_positions {positions} out of range for outcome {outcome!r} "
                f"of width {len(outcome)}."
            )
        key = "".join(outcome[p] for p in positions)
        marginal[key] = marginal.get(key, 0) + c
    return marginal


def _format_failure(
    circuit: Any,
    backend: Backend,
    positions: list[int],
    metric: str,
    tolerance: float,
    reported: float,
    label: str,
    expected: dict[str, float],
    measured: dict[str, float],
    user_msg: str | None,
    extra_note: str | None = None,
) -> str:
    name = getattr(circuit, "name", None) or type(circuit).__name__
    lines: list[str] = []
    if user_msg:
        lines.extend([user_msg, ""])
    lines.append("Marginal distribution mismatch beyond tolerance")
    lines.append("")
    lines.append(f"  Circuit: {name}")
    lines.append(f"  Backend: {backend.name}")
    lines.append(f"  Bit positions: {positions}")
    lines.append(f"  Metric: {metric}")
    lines.append(f"  Tolerance: {tolerance}")
    lines.append(f"  Measured {label}: {reported:.6f}")
    lines.append("")
    lines.append("  Expected vs Measured (marginal):")
    for key in sorted(set(expected) | set(measured)):
        e = expected.get(key, 0.0)
        m = measured.get(key, 0.0)
        lines.append(f'    "{key}": expected {e:.3f}, measured {m:.3f}  (diff: {m - e:+.3f})')
    if extra_note:
        lines.append("")
        lines.append(f"  Note: {extra_note}")
    return "\n".join(lines)
