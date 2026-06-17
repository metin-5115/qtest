"""``assert_robust_to_noise`` — noise-degradation regression assertion.

Where :func:`qtest.assert_distribution_close` checks a circuit at a single
(possibly noisy) operating point, :func:`assert_robust_to_noise` sweeps a
*ladder* of increasing noise strengths and asserts that the circuit's output
distribution never drifts further than ``max_distance`` from *expected*. This
is the natural shape of an error-mitigation / error-correction regression
test: "as noise rises to level X, my mitigated circuit must still track the
ideal result to within Y".
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable

from qtest.assertions.distribution import (
    _has_measurements,
    _normalize_counts,
    _validate_bitstring_lengths,
    _validate_expected,
)
from qtest.backends import Backend
from qtest.backends.registry import get_backend
from qtest.config import _resolve_value, get_config
from qtest.metrics import hellinger_distance, total_variation_distance
from qtest.noise import NoiseModel, bit_flip, depolarizing, phase_flip, readout_error

# Single-parameter noise constructors selectable via ``noise_type``.
_NOISE_FACTORIES: dict[str, Callable[[float], NoiseModel]] = {
    "depolarizing": depolarizing,
    "bit_flip": bit_flip,
    "phase_flip": phase_flip,
    "readout": readout_error,
}

_DistanceFn = Callable[[Mapping[str, float], Mapping[str, float]], float]
_DISTANCE_METRICS: dict[str, _DistanceFn] = {
    "tv": total_variation_distance,
    "hellinger": hellinger_distance,
}


def assert_robust_to_noise(
    circuit: Any,
    expected: dict[str, float],
    noise_levels: Sequence[float] = (0.001, 0.01, 0.05),
    max_distance: float = 0.15,
    noise_type: str = "depolarizing",
    metric: str = "tv",
    shots: int | None = None,
    backend: Backend | None = None,
    seed: int | None = None,
    msg: str | None = None,
) -> None:
    """Assert *circuit*'s output stays within *max_distance* of *expected* under noise.

    For each value ``p`` in *noise_levels* a noise model of type *noise_type*
    and strength ``p`` is applied, the circuit is sampled, and the distance
    between the measured and *expected* distributions is computed. The
    assertion fails if **any** level exceeds *max_distance*.

    Parameters
    ----------
    circuit
        A backend-native circuit with at least one measurement instruction.
    expected
        Reference distribution as ``{bitstring: probability}`` (summing to 1).
    noise_levels
        Increasing noise strengths to sweep. Each is passed as the single
        probability argument of the chosen *noise_type* constructor.
    max_distance
        Maximum permitted distance at every noise level, in ``[0, 1]``.
    noise_type
        One of ``"depolarizing"``, ``"bit_flip"``, ``"phase_flip"``,
        ``"readout"``.
    metric
        Distance metric: ``"tv"`` (default) or ``"hellinger"``.
    shots, backend, seed
        Per-call overrides for the corresponding global config defaults.
    msg
        Optional prefix prepended to the assertion failure message.

    Raises
    ------
    ValueError
        For malformed input (bad noise_type, metric, empty noise_levels,
        out-of-range max_distance, circuit without measurements, ...).
    AssertionError
        When the measured distribution exceeds *max_distance* at any level.
    """
    if noise_type not in _NOISE_FACTORIES:
        raise ValueError(
            f"Unknown noise_type {noise_type!r}. Must be one of {sorted(_NOISE_FACTORIES)}."
        )
    if metric not in _DISTANCE_METRICS:
        raise ValueError(f"metric must be one of {sorted(_DISTANCE_METRICS)}, got {metric!r}.")
    if not noise_levels:
        raise ValueError("noise_levels must be a non-empty sequence of probabilities.")
    if not isinstance(max_distance, (int, float)) or not (0.0 <= float(max_distance) <= 1.0):
        raise ValueError(f"max_distance must be in [0, 1], got {max_distance!r}")

    shots = _resolve_value("default_shots", shots)
    seed = _resolve_value("default_seed", seed)

    expected_norm = _validate_expected(expected)
    if not _has_measurements(circuit):
        raise ValueError(
            "circuit has no measurement instructions; assert_robust_to_noise "
            "requires a circuit that produces classical bitstrings."
        )

    if backend is None:
        backend = get_backend(get_config().default_backend)

    noise_factory = _NOISE_FACTORIES[noise_type]
    distance_fn = _DISTANCE_METRICS[metric]

    results: list[tuple[float, float, bool]] = []
    for level in noise_levels:
        model = noise_factory(level)
        raw_counts = backend.run_circuit(circuit, shots=shots, seed=seed, noise_model=model)
        measured = _normalize_counts({k.replace(" ", ""): v for k, v in raw_counts.items()})
        _validate_bitstring_lengths(expected_norm, measured)

        all_keys = set(expected_norm) | set(measured)
        p_padded = {k: expected_norm.get(k, 0.0) for k in all_keys}
        q_padded = {k: measured.get(k, 0.0) for k in all_keys}
        distance = distance_fn(p_padded, q_padded)
        results.append((float(level), distance, distance <= max_distance))

    if all(passed for _, _, passed in results):
        return

    raise AssertionError(
        _format_failure_message(
            circuit=circuit,
            backend=backend,
            noise_type=noise_type,
            metric=metric,
            max_distance=max_distance,
            shots=shots,
            results=results,
            user_msg=msg,
        )
    )


def _format_failure_message(
    *,
    circuit: Any,
    backend: Backend,
    noise_type: str,
    metric: str,
    max_distance: float,
    shots: int,
    results: list[tuple[float, float, bool]],
    user_msg: str | None,
) -> str:
    """Build the multi-line ``AssertionError`` payload for a robustness sweep."""
    name = getattr(circuit, "name", None) or type(circuit).__name__
    lines: list[str] = []
    if user_msg:
        lines.extend([user_msg, ""])
    lines.append("Circuit is not robust to noise")
    lines.append("")
    lines.append(f"  Circuit: {name}")
    lines.append(f"  Backend: {backend.name}")
    lines.append(f"  Noise type: {noise_type}")
    lines.append(f"  Metric: {metric} distance")
    lines.append(f"  Shots: {shots}")
    lines.append(f"  Max allowed distance: {max_distance}")
    lines.append("")
    lines.append("  Noise level     distance     status")
    for level, distance, passed in results:
        status = "ok" if passed else "FAIL"
        lines.append(f"    {level:<12.5g} {distance:<12.6f} {status}")
    return "\n".join(lines)
