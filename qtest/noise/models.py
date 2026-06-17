"""Noise models for qtest.

Real quantum hardware is noisy, and a test that only ever runs on an ideal
simulator says nothing about how a circuit behaves under decoherence and gate
error. This module provides a small, SDK-agnostic :class:`NoiseModel` wrapper
plus a handful of ready-made constructors so tests can opt into noisy
simulation with a single argument::

    from qtest import assert_distribution_close
    from qtest.noise import depolarizing

    assert_distribution_close(qc, expected, noise_model=depolarizing(0.01))

Design
------
A :class:`NoiseModel` is a lightweight, immutable *description* of the noise to
apply — it stores a list of error *specs* and nothing else. The heavy Qiskit
Aer objects are built lazily in :meth:`NoiseModel.to_qiskit`, so constructing a
:class:`NoiseModel` (and importing this module) never requires ``qiskit-aer``.
The import is only triggered when a noisy assertion actually runs.

Models compose with ``+`` so several error channels can be layered::

    combined = depolarizing(0.01) + readout_error(0.02)
"""

from __future__ import annotations

import math
from typing import Any, Callable

_AER_MISSING_MSG = (
    "Noisy simulation requires qiskit-aer. Install it with:\n" "  pip install 'qtest[aer]'"
)

# Basis gates that single- and two-qubit errors are attached to. The lists are
# intentionally broad: Aer registers an error per gate *name*, harmlessly
# ignoring names absent from a given circuit, and transpilation targets the
# noise model's basis gates so errors fire on the gates that actually run.
_DEFAULT_1Q_GATES: tuple[str, ...] = (
    "id",
    "u",
    "u1",
    "u2",
    "u3",
    "p",
    "rx",
    "ry",
    "rz",
    "h",
    "x",
    "y",
    "z",
    "s",
    "sdg",
    "t",
    "tdg",
    "sx",
    "sxdg",
)
_DEFAULT_2Q_GATES: tuple[str, ...] = ("cx", "cz", "swap", "ecr", "rzz", "rxx")


def _check_probability(name: str, value: float) -> float:
    """Validate that *value* is a real probability in ``[0, 1]``; return it as float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number in [0, 1], got {value!r}")
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if not (0.0 <= float(value) <= 1.0):
        raise ValueError(f"{name} must lie in [0, 1], got {value}")
    return float(value)


def _check_positive(name: str, value: float) -> float:
    """Validate that *value* is a strictly positive real; return it as float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive real number, got {value!r}")
    if math.isnan(value) or math.isinf(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite number, got {value}")
    return float(value)


class NoiseModel:
    """An SDK-agnostic description of a noise channel to apply during simulation.

    Instances are immutable value objects: the recommended way to build one is
    via the module-level constructors (:func:`depolarizing`, :func:`bit_flip`,
    :func:`phase_flip`, :func:`thermal_relaxation`, :func:`readout_error`) and to
    layer channels with ``+``.

    Parameters
    ----------
    specs
        Internal list of error-spec dicts. Prefer the public constructors.
    label
        Human-readable identifier used in assertion failure messages.
    """

    def __init__(self, specs: list[dict[str, Any]] | None = None, label: str = "custom") -> None:
        self._specs: list[dict[str, Any]] = list(specs or [])
        self.label = label

    def __add__(self, other: object) -> NoiseModel:
        if not isinstance(other, NoiseModel):
            return NotImplemented
        return NoiseModel(self._specs + other._specs, label=f"{self.label}+{other.label}")

    def __repr__(self) -> str:
        return f"NoiseModel(label={self.label!r}, channels={len(self._specs)})"

    @property
    def is_empty(self) -> bool:
        """Whether this model carries no error channels (a no-op)."""
        return not self._specs

    def to_qiskit(self) -> Any:
        """Build and return a :class:`qiskit_aer.noise.NoiseModel`.

        Raises
        ------
        ImportError
            If ``qiskit-aer`` is not installed.
        """
        try:
            from qiskit_aer.noise import (
                NoiseModel as AerNoiseModel,
            )
            from qiskit_aer.noise import (
                ReadoutError,
                depolarizing_error,
                pauli_error,
                thermal_relaxation_error,
            )
        except ImportError as exc:  # pragma: no cover - exercised only without aer
            raise ImportError(_AER_MISSING_MSG) from exc

        nm = AerNoiseModel()
        for spec in self._specs:
            kind = spec["kind"]
            if kind == "depolarizing":
                p = spec["p"]
                nm.add_all_qubit_quantum_error(depolarizing_error(p, 1), list(_DEFAULT_1Q_GATES))
                nm.add_all_qubit_quantum_error(depolarizing_error(p, 2), list(_DEFAULT_2Q_GATES))
            elif kind == "bit_flip":
                p = spec["p"]
                err = pauli_error([("X", p), ("I", 1.0 - p)])
                nm.add_all_qubit_quantum_error(err, list(_DEFAULT_1Q_GATES))
            elif kind == "phase_flip":
                p = spec["p"]
                err = pauli_error([("Z", p), ("I", 1.0 - p)])
                nm.add_all_qubit_quantum_error(err, list(_DEFAULT_1Q_GATES))
            elif kind == "thermal_relaxation":
                err = thermal_relaxation_error(spec["t1"], spec["t2"], spec["time"])
                nm.add_all_qubit_quantum_error(err, list(_DEFAULT_1Q_GATES))
            elif kind == "readout":
                p01, p10 = spec["p01"], spec["p10"]
                ro = ReadoutError([[1.0 - p01, p01], [p10, 1.0 - p10]])
                nm.add_all_qubit_readout_error(ro)
            else:  # pragma: no cover - guarded by constructors
                raise ValueError(f"Unknown noise channel kind: {kind!r}")
        return nm


# --------------------------------------------------------------------------- #
# Constructors                                                                #
# --------------------------------------------------------------------------- #


def depolarizing(p: float) -> NoiseModel:
    """Depolarizing error of strength *p* on every gate.

    With probability *p* the qubit(s) acted on by a gate are replaced by the
    maximally mixed state. Applied as a 1-qubit channel to single-qubit gates
    and a 2-qubit channel to two-qubit gates.

    Parameters
    ----------
    p
        Depolarizing probability in ``[0, 1]``.
    """
    p = _check_probability("p", p)
    return NoiseModel([{"kind": "depolarizing", "p": p}], label=f"depolarizing(p={p:g})")


def bit_flip(p: float) -> NoiseModel:
    """Bit-flip (Pauli-X) error: apply X with probability *p* after 1-qubit gates."""
    p = _check_probability("p", p)
    return NoiseModel([{"kind": "bit_flip", "p": p}], label=f"bit_flip(p={p:g})")


def phase_flip(p: float) -> NoiseModel:
    """Phase-flip (Pauli-Z) error: apply Z with probability *p* after 1-qubit gates."""
    p = _check_probability("p", p)
    return NoiseModel([{"kind": "phase_flip", "p": p}], label=f"phase_flip(p={p:g})")


def thermal_relaxation(t1: float, t2: float, time: float) -> NoiseModel:
    """Thermal-relaxation (T1/T2) error applied to single-qubit gates.

    Parameters
    ----------
    t1
        Relaxation time constant (same time unit as *time*).
    t2
        Dephasing time constant. Must satisfy ``t2 <= 2 * t1``.
    time
        Gate duration over which relaxation accumulates.
    """
    t1 = _check_positive("t1", t1)
    t2 = _check_positive("t2", t2)
    time = _check_positive("time", time)
    if t2 > 2.0 * t1:
        raise ValueError(f"thermal_relaxation requires t2 <= 2*t1, got t2={t2}, t1={t1}")
    return NoiseModel(
        [{"kind": "thermal_relaxation", "t1": t1, "t2": t2, "time": time}],
        label=f"thermal_relaxation(t1={t1:g}, t2={t2:g}, time={time:g})",
    )


def readout_error(p01: float, p10: float | None = None) -> NoiseModel:
    """Measurement (readout) error.

    Parameters
    ----------
    p01
        Probability of reading ``1`` when the true state is ``0``.
    p10
        Probability of reading ``0`` when the true state is ``1``. Defaults to
        *p01* (a symmetric readout error).
    """
    p01 = _check_probability("p01", p01)
    p10 = p01 if p10 is None else _check_probability("p10", p10)
    return NoiseModel(
        [{"kind": "readout", "p01": p01, "p10": p10}],
        label=f"readout_error(p01={p01:g}, p10={p10:g})",
    )


# --------------------------------------------------------------------------- #
# Named presets (used by the --qtest-noise CLI flag)                          #
# --------------------------------------------------------------------------- #


_PRESETS: dict[str, Callable[[], NoiseModel]] = {
    "depolarizing": lambda: depolarizing(0.01),
    "bit_flip": lambda: bit_flip(0.01),
    "phase_flip": lambda: phase_flip(0.01),
    "readout": lambda: readout_error(0.02),
}


def available_presets() -> list[str]:
    """Return the sorted names of the built-in noise presets."""
    return sorted(_PRESETS)


def from_preset(name: str) -> NoiseModel:
    """Return the :class:`NoiseModel` for a built-in preset *name*.

    Raises
    ------
    ValueError
        If *name* is not a known preset.
    """
    if name not in _PRESETS:
        raise ValueError(f"Unknown noise preset {name!r}. Available presets: {available_presets()}")
    return _PRESETS[name]()
