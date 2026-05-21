"""Tests for :func:`qtest.assertions.assert_distribution_close`."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from qtest.assertions import assert_distribution_close
from qtest.assertions.distribution import (
    _circuit_summary,
    _format_diff_table,
    _has_measurements,
    _normalize_counts,
    _suggest_diagnosis,
    _validate_bitstring_lengths,
    _validate_expected,
)
from qtest.backends.base import Backend
from qtest.config import configure, reset_config

# --------------------------------------------------------------------------- #
# Mock backend                                                                #
# --------------------------------------------------------------------------- #


class _MockBackend(Backend):
    """Deterministic backend driven by a fixed probability dict.

    Each ``run_circuit(..., shots=N)`` returns integer counts whose
    fractions match ``probs`` exactly (rounded, with the remainder
    absorbed by the largest-probability outcome). No actual circuit
    execution happens — the backend ignores ``circuit`` and ``seed``.
    """

    def __init__(
        self,
        probs: dict[str, float],
        *,
        name: str = "mock",
        record_calls: bool = False,
    ) -> None:
        self._probs = probs
        self._name = name
        self.calls: list[dict[str, Any]] = [] if record_calls else []
        self._record = record_calls

    def run_circuit(
        self,
        circuit: Any,
        shots: int | None = None,
        seed: int | None = None,
    ) -> dict[str, int]:
        if self._record:
            self.calls.append({"circuit": circuit, "shots": shots, "seed": seed})
        if shots is None:
            shots = 1024
        if not self._probs:
            return {}
        counts = {k: int(round(p * shots)) for k, p in self._probs.items()}
        diff = shots - sum(counts.values())
        if diff != 0:
            largest = max(counts, key=lambda k: counts[k])
            counts[largest] += diff
        return counts

    def get_statevector(self, circuit: Any) -> np.ndarray:
        raise NotImplementedError

    def get_unitary(self, circuit: Any) -> np.ndarray:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self._name

    @property
    def supports_statevector(self) -> bool:
        return False


class _FakeCircuit:
    """Plain-Python stand-in for a Qiskit circuit (skips measurement check)."""

    def __init__(self, name: str = "fake", num_qubits: int | None = None) -> None:
        self.name = name
        if num_qubits is not None:
            self.num_qubits = num_qubits


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _reset_config() -> Any:
    yield
    reset_config()


@pytest.fixture
def bell_expected() -> dict[str, float]:
    return {"00": 0.5, "11": 0.5}


@pytest.fixture
def perfect_bell_backend(bell_expected: dict[str, float]) -> _MockBackend:
    return _MockBackend(bell_expected)


@pytest.fixture
def fake_circuit() -> _FakeCircuit:
    return _FakeCircuit(name="bell", num_qubits=2)


# --------------------------------------------------------------------------- #
# Happy path                                                                  #
# --------------------------------------------------------------------------- #


def test_perfect_match_passes(
    fake_circuit: _FakeCircuit,
    bell_expected: dict[str, float],
    perfect_bell_backend: _MockBackend,
) -> None:
    assert_distribution_close(
        fake_circuit,
        bell_expected,
        shots=1000,
        backend=perfect_bell_backend,
    )


def test_default_metric_is_tv(fake_circuit: _FakeCircuit, bell_expected: dict[str, float]) -> None:
    # Slightly-off backend (3% off in TV)
    backend = _MockBackend({"00": 0.53, "11": 0.47})
    # tolerance 0.05 -> should pass
    assert_distribution_close(
        fake_circuit,
        bell_expected,
        shots=1000,
        tolerance=0.05,
        backend=backend,
    )


# --------------------------------------------------------------------------- #
# Failure path                                                                #
# --------------------------------------------------------------------------- #


def test_off_distribution_fails(
    fake_circuit: _FakeCircuit, bell_expected: dict[str, float]
) -> None:
    backend = _MockBackend({"00": 0.9, "11": 0.1})  # TVD = 0.4
    with pytest.raises(AssertionError) as exc:
        assert_distribution_close(
            fake_circuit,
            bell_expected,
            shots=1000,
            tolerance=0.05,
            backend=backend,
        )
    assert "Distribution mismatch" in str(exc.value)


def test_failure_message_contains_diagnostics(
    fake_circuit: _FakeCircuit, bell_expected: dict[str, float]
) -> None:
    backend = _MockBackend({"00": 0.9, "11": 0.1}, name="mock-A")
    with pytest.raises(AssertionError) as exc:
        assert_distribution_close(
            fake_circuit,
            bell_expected,
            shots=2000,
            tolerance=0.01,
            backend=backend,
        )
    text = str(exc.value)
    assert "bell" in text  # circuit name
    assert "mock-A" in text  # backend name
    assert "Shots: 2000" in text
    assert "Metric: tv" in text
    assert "Tolerance: 0.01" in text
    assert "Measured tv distance:" in text
    assert "Expected vs Measured" in text
    assert "Suggestion:" in text  # verbose_failures=True by default


def test_failure_message_user_msg_prefixed(
    fake_circuit: _FakeCircuit, bell_expected: dict[str, float]
) -> None:
    backend = _MockBackend({"00": 1.0})
    with pytest.raises(AssertionError) as exc:
        assert_distribution_close(
            fake_circuit,
            bell_expected,
            shots=100,
            tolerance=0.01,
            backend=backend,
            msg="Bell preparation failed",
        )
    assert str(exc.value).startswith("Bell preparation failed")


def test_failure_message_omits_suggestion_when_verbose_off(
    fake_circuit: _FakeCircuit, bell_expected: dict[str, float]
) -> None:
    configure(verbose_failures=False)
    backend = _MockBackend({"00": 1.0})
    with pytest.raises(AssertionError) as exc:
        assert_distribution_close(
            fake_circuit,
            bell_expected,
            shots=100,
            tolerance=0.01,
            backend=backend,
        )
    assert "Suggestion:" not in str(exc.value)


# --------------------------------------------------------------------------- #
# Metric switches                                                             #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("metric", ["tv", "hellinger"])
def test_distance_metrics_pass_on_perfect_match(
    fake_circuit: _FakeCircuit,
    bell_expected: dict[str, float],
    perfect_bell_backend: _MockBackend,
    metric: str,
) -> None:
    assert_distribution_close(
        fake_circuit,
        bell_expected,
        shots=1000,
        tolerance=0.01,
        metric=metric,
        backend=perfect_bell_backend,
    )


def test_chi_square_passes_on_perfect_match(
    fake_circuit: _FakeCircuit,
    bell_expected: dict[str, float],
    perfect_bell_backend: _MockBackend,
) -> None:
    # Perfect match -> p-value should be ~1.0, well above α=0.05
    assert_distribution_close(
        fake_circuit,
        bell_expected,
        shots=1000,
        tolerance=0.05,
        metric="chi_square",
        backend=perfect_bell_backend,
    )


def test_chi_square_rejects_inconsistent_distribution(
    fake_circuit: _FakeCircuit, bell_expected: dict[str, float]
) -> None:
    # 9/1 split is statistically very far from 50/50 at 1000 shots
    backend = _MockBackend({"00": 0.9, "11": 0.1})
    with pytest.raises(AssertionError):
        assert_distribution_close(
            fake_circuit,
            bell_expected,
            shots=1000,
            tolerance=0.05,
            metric="chi_square",
            backend=backend,
        )


def test_chi_square_hard_rejection_on_forbidden_outcome(
    fake_circuit: _FakeCircuit,
) -> None:
    """expected forbids '01' (prob 0) but backend produces it."""
    backend = _MockBackend({"00": 0.5, "01": 0.5})
    with pytest.raises(AssertionError, match="rejected hypothesis outright"):
        assert_distribution_close(
            fake_circuit,
            {"00": 0.5, "11": 0.5},
            shots=1000,
            tolerance=0.05,
            metric="chi_square",
            backend=backend,
        )


def test_unknown_metric_raises() -> None:
    with pytest.raises(ValueError, match="Unknown metric"):
        assert_distribution_close(
            _FakeCircuit(),
            {"0": 1.0},
            shots=100,
            metric="bogus",
            backend=_MockBackend({"0": 1.0}),
        )


# --------------------------------------------------------------------------- #
# Config integration                                                          #
# --------------------------------------------------------------------------- #


def test_config_defaults_are_used_when_args_omitted(
    fake_circuit: _FakeCircuit, bell_expected: dict[str, float]
) -> None:
    backend = _MockBackend(bell_expected, record_calls=True)
    configure(default_shots=777, default_tolerance=0.5, statistical_metric="hellinger")
    assert_distribution_close(fake_circuit, bell_expected, backend=backend)
    assert backend.calls[-1]["shots"] == 777


def test_seed_is_passed_through_to_backend(
    fake_circuit: _FakeCircuit, bell_expected: dict[str, float]
) -> None:
    backend = _MockBackend(bell_expected, record_calls=True)
    assert_distribution_close(fake_circuit, bell_expected, shots=100, seed=12345, backend=backend)
    assert backend.calls[-1]["seed"] == 12345


def test_auto_tolerance_kicks_in_when_enabled(
    fake_circuit: _FakeCircuit, bell_expected: dict[str, float]
) -> None:
    """With auto_tolerance=True, no explicit tolerance -> derived from shots."""
    configure(auto_tolerance=True)
    # Very imperfect backend (~0.1 TVD off); 100 shots -> auto_tolerance ~ 0.26
    # which is generous enough to pass.
    backend = _MockBackend({"00": 0.6, "11": 0.4})
    assert_distribution_close(fake_circuit, bell_expected, shots=100, backend=backend)


# --------------------------------------------------------------------------- #
# Input validation                                                            #
# --------------------------------------------------------------------------- #


def test_empty_expected_raises(fake_circuit: _FakeCircuit) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        assert_distribution_close(fake_circuit, {}, shots=100, backend=_MockBackend({"0": 1.0}))


def test_expected_not_summing_to_one_raises(fake_circuit: _FakeCircuit) -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        assert_distribution_close(
            fake_circuit,
            {"00": 0.5, "11": 0.4},
            shots=100,
            backend=_MockBackend({"00": 1.0}),
        )


def test_negative_probability_raises(fake_circuit: _FakeCircuit) -> None:
    with pytest.raises(ValueError, match="outside"):
        assert_distribution_close(
            fake_circuit,
            {"00": -0.1, "11": 1.1},
            shots=100,
            backend=_MockBackend({"00": 1.0}),
        )


def test_non_dict_expected_raises(fake_circuit: _FakeCircuit) -> None:
    with pytest.raises(ValueError, match="must be a dict"):
        assert_distribution_close(
            fake_circuit,
            [("00", 0.5), ("11", 0.5)],  # type: ignore[arg-type]
            shots=100,
            backend=_MockBackend({"00": 1.0}),
        )


def test_non_string_keys_raises(fake_circuit: _FakeCircuit) -> None:
    with pytest.raises(ValueError, match="keys must be strings"):
        assert_distribution_close(
            fake_circuit,
            {0: 0.5, 1: 0.5},  # type: ignore[dict-item]
            shots=100,
            backend=_MockBackend({"0": 1.0}),
        )


@pytest.mark.parametrize("bad_shots", [0, -1, -100])
def test_non_positive_shots_raises(
    fake_circuit: _FakeCircuit,
    bell_expected: dict[str, float],
    bad_shots: int,
) -> None:
    with pytest.raises(ValueError, match="shots"):
        assert_distribution_close(
            fake_circuit,
            bell_expected,
            shots=bad_shots,
            backend=_MockBackend(bell_expected),
        )


@pytest.mark.parametrize("bad_tol", [-0.1, 1.5, 99])
def test_out_of_range_tolerance_raises(
    fake_circuit: _FakeCircuit,
    bell_expected: dict[str, float],
    bad_tol: float,
) -> None:
    with pytest.raises(ValueError, match="tolerance"):
        assert_distribution_close(
            fake_circuit,
            bell_expected,
            shots=100,
            tolerance=bad_tol,
            backend=_MockBackend(bell_expected),
        )


def test_inconsistent_bitstring_lengths_raises(
    fake_circuit: _FakeCircuit,
) -> None:
    """expected uses 2-bit strings, measured uses 3-bit strings."""
    backend = _MockBackend({"000": 1.0})
    with pytest.raises(ValueError, match="bitstring lengths"):
        assert_distribution_close(
            fake_circuit,
            {"00": 0.5, "11": 0.5},
            shots=100,
            backend=backend,
        )


# --------------------------------------------------------------------------- #
# Measurement check                                                           #
# --------------------------------------------------------------------------- #


def test_missing_measurement_raises_for_qiskit_circuit() -> None:
    qiskit = pytest.importorskip("qiskit")
    qc = qiskit.QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)  # no measurement!
    with pytest.raises(ValueError, match="no measurement"):
        assert_distribution_close(
            qc, {"00": 0.5, "11": 0.5}, shots=100, backend=_MockBackend({"00": 1.0})
        )


def test_fake_circuit_skips_measurement_check(
    fake_circuit: _FakeCircuit, bell_expected: dict[str, float]
) -> None:
    """Non-Qiskit circuit objects skip the measurement check (best-effort)."""
    backend = _MockBackend(bell_expected)
    assert_distribution_close(fake_circuit, bell_expected, shots=100, backend=backend)


# --------------------------------------------------------------------------- #
# Backend selection                                                           #
# --------------------------------------------------------------------------- #


def test_explicit_backend_overrides_config(
    fake_circuit: _FakeCircuit, bell_expected: dict[str, float]
) -> None:
    backend = _MockBackend(bell_expected, name="explicit", record_calls=True)
    assert_distribution_close(fake_circuit, bell_expected, shots=100, backend=backend)
    assert backend.calls, "explicit backend was not called"


# --------------------------------------------------------------------------- #
# Helper unit tests                                                           #
# --------------------------------------------------------------------------- #


def test_normalize_counts_basic() -> None:
    assert _normalize_counts({"0": 600, "1": 400}) == {"0": 0.6, "1": 0.4}


def test_normalize_counts_empty() -> None:
    assert _normalize_counts({}) == {}


def test_validate_expected_strips_spaces() -> None:
    out = _validate_expected({"0 0": 0.5, "1 1": 0.5})
    assert out == {"00": 0.5, "11": 0.5}


def test_validate_bitstring_lengths_consistent_ok() -> None:
    _validate_bitstring_lengths({"00": 1.0}, {"00": 1.0, "11": 0.0})


def test_validate_bitstring_lengths_inconsistent_raises() -> None:
    with pytest.raises(ValueError):
        _validate_bitstring_lengths({"00": 1.0}, {"000": 1.0})


def test_circuit_summary_uses_name_and_qubits() -> None:
    s = _circuit_summary(_FakeCircuit(name="ghz", num_qubits=3))
    assert "ghz" in s
    assert "3" in s


def test_format_diff_table_includes_sign_and_keys() -> None:
    table = _format_diff_table({"00": 0.5, "11": 0.5}, {"00": 0.6, "11": 0.4})
    assert '"00"' in table
    assert "+0.100" in table
    assert "-0.100" in table


def test_has_measurements_for_qiskit_circuit() -> None:
    qiskit = pytest.importorskip("qiskit")
    qc = qiskit.QuantumCircuit(1, 1)
    qc.h(0)
    assert _has_measurements(qc) is False
    qc.measure(0, 0)
    assert _has_measurements(qc) is True


def test_suggest_diagnosis_detects_unexpected_outcomes() -> None:
    hint = _suggest_diagnosis({"00": 1.0}, {"00": 0.5, "01": 0.5})
    assert "unexpected outcomes" in hint


def test_suggest_diagnosis_detects_peak_shift() -> None:
    hint = _suggest_diagnosis({"00": 0.7, "11": 0.3}, {"00": 0.3, "11": 0.7})
    assert "Peak shifted" in hint or "shape but differ" in hint


# --------------------------------------------------------------------------- #
# Reproducibility                                                             #
# --------------------------------------------------------------------------- #


def test_same_seed_same_result(fake_circuit: _FakeCircuit, bell_expected: dict[str, float]) -> None:
    """Twice the same call should not raise, given the mock is deterministic."""
    backend = _MockBackend(bell_expected)
    for _ in range(3):
        assert_distribution_close(fake_circuit, bell_expected, shots=500, seed=99, backend=backend)
