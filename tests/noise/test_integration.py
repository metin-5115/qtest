"""Integration tests for noise-aware assertions (require qiskit + qiskit-aer)."""

from __future__ import annotations

from typing import Any

import pytest

import qtest
from qtest.config import configure, reset_config
from qtest.noise import depolarizing, readout_error

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")

from qiskit import QuantumCircuit  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_config() -> Any:
    yield
    reset_config()


def _bell_measured() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    return qc


# --------------------------------------------------------------------------- #
# assert_distribution_close + noise                                           #
# --------------------------------------------------------------------------- #


def test_distribution_noiseless_baseline() -> None:
    qtest.assert_distribution_close(
        _bell_measured(), {"00": 0.5, "11": 0.5}, shots=4096, tolerance=0.05, seed=7
    )


def test_distribution_with_light_noise_still_passes() -> None:
    qtest.assert_distribution_close(
        _bell_measured(),
        {"00": 0.5, "11": 0.5},
        shots=4096,
        tolerance=0.15,
        seed=7,
        noise_model=depolarizing(0.01),
    )


def test_distribution_noise_by_preset_name() -> None:
    qtest.assert_distribution_close(
        _bell_measured(),
        {"00": 0.5, "11": 0.5},
        shots=4096,
        tolerance=0.2,
        seed=7,
        noise_model="depolarizing",
    )


def test_distribution_heavy_noise_failure_message_mentions_noise() -> None:
    qc = QuantumCircuit(1)
    qc.x(0)
    qc.measure_all()
    with pytest.raises(AssertionError) as exc:
        qtest.assert_distribution_close(
            qc, {"1": 1.0}, shots=2048, tolerance=0.001, seed=1, noise_model=readout_error(0.4)
        )
    assert "Noise:" in str(exc.value)


def test_default_noise_from_config_is_applied() -> None:
    configure(default_noise="depolarizing")
    # With config-level noise and a tight tolerance, an ideal-only expectation
    # is allowed enough slack to pass; this just exercises the config path.
    qtest.assert_distribution_close(
        _bell_measured(), {"00": 0.5, "11": 0.5}, shots=4096, tolerance=0.2, seed=7
    )


# --------------------------------------------------------------------------- #
# assert_state_close + noise (density matrix)                                 #
# --------------------------------------------------------------------------- #


def test_state_noiseless_unchanged() -> None:
    qc = QuantumCircuit(1)
    qc.h(0)
    qtest.assert_state_close(qc, "plus", tolerance=1e-9)


def test_state_light_noise_passes() -> None:
    qc = QuantumCircuit(1)
    qc.h(0)
    qtest.assert_state_close(qc, "plus", tolerance=0.1, noise_model=depolarizing(0.01))


def test_state_heavy_noise_fails() -> None:
    qc = QuantumCircuit(1)
    qc.h(0)
    with pytest.raises(AssertionError, match="under noise"):
        qtest.assert_state_close(qc, "plus", tolerance=1e-6, noise_model=depolarizing(0.3))


def test_state_noise_requires_global_phase() -> None:
    qc = QuantumCircuit(1)
    qc.h(0)
    with pytest.raises(ValueError, match="global_phase"):
        qtest.assert_state_close(
            qc, "plus", tolerance=0.1, global_phase=False, noise_model=depolarizing(0.01)
        )


# --------------------------------------------------------------------------- #
# assert_robust_to_noise                                                      #
# --------------------------------------------------------------------------- #


def test_robust_passes_for_mild_levels() -> None:
    qtest.assert_robust_to_noise(
        _bell_measured(),
        {"00": 0.5, "11": 0.5},
        noise_levels=[0.001, 0.01],
        max_distance=0.2,
        shots=4096,
        seed=7,
    )


@pytest.mark.parametrize("noise_type", ["depolarizing", "bit_flip", "phase_flip", "readout"])
def test_robust_supports_each_noise_type(noise_type: str) -> None:
    qtest.assert_robust_to_noise(
        _bell_measured(),
        {"00": 0.5, "11": 0.5},
        noise_levels=[0.001],
        max_distance=0.5,
        noise_type=noise_type,
        metric="hellinger",
        shots=2048,
        seed=7,
    )


def test_robust_fails_when_distance_exceeded() -> None:
    with pytest.raises(AssertionError, match="not robust to noise"):
        qtest.assert_robust_to_noise(
            _bell_measured(),
            {"00": 0.5, "11": 0.5},
            noise_levels=[0.4],
            max_distance=0.01,
            noise_type="readout",
            shots=2048,
            seed=7,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"noise_type": "bogus"},
        {"metric": "bogus"},
        {"noise_levels": []},
        {"max_distance": 2.0},
    ],
)
def test_robust_validation_errors(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        qtest.assert_robust_to_noise(_bell_measured(), {"00": 0.5, "11": 0.5}, **kwargs)


def test_robust_requires_measurements() -> None:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    with pytest.raises(ValueError, match="measurement"):
        qtest.assert_robust_to_noise(qc, {"00": 0.5, "11": 0.5})
