"""Noise-aware quantum testing with qtest.

Run: pytest examples/06_noisy_testing.py -v

Real quantum hardware is noisy. A circuit that is perfect on an ideal
simulator can still drift once depolarizing error, readout error, and
T1/T2 relaxation enter the picture. This file shows how qtest lets you
test *under noise* with a single extra argument.

What you will see (when the suite passes):

    ============= 5 passed in ~2s =============

Requires the Aer simulator: ``pip install 'qtest[aer]'``.
"""

from __future__ import annotations

from qiskit import QuantumCircuit

from qtest import (
    assert_distribution_close,
    assert_robust_to_noise,
    assert_state_close,
)
from qtest.noise import depolarizing, readout_error, thermal_relaxation


def _bell() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    return qc


# --------------------------------------------------------------------------- #
# 1. A Bell state still looks 50/50 under light depolarizing noise            #
# --------------------------------------------------------------------------- #
#
# Pass a NoiseModel via `noise_model`. Note the looser tolerance: noise
# spreads probability into the "01"/"10" outcomes, so we allow more slack
# than we would on an ideal simulator.


def test_bell_survives_light_depolarizing_noise() -> None:
    assert_distribution_close(
        _bell(),
        expected={"00": 0.5, "11": 0.5},
        shots=4096,
        tolerance=0.15,
        seed=42,
        noise_model=depolarizing(0.01),
    )


# --------------------------------------------------------------------------- #
# 2. Readout error: you can also pass a preset name as a string              #
# --------------------------------------------------------------------------- #


def test_bell_with_readout_error_preset() -> None:
    assert_distribution_close(
        _bell(),
        expected={"00": 0.5, "11": 0.5},
        shots=4096,
        tolerance=0.2,
        seed=42,
        noise_model="readout",
    )


# --------------------------------------------------------------------------- #
# 3. Noisy state comparison via density-matrix fidelity                      #
# --------------------------------------------------------------------------- #
#
# Under noise a circuit yields a *mixed* state. With `noise_model` set,
# assert_state_close compares the resulting density matrix to the ideal
# target via fidelity.


def test_plus_state_fidelity_under_noise() -> None:
    qc = QuantumCircuit(1)
    qc.h(0)
    assert_state_close(
        qc,
        expected_state="plus",
        tolerance=0.05,
        noise_model=thermal_relaxation(t1=200.0, t2=120.0, time=1.0),
    )


# --------------------------------------------------------------------------- #
# 4. Robustness sweep: bound the degradation as noise increases              #
# --------------------------------------------------------------------------- #
#
# assert_robust_to_noise sweeps a ladder of noise strengths and checks the
# output never drifts further than `max_distance` from the ideal result —
# the natural shape of an error-mitigation regression test.


def test_bell_is_robust_across_noise_levels() -> None:
    assert_robust_to_noise(
        _bell(),
        expected={"00": 0.5, "11": 0.5},
        noise_levels=[0.001, 0.005, 0.02],
        max_distance=0.2,
        noise_type="depolarizing",
        shots=4096,
        seed=42,
    )


# --------------------------------------------------------------------------- #
# 5. Layering channels with `+`                                              #
# --------------------------------------------------------------------------- #


def test_combined_noise_channels() -> None:
    combined = depolarizing(0.005) + readout_error(0.01)
    assert_distribution_close(
        _bell(),
        expected={"00": 0.5, "11": 0.5},
        shots=4096,
        tolerance=0.2,
        seed=42,
        noise_model=combined,
    )


# Try it: tighten `tolerance` in test #1 to 0.01 (well below the noise
# floor) and watch the failure message report the measured distribution,
# the active noise model, and a diagnosis.
