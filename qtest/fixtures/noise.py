"""Pytest fixtures for ready-made noise models.

Register in your ``conftest.py`` to use them::

    # tests/conftest.py
    pytest_plugins = ["qtest.fixtures.noise"]

Then request them in test signatures::

    def test_bell_under_noise(bell_state, light_noise):
        bell_state.measure_all()
        assert_distribution_close(
            bell_state, {"00": 0.5, "11": 0.5},
            shots=4096, tolerance=0.1, noise_model=light_noise,
        )

The factory fixtures (``depolarizing_noise`` …) yield the constructor functions
so a test can dial in its own strength; the ``*_noise`` fixtures yield concrete,
mild presets convenient for smoke tests.
"""

from __future__ import annotations

from typing import Callable

import pytest

from qtest.noise import NoiseModel, depolarizing, readout_error, thermal_relaxation


@pytest.fixture
def depolarizing_noise() -> Callable[[float], NoiseModel]:
    """Yield the :func:`qtest.noise.depolarizing` constructor (``p -> NoiseModel``)."""
    return depolarizing


@pytest.fixture
def readout_noise() -> Callable[..., NoiseModel]:
    """Yield the :func:`qtest.noise.readout_error` constructor."""
    return readout_error


@pytest.fixture
def thermal_noise() -> Callable[..., NoiseModel]:
    """Yield the :func:`qtest.noise.thermal_relaxation` constructor."""
    return thermal_relaxation


@pytest.fixture
def light_noise() -> NoiseModel:
    """A mild depolarizing model (``p = 0.001``) for smoke-testing noise paths."""
    return depolarizing(0.001)


@pytest.fixture
def heavy_noise() -> NoiseModel:
    """A strong depolarizing model (``p = 0.1``) for stress-testing robustness."""
    return depolarizing(0.1)
