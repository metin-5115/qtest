"""Tests for the noise fixtures in :mod:`qtest.fixtures.noise`."""

from __future__ import annotations

from qtest.noise import NoiseModel

pytest_plugins = ["qtest.fixtures.noise"]


def test_light_and_heavy_noise_fixtures(light_noise: NoiseModel, heavy_noise: NoiseModel) -> None:
    assert isinstance(light_noise, NoiseModel)
    assert isinstance(heavy_noise, NoiseModel)
    assert not light_noise.is_empty


def test_factory_fixtures_build_models(
    depolarizing_noise,  # type: ignore[no-untyped-def]
    readout_noise,  # type: ignore[no-untyped-def]
    thermal_noise,  # type: ignore[no-untyped-def]
) -> None:
    assert isinstance(depolarizing_noise(0.01), NoiseModel)
    assert isinstance(readout_noise(0.02), NoiseModel)
    assert isinstance(thermal_noise(100.0, 80.0, 1.0), NoiseModel)
