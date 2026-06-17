"""Tests for noise-model resolution and config integration."""

from __future__ import annotations

from typing import Any

import pytest

from qtest.config import configure, reset_config
from qtest.noise import NoiseModel, depolarizing
from qtest.noise.resolve import resolve_noise_model


@pytest.fixture(autouse=True)
def _isolate_config() -> Any:
    yield
    reset_config()


def test_resolve_none_without_config_returns_none() -> None:
    assert resolve_noise_model(None) is None


def test_resolve_passthrough_instance() -> None:
    model = depolarizing(0.01)
    assert resolve_noise_model(model) is model


def test_resolve_preset_name() -> None:
    assert isinstance(resolve_noise_model("depolarizing"), NoiseModel)


def test_resolve_none_uses_config_default() -> None:
    configure(default_noise="readout")
    resolved = resolve_noise_model(None)
    assert isinstance(resolved, NoiseModel)
    assert "readout" in resolved.label


def test_resolve_bad_type_raises() -> None:
    with pytest.raises(TypeError):
        resolve_noise_model(123)  # type: ignore[arg-type]


def test_resolve_unknown_preset_raises() -> None:
    with pytest.raises(ValueError):
        resolve_noise_model("nope")


def test_config_rejects_unknown_default_noise() -> None:
    with pytest.raises(ValueError, match="default_noise"):
        configure(default_noise="not-a-preset")


def test_config_accepts_none_and_valid_preset() -> None:
    configure(default_noise=None)
    configure(default_noise="depolarizing")
