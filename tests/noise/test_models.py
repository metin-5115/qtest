"""Tests for :mod:`qtest.noise.models`."""

from __future__ import annotations

import pytest

from qtest.noise import (
    NoiseModel,
    available_presets,
    bit_flip,
    depolarizing,
    from_preset,
    phase_flip,
    readout_error,
    thermal_relaxation,
)

# --------------------------------------------------------------------------- #
# Constructors & validation                                                   #
# --------------------------------------------------------------------------- #


def test_constructors_return_noise_model() -> None:
    for model in (
        depolarizing(0.01),
        bit_flip(0.02),
        phase_flip(0.03),
        thermal_relaxation(100.0, 80.0, 1.0),
        readout_error(0.05),
    ):
        assert isinstance(model, NoiseModel)
        assert not model.is_empty
        assert model.label  # non-empty human-readable label


def test_readout_error_defaults_p10_to_p01() -> None:
    model = readout_error(0.07)
    assert "p10=0.07" in model.label


@pytest.mark.parametrize("p", [-0.1, 1.1, float("nan"), float("inf")])
def test_probability_out_of_range_raises(p: float) -> None:
    with pytest.raises(ValueError):
        depolarizing(p)


def test_probability_bool_rejected() -> None:
    with pytest.raises(ValueError):
        bit_flip(True)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf")])
def test_thermal_positive_args_required(bad: float) -> None:
    with pytest.raises(ValueError):
        thermal_relaxation(bad, 1.0, 1.0)


def test_thermal_t2_must_not_exceed_2t1() -> None:
    with pytest.raises(ValueError, match="t2 <= 2"):
        thermal_relaxation(10.0, 30.0, 1.0)


# --------------------------------------------------------------------------- #
# Composition                                                                 #
# --------------------------------------------------------------------------- #


def test_addition_combines_channels() -> None:
    combined = depolarizing(0.01) + readout_error(0.02)
    assert isinstance(combined, NoiseModel)
    assert "depolarizing" in combined.label
    assert "readout_error" in combined.label


def test_addition_with_non_noise_model_returns_notimplemented() -> None:
    assert depolarizing(0.01).__add__(42) is NotImplemented


def test_repr_mentions_label_and_channel_count() -> None:
    r = repr(depolarizing(0.01))
    assert "NoiseModel" in r and "depolarizing" in r


# --------------------------------------------------------------------------- #
# Presets                                                                     #
# --------------------------------------------------------------------------- #


def test_available_presets_sorted_and_nonempty() -> None:
    presets = available_presets()
    assert presets == sorted(presets)
    assert "depolarizing" in presets


def test_from_preset_returns_model() -> None:
    assert isinstance(from_preset("depolarizing"), NoiseModel)


def test_from_preset_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown noise preset"):
        from_preset("nope")


# --------------------------------------------------------------------------- #
# to_qiskit (requires qiskit-aer)                                             #
# --------------------------------------------------------------------------- #


def test_to_qiskit_builds_aer_model() -> None:
    pytest.importorskip("qiskit_aer")
    aer_nm = (depolarizing(0.01) + readout_error(0.02)).to_qiskit()
    # An Aer NoiseModel reports the basis gates it touches.
    assert hasattr(aer_nm, "basis_gates")


@pytest.mark.parametrize(
    "model",
    [
        depolarizing(0.01),
        bit_flip(0.02),
        phase_flip(0.03),
        thermal_relaxation(100.0, 80.0, 1.0),
        readout_error(0.05),
    ],
)
def test_to_qiskit_per_channel(model: NoiseModel) -> None:
    pytest.importorskip("qiskit_aer")
    from qiskit_aer.noise import NoiseModel as AerNoiseModel

    assert isinstance(model.to_qiskit(), AerNoiseModel)


def test_to_qiskit_combined_all_channels() -> None:
    pytest.importorskip("qiskit_aer")
    combined = (
        depolarizing(0.01)
        + bit_flip(0.01)
        + phase_flip(0.01)
        + thermal_relaxation(100.0, 80.0, 1.0)
        + readout_error(0.02)
    )
    assert hasattr(combined.to_qiskit(), "basis_gates")
