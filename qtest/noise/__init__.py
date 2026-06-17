"""Noise models for noise-aware quantum testing.

Public API:

* :class:`NoiseModel` — an SDK-agnostic description of a noise channel.
* Constructors :func:`depolarizing`, :func:`bit_flip`, :func:`phase_flip`,
  :func:`thermal_relaxation`, :func:`readout_error`.
* Preset helpers :func:`from_preset`, :func:`available_presets`.

Importing this package does **not** import Qiskit Aer; the Aer noise objects
are built lazily the first time :meth:`NoiseModel.to_qiskit` is called.

Pass any of these to the ``noise_model`` argument of
:func:`qtest.assert_distribution_close` / :func:`qtest.assert_state_close`, or
set a preset name globally via ``--qtest-noise`` / ``[tool.qtest] default_noise``.
"""

from qtest.noise.models import (
    NoiseModel,
    available_presets,
    bit_flip,
    depolarizing,
    from_preset,
    phase_flip,
    readout_error,
    thermal_relaxation,
)

__all__ = [
    "NoiseModel",
    "available_presets",
    "bit_flip",
    "depolarizing",
    "from_preset",
    "phase_flip",
    "readout_error",
    "thermal_relaxation",
]
