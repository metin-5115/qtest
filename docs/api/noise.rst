Noise models
============

The :mod:`qtest.noise` subpackage provides an SDK-agnostic
:class:`~qtest.noise.NoiseModel` plus ready-made constructors for the common
error channels. Pass any of them to the ``noise_model`` argument of
:func:`qtest.assert_distribution_close` / :func:`qtest.assert_state_close`, or
set a preset globally with ``--qtest-noise`` / ``[tool.qtest] default_noise``.

.. currentmodule:: qtest.noise

Public API
----------

.. autosummary::
   :nosignatures:

   NoiseModel
   depolarizing
   bit_flip
   phase_flip
   thermal_relaxation
   readout_error
   from_preset
   available_presets

Reference
---------

.. automodule:: qtest.noise.models
   :members:
   :show-inheritance:
   :member-order: bysource
