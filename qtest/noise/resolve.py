"""Resolve the ``noise_model`` assertion argument against global config.

Kept in its own module (rather than ``qtest.noise.models``) so that the noise
*models* stay free of any dependency on :mod:`qtest.config`. Assertion
functions call :func:`resolve_noise_model` to turn the flexible ``noise_model``
argument — a :class:`NoiseModel`, a preset name, or ``None`` — into a concrete
:class:`NoiseModel` or ``None``, honouring the ``default_noise`` config field
when the caller passes nothing.
"""

from __future__ import annotations

from qtest.config import get_config
from qtest.noise.models import NoiseModel, from_preset


def resolve_noise_model(noise_model: NoiseModel | str | None) -> NoiseModel | None:
    """Resolve *noise_model* to a :class:`NoiseModel` or ``None``.

    Resolution order:

    * a :class:`NoiseModel` instance is returned as-is;
    * a ``str`` is looked up as a built-in preset (:func:`from_preset`);
    * ``None`` falls back to the ``default_noise`` config preset, or ``None``
      when that too is unset (ideal, noiseless simulation).

    Raises
    ------
    ValueError
        If a string does not name a known preset.
    TypeError
        If *noise_model* is neither ``None``, a ``str``, nor a
        :class:`NoiseModel`.
    """
    if noise_model is None:
        preset = get_config().default_noise
        return from_preset(preset) if preset else None
    if isinstance(noise_model, NoiseModel):
        return noise_model
    if isinstance(noise_model, str):
        return from_preset(noise_model)
    raise TypeError(
        "noise_model must be a qtest.noise.NoiseModel, a preset name (str), "
        f"or None; got {type(noise_model).__name__}"
    )
