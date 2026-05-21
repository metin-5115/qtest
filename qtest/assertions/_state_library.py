"""Named quantum states used by :func:`qtest.assertions.assert_state_close`.

Users may pass a short string (e.g. ``"bell"``, ``"ghz_5"``, ``"plus"``) in
place of an explicit ``numpy`` array; :func:`get_state` resolves the name to
the corresponding unit-norm complex vector.

Conventions
-----------
* Qubit ordering follows Qiskit's **little-endian** convention: for an
  ``n``-qubit register, index ``i`` of the state vector corresponds to the
  basis state ``|q_{n-1} ... q_1 q_0>`` where the bits of ``i`` give
  ``q_0`` (LSB) ... ``q_{n-1}`` (MSB). This matches what
  :class:`qiskit.quantum_info.Statevector` produces.

Known names
-----------
======================  ===========================================
``"bell"``              :math:`(\\,|00\\rangle + |11\\rangle\\,)/\\sqrt{2}`
``"ghz_<n>"``           :math:`(\\,|0\\rangle^{\\otimes n} + |1\\rangle^{\\otimes n}\\,)/\\sqrt{2}`
``"w_<n>"``             Symmetric single-excitation state on ``n`` qubits
``"plus"``              :math:`(\\,|0\\rangle + |1\\rangle\\,)/\\sqrt{2}`
``"minus"``             :math:`(\\,|0\\rangle - |1\\rangle\\,)/\\sqrt{2}`
``"i_plus"``            :math:`(\\,|0\\rangle + i|1\\rangle\\,)/\\sqrt{2}`
``"i_minus"``           :math:`(\\,|0\\rangle - i|1\\rangle\\,)/\\sqrt{2}`
======================  ===========================================
"""

from __future__ import annotations

import numpy as np

_SQRT2 = float(np.sqrt(2.0))


_STATIC_STATES: dict[str, np.ndarray] = {
    "plus": np.array([1.0, 1.0], dtype=complex) / _SQRT2,
    "minus": np.array([1.0, -1.0], dtype=complex) / _SQRT2,
    "i_plus": np.array([1.0, 1.0j], dtype=complex) / _SQRT2,
    "i_minus": np.array([1.0, -1.0j], dtype=complex) / _SQRT2,
    "bell": np.array([1.0, 0.0, 0.0, 1.0], dtype=complex) / _SQRT2,
}


def get_state(name: str) -> np.ndarray:
    """Resolve a named state to its unit-norm complex vector.

    Parameters
    ----------
    name
        One of the names listed in the module docstring. Case-insensitive
        and whitespace-insensitive.

    Returns
    -------
    np.ndarray
        Complex ``ndarray`` of length :math:`2^n` with unit Euclidean norm.

    Raises
    ------
    ValueError
        If *name* is not recognised, or the ``ghz_<n>`` / ``w_<n>`` qubit
        count cannot be parsed or is non-positive.
    """
    if not isinstance(name, str):
        raise ValueError(f"state name must be a string, got {type(name).__name__}")
    key = name.strip().lower()

    if key in _STATIC_STATES:
        return np.asarray(_STATIC_STATES[key].copy(), dtype=complex)

    if key.startswith("ghz_"):
        n = _parse_qubit_count(key, "ghz_")
        return _ghz_state(n)

    if key.startswith("w_"):
        n = _parse_qubit_count(key, "w_")
        return _w_state(n)

    raise ValueError(
        f"Unknown state name {name!r}. Known names: bell, ghz_<n>, w_<n>, "
        "plus, minus, i_plus, i_minus."
    )


def list_known_states() -> list[str]:
    """Return the list of statically named states (parametric families excluded)."""
    return sorted([*_STATIC_STATES, "ghz_<n>", "w_<n>"])


# --------------------------------------------------------------------------- #
# Internals                                                                   #
# --------------------------------------------------------------------------- #


def _parse_qubit_count(name: str, prefix: str) -> int:
    rest = name[len(prefix) :]
    try:
        n = int(rest)
    except ValueError as exc:
        raise ValueError(
            f"Cannot parse qubit count from {name!r}; expected " f"'{prefix}<positive_integer>'."
        ) from exc
    if n < 1:
        raise ValueError(f"{name!r} requires a positive qubit count, got {n}.")
    return n


def _ghz_state(n: int) -> np.ndarray:
    r"""Return :math:`(|0\rangle^{\otimes n} + |1\rangle^{\otimes n})/\sqrt{2}`."""
    dim = 1 << n
    vec = np.zeros(dim, dtype=complex)
    vec[0] = 1.0
    vec[dim - 1] = 1.0
    return vec / _SQRT2


def _w_state(n: int) -> np.ndarray:
    r"""Return the symmetric single-excitation state on *n* qubits.

    .. math::

        |W_n\rangle = \frac{1}{\sqrt{n}} \sum_{k=0}^{n-1} |2^{k}\rangle
    """
    dim = 1 << n
    vec = np.zeros(dim, dtype=complex)
    for k in range(n):
        vec[1 << k] = 1.0
    return vec / float(np.sqrt(n))
