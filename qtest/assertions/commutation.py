"""``assert_commutes`` — operator commutation / anticommutation assertion.

Checks whether two operators commute (``AB == BA``) or anticommute
(``AB == -BA``) by comparing the (anti)commutator to the zero matrix. Operands
may be raw matrices, gate objects (anything with ``to_matrix()``), or circuits
(whose unitary is extracted via the backend) — the same coercion used by
:func:`qtest.assert_unitary`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qtest.assertions.unitary import _to_matrix
from qtest.backends import Backend

_FP_SLACK = 1e-14


def assert_commutes(
    a: Any,
    b: Any,
    tolerance: float = 1e-9,
    anti: bool = False,
    backend: Backend | None = None,
    msg: str | None = None,
) -> None:
    r"""Assert that operators *a* and *b* commute (or anticommute).

    Parameters
    ----------
    a, b
        Operators as ``np.ndarray``, gate objects with ``to_matrix()``, or
        circuits. Must have identical shape.
    tolerance
        Maximum permitted entry-wise magnitude of the (anti)commutator
        :math:`\max_{ij}|C_{ij}|`, where ``C = AB - BA`` (commutator) or
        ``C = AB + BA`` (anticommutator when ``anti=True``).
    anti
        When ``True`` check anticommutation (``AB + BA ≈ 0``) instead of
        commutation.
    backend
        Backend used to extract unitaries from circuit-shaped operands.
    msg
        Optional prefix prepended to the assertion failure message.

    Raises
    ------
    ValueError
        For malformed or mismatched operand shapes.
    AssertionError
        When *a* and *b* do not (anti)commute within *tolerance*.
    """
    if not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool) or tolerance < 0.0:
        raise ValueError(f"tolerance must be a non-negative real number, got {tolerance!r}")

    mat_a = _to_matrix(a, backend)
    mat_b = _to_matrix(b, backend)
    for name, m in (("a", mat_a), ("b", mat_b)):
        if m.ndim != 2 or m.shape[0] != m.shape[1] or m.size == 0:
            raise ValueError(f"operand {name} must be a non-empty square matrix, got {m.shape}")
    if mat_a.shape != mat_b.shape:
        raise ValueError(f"operands must have the same shape; got {mat_a.shape} and {mat_b.shape}.")

    ab = mat_a @ mat_b
    ba = mat_b @ mat_a
    commutator = ab + ba if anti else ab - ba
    deviation = float(np.max(np.abs(commutator)))
    if deviation <= tolerance + _FP_SLACK:
        return

    raise AssertionError(
        _format_failure(a=a, b=b, anti=anti, deviation=deviation, tolerance=tolerance, user_msg=msg)
    )


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _operand_summary(op: Any) -> str:
    if isinstance(op, np.ndarray):
        return f"<ndarray shape={op.shape}>"
    name = getattr(op, "name", None)
    return str(name) if name else f"<{type(op).__name__}>"


def _format_failure(
    *,
    a: Any,
    b: Any,
    anti: bool,
    deviation: float,
    tolerance: float,
    user_msg: str | None,
) -> str:
    kind = "anticommute" if anti else "commute"
    symbol = "AB + BA" if anti else "AB - BA"
    lines: list[str] = []
    if user_msg:
        lines.extend([user_msg, ""])
    lines.append(f"Operators do not {kind}")
    lines.append("")
    lines.append(f"  A: {_operand_summary(a)}")
    lines.append(f"  B: {_operand_summary(b)}")
    lines.append(f"  max|{symbol}|: {deviation:.3e}")
    lines.append(f"  Tolerance: {tolerance:g}")
    return "\n".join(lines)
