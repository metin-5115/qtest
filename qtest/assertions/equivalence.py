"""``assert_circuit_equivalent`` — verify two circuits implement the same unitary.

Three comparison methods, with ``"auto"`` selecting one based on circuit size:

* ``"unitary"`` (default for ≤ 8 qubits) — **phase-insensitive**. Computes
  the *process fidelity*
  :math:`F = |\\mathrm{tr}(U_a^\\dagger U_b)|^{2} / d^{2}` and fails when
  :math:`1 - F > \\text{tolerance}`. Insensitive to a global phase factor.
* ``"hilbert_schmidt"`` (default for 9–15 qubits) — **phase-sensitive**.
  Computes :math:`\\lVert U_a - U_b \\rVert_{F}` via
  :func:`qtest.metrics.hilbert_schmidt_distance` and fails when the
  distance exceeds ``tolerance``.
* ``"random_sampling"`` (default for > 15 qubits) — samples ``n_samples``
  Haar-uniform pure states, computes per-state fidelity of the two
  evolved states, and fails when ``1 - mean(fidelity) > tolerance``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qtest.backends import Backend
from qtest.backends.registry import get_backend
from qtest.config import get_config
from qtest.metrics import fidelity, hilbert_schmidt_distance

_VALID_METHODS = frozenset({"auto", "unitary", "hilbert_schmidt", "random_sampling"})
_FP_SLACK = 1e-12


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #


def assert_circuit_equivalent(
    circuit_a: Any,
    circuit_b: Any,
    method: str = "auto",
    tolerance: float = 1e-6,
    n_samples: int = 100,
    backend: Backend | None = None,
    seed: int | None = None,
    msg: str | None = None,
) -> None:
    """Assert that *circuit_a* and *circuit_b* implement equivalent unitaries.

    Parameters
    ----------
    circuit_a, circuit_b
        Backend-native circuit objects with identical qubit counts. Must
        consist only of unitary operations (no measurements / resets).
    method
        One of ``"auto"``, ``"unitary"``, ``"hilbert_schmidt"``,
        ``"random_sampling"``. See module docstring for semantics. With
        ``"auto"`` the method is chosen from the qubit count.
    tolerance
        Maximum permitted infidelity (``"unitary"`` / ``"random_sampling"``)
        or HS distance (``"hilbert_schmidt"``). Defaults to ``1e-6``.
    n_samples
        Number of random Haar states for ``"random_sampling"``. Ignored
        otherwise. Defaults to ``100``.
    backend
        Backend used to extract unitaries. Defaults to the configured
        default backend.
    seed
        Seed for the random state generator (``"random_sampling"`` only).
    msg
        Optional prefix prepended to the assertion failure message.

    Raises
    ------
    ValueError
        For invalid ``method``, mismatched qubit counts, non-positive
        ``n_samples``, etc.
    AssertionError
        When the circuits diverge beyond ``tolerance``.
    """
    if method not in _VALID_METHODS:
        raise ValueError(f"Unknown method {method!r}. Must be one of {sorted(_VALID_METHODS)}.")
    if not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool):
        raise ValueError(f"tolerance must be a real number, got {tolerance!r}")
    if tolerance < 0.0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance}")
    if not isinstance(n_samples, int) or isinstance(n_samples, bool) or n_samples <= 0:
        raise ValueError(f"n_samples must be a positive integer, got {n_samples!r}")

    n_qubits_a = _qubit_count(circuit_a)
    n_qubits_b = _qubit_count(circuit_b)
    if n_qubits_a != n_qubits_b:
        raise ValueError(
            f"Qubit count mismatch: circuit_a has {n_qubits_a}, " f"circuit_b has {n_qubits_b}."
        )
    n = n_qubits_a

    chosen_method = _auto_select_method(n) if method == "auto" else method

    if backend is None:
        backend = get_backend(get_config().default_backend)

    u_a = np.asarray(backend.get_unitary(circuit_a), dtype=complex)
    u_b = np.asarray(backend.get_unitary(circuit_b), dtype=complex)
    if u_a.shape != u_b.shape:
        raise ValueError(
            f"Backend produced unitaries of mismatched shape: " f"{u_a.shape} vs {u_b.shape}"
        )

    if chosen_method == "unitary":
        measured, label = _check_unitary(u_a, u_b)
        passed = measured <= tolerance + _FP_SLACK
    elif chosen_method == "hilbert_schmidt":
        measured = hilbert_schmidt_distance(u_a, u_b)
        label = "Hilbert-Schmidt distance"
        passed = measured <= tolerance + _FP_SLACK
    else:  # random_sampling
        measured, label = _check_random_sampling(u_a, u_b, n, n_samples, seed)
        passed = measured <= tolerance + _FP_SLACK

    if not passed:
        raise AssertionError(
            _format_failure(
                circuit_a=circuit_a,
                circuit_b=circuit_b,
                method=chosen_method,
                auto_selected=(method == "auto"),
                n_qubits=n,
                tolerance=tolerance,
                measured=measured,
                label=label,
                n_samples=n_samples if chosen_method == "random_sampling" else None,
                user_msg=msg,
                u_a=u_a,
                u_b=u_b,
            )
        )


# --------------------------------------------------------------------------- #
# Internal helpers                                                            #
# --------------------------------------------------------------------------- #


def _auto_select_method(n_qubits: int) -> str:
    if n_qubits <= 8:
        return "unitary"
    if n_qubits <= 15:
        return "hilbert_schmidt"
    return "random_sampling"


def _qubit_count(circuit: Any) -> int:
    n = getattr(circuit, "num_qubits", None)
    if n is None or not isinstance(n, int) or n <= 0:
        raise ValueError(
            f"Cannot determine qubit count for {type(circuit).__name__!r}; "
            "circuit must expose `num_qubits` attribute."
        )
    return n


def _check_unitary(u_a: np.ndarray, u_b: np.ndarray) -> tuple[float, str]:
    """Phase-insensitive comparison via process fidelity."""
    d = u_a.shape[0]
    inner = np.trace(u_a.conj().T @ u_b)
    f_process = float(np.clip(abs(inner) ** 2 / (d * d), 0.0, 1.0))
    return 1.0 - f_process, "process infidelity"


def _check_random_sampling(
    u_a: np.ndarray,
    u_b: np.ndarray,
    n_qubits: int,
    n_samples: int,
    seed: int | None,
) -> tuple[float, str]:
    """Average state infidelity over ``n_samples`` Haar-random pure states."""
    rng = np.random.default_rng(seed)
    fidelities = np.empty(n_samples, dtype=float)
    for i in range(n_samples):
        psi = _generate_random_state(n_qubits, rng=rng)
        phi_a = u_a @ psi
        phi_b = u_b @ psi
        fidelities[i] = float(np.clip(fidelity(phi_a, phi_b), 0.0, 1.0))
    mean_fid = float(np.mean(fidelities))
    return 1.0 - mean_fid, f"mean infidelity over {n_samples} samples"


def _generate_random_state(
    n_qubits: int,
    seed: int | None = None,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample a Haar-uniform pure state on *n_qubits*.

    A complex standard-normal vector, divided by its Euclidean norm, is
    uniform on the unit sphere in :math:`\\mathbb{C}^{2^n}` — which is the
    Haar measure on pure states.

    Parameters
    ----------
    n_qubits
        Number of qubits (state dimension is :math:`2^{n}`).
    seed
        Convenience seed (creates a fresh ``np.random.Generator``). Ignored
        if *rng* is provided.
    rng
        Optional pre-built generator (preferred when sampling many states
        in a loop — avoids re-seeding bias).
    """
    if n_qubits < 1:
        raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
    if rng is None:
        rng = np.random.default_rng(seed)
    dim = 1 << n_qubits
    real = rng.standard_normal(dim)
    imag = rng.standard_normal(dim)
    vec = real + 1j * imag
    return vec / float(np.linalg.norm(vec))


def _format_failure(
    *,
    circuit_a: Any,
    circuit_b: Any,
    method: str,
    auto_selected: bool,
    n_qubits: int,
    tolerance: float,
    measured: float,
    label: str,
    n_samples: int | None,
    user_msg: str | None,
    u_a: np.ndarray | None = None,
    u_b: np.ndarray | None = None,
) -> str:
    lines: list[str] = []
    if user_msg:
        lines.append(user_msg)
        lines.append("")

    lines.append("Circuits are not equivalent")
    lines.append("")
    lines.append(f"  Circuit A: {_circuit_summary(circuit_a)}")
    lines.append(f"  Circuit B: {_circuit_summary(circuit_b)}")
    lines.append(f"  Qubits: {n_qubits}")
    method_line = f"  Method: {method}"
    if auto_selected:
        method_line += " (auto)"
    lines.append(method_line)
    if n_samples is not None:
        lines.append(f"  Samples: {n_samples}")
    lines.append(f"  Tolerance: {tolerance:g}")
    lines.append(f"  Measured {label}: {measured:.6e}")
    lines.append(f"    -> {_severity_label(measured, tolerance, label)}")

    # Severity bar — only meaningful for fidelity-based metrics (bounded in [0, 1]).
    if "infidelity" in label:
        bar = _severity_bar(measured)
        lines.append("")
        lines.append(f"  Severity:  0.0  {bar}  1.0")
        lines.append("             identical                          orthogonal")

    # Basis-state divergence sampling — show concrete inputs that produce different
    # outputs in the two circuits. Skipped for large circuits (>6 qubits) where the
    # full table would be unreadable, and for cases where the unitary matrices
    # were not computed.
    if (
        u_a is not None
        and u_b is not None
        and n_qubits <= 6
        and measured > tolerance
    ):
        divergent = _find_divergent_inputs(u_a, u_b, n_qubits, top_k=2)
        if divergent and divergent[0][1] > max(tolerance, 1e-9):
            lines.append("")
            lines.append("  Where they diverge most (sampled basis-state inputs):")
            for input_idx, infid, psi_a, psi_b in divergent:
                if infid <= max(tolerance, 1e-9):
                    break
                label_in = _format_basis_label(input_idx, n_qubits)
                top_a = _top_probabilities(psi_a, n_qubits, k=3)
                top_b = _top_probabilities(psi_b, n_qubits, k=3)
                lines.append(f"    input {label_in}")
                lines.append(f"      A output: {top_a}")
                lines.append(f"      B output: {top_b}")
                lines.append(f"      per-input infidelity: {infid:.3e}")

    # Actionable hint — the single most useful thing a confused user can do.
    lines.append("")
    lines.append("  Tip: print(circuit_a.draw('text')) and print(circuit_b.draw('text'))")
    lines.append("       to compare them gate by gate. The bug is usually in the first")
    lines.append("       gate position where the two listings differ.")

    return "\n".join(lines)


def _circuit_summary(circuit: Any) -> str:
    name = getattr(circuit, "name", None)
    n_qubits = getattr(circuit, "num_qubits", None)
    if name and n_qubits is not None:
        return f"{name} ({n_qubits} qubits)"
    if name:
        return str(name)
    return f"<{type(circuit).__name__}>"


# --------------------------------------------------------------------------- #
# Failure-message helpers (severity scale + basis-state divergence sampling)  #
# --------------------------------------------------------------------------- #


def _severity_label(measured: float, tolerance: float, label: str) -> str:
    """Plain-language description of how far apart the two circuits are."""
    if "infidelity" in label:
        # Process / state infidelity is bounded in [0, 1].
        if measured < 1e-9:
            return "numerically identical"
        if measured < 1e-6:
            return "essentially identical (within numerical noise)"
        if measured < 1e-3:
            return "tiny mismatch (< 0.1%)"
        if measured < 0.01:
            return "small mismatch (< 1%)"
        if measured < 0.1:
            return "moderate mismatch (< 10%)"
        if measured < 0.5:
            return "large mismatch"
        if measured < 0.99:
            return "very large mismatch"
        return "maximum -- circuits are essentially orthogonal"

    # Hilbert-Schmidt distance is unbounded; describe relative to tolerance.
    if tolerance <= 0.0:
        return "above the configured tolerance"
    ratio = measured / tolerance
    if ratio < 10:
        return f"small (~{ratio:.1f}x tolerance)"
    if ratio < 100:
        return f"moderate (~{ratio:.0f}x tolerance)"
    if ratio < 1000:
        return f"large (~{ratio:.0f}x tolerance)"
    return f"huge (~{ratio:.0e}x tolerance)"


def _severity_bar(measured: float, width: int = 30) -> str:
    """A single-line text bar visualising the measured value on [0, 1]."""
    clipped = float(np.clip(measured, 0.0, 1.0))
    filled = round(clipped * width)
    return "#" * filled + "-" * (width - filled)


def _format_basis_label(index: int, n_qubits: int) -> str:
    """Render a computational basis index as ``|bits>``."""
    bits = format(index, f"0{n_qubits}b")
    return f"|{bits}>"


def _top_probabilities(amplitudes: np.ndarray, n_qubits: int, k: int = 3) -> str:
    """Return the top-``k`` measurement outcomes formatted as ``|bits> (prob)``."""
    probs = np.abs(amplitudes) ** 2
    if probs.sum() == 0.0:  # pragma: no cover — defensive only
        return "no support"
    order = np.argsort(probs)[::-1]
    parts: list[str] = []
    for idx in order[:k]:
        p = float(probs[idx])
        if p < 1e-4:
            break
        parts.append(f"{_format_basis_label(int(idx), n_qubits)} {p:.3f}")
    return ", ".join(parts) if parts else "no significant outcomes"


def _find_divergent_inputs(
    u_a: np.ndarray,
    u_b: np.ndarray,
    n_qubits: int,
    top_k: int = 2,
) -> list[tuple[int, float, np.ndarray, np.ndarray]]:
    """Locate basis-state inputs whose outputs differ the most between the unitaries.

    For each computational basis input ``|i>`` we compute the two output states
    ``U_a |i>`` and ``U_b |i>``, then their state infidelity. The basis inputs
    are returned sorted by infidelity, largest first.
    """
    dim = 1 << n_qubits
    results: list[tuple[int, float, np.ndarray, np.ndarray]] = []
    for i in range(dim):
        psi_a = u_a[:, i]
        psi_b = u_b[:, i]
        inner = abs(np.vdot(psi_a, psi_b)) ** 2
        infidelity = float(np.clip(1.0 - inner, 0.0, 1.0))
        results.append((i, infidelity, psi_a, psi_b))
    results.sort(key=lambda row: row[1], reverse=True)
    return results[:top_k]
