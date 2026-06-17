"""Entanglement and information-theoretic measures (pure NumPy).

These back the entanglement-aware assertions (``assert_entangled`` /
``assert_separable``). Like :mod:`qtest.metrics.distances`, every function here
is pure: it validates input, never mutates its arguments, and works on plain
NumPy arrays so it can be used without any quantum SDK.
"""

from __future__ import annotations

import numpy as np

_LOG_FLOOR = 1e-12  # eigenvalues below this contribute 0 to the entropy sum


def _as_square_matrix(arr: np.ndarray, name: str) -> np.ndarray:
    """Coerce a state vector (1-D) or square matrix (2-D) to a density matrix."""
    a = np.asarray(arr, dtype=complex)
    if a.ndim == 1:
        return np.outer(a, a.conj())
    if a.ndim == 2 and a.shape[0] == a.shape[1]:
        return a
    raise ValueError(f"{name} must be a 1-D state vector or a square 2-D matrix; got {a.shape}.")


def _infer_num_qubits(dim: int) -> int:
    """Return ``log2(dim)`` for a power-of-two *dim*, else raise."""
    if dim < 1 or (dim & (dim - 1)) != 0:
        raise ValueError(f"dimension must be a positive power of 2, got {dim}")
    return dim.bit_length() - 1


def partial_trace(matrix: np.ndarray, keep: list[int], num_qubits: int | None = None) -> np.ndarray:
    r"""Partial trace of a state / density matrix, keeping *keep* qubits.

    Parameters
    ----------
    matrix
        A state vector (1-D, lifted to a projector) or a density matrix
        (2-D, square) over ``num_qubits`` qubits.
    keep
        Qubit indices (0-based) of the subsystem to keep; the rest are traced
        out. Order of the returned subsystem follows ascending qubit index.
    num_qubits
        Total qubit count. Inferred from the matrix dimension when ``None``.

    Returns
    -------
    np.ndarray
        Reduced density matrix of shape ``(2**k, 2**k)`` with ``k = len(keep)``.
    """
    rho = _as_square_matrix(matrix, "matrix")
    n = _infer_num_qubits(rho.shape[0]) if num_qubits is None else num_qubits
    if rho.shape[0] != 2**n:
        raise ValueError(f"matrix dimension {rho.shape[0]} does not match num_qubits={n}.")

    keep_sorted = sorted(set(keep))
    if any(q < 0 or q >= n for q in keep_sorted):
        raise ValueError(f"keep indices must be in [0, {n}); got {keep}.")

    tensor = rho.reshape([2] * (2 * n))
    rows = list(range(n))  # qubit id at each remaining row axis, in order
    cols = list(range(n))  # qubit id at each remaining col axis, in order
    for q in range(n):
        if q in keep_sorted:
            continue
        r = rows.index(q)
        c = len(rows) + cols.index(q)
        tensor = np.trace(tensor, axis1=r, axis2=c)
        rows.remove(q)
        cols.remove(q)

    k = len(keep_sorted)
    return tensor.reshape(2**k, 2**k)


def purity(rho: np.ndarray) -> float:
    r"""Purity :math:`\mathrm{tr}(\rho^2)` of a state / density matrix.

    Equals ``1`` for a pure state and ``< 1`` for a mixed one (minimum
    ``1/d`` for the maximally mixed state).
    """
    mat = _as_square_matrix(rho, "rho")
    return float(np.clip(np.real(np.trace(mat @ mat)), 0.0, 1.0))


def von_neumann_entropy(rho: np.ndarray, base: float = 2.0) -> float:
    r"""Von Neumann entropy :math:`S(\rho) = -\mathrm{tr}(\rho \log \rho)`.

    Parameters
    ----------
    rho
        State vector (entropy ``0``) or density matrix.
    base
        Logarithm base. ``2`` (default) gives the entropy in bits — a single
        qubit's maximally mixed state then has entropy ``1``.
    """
    mat = _as_square_matrix(rho, "rho")
    eigvals = np.linalg.eigvalsh(0.5 * (mat + mat.conj().T)).real
    eigvals = eigvals[eigvals > _LOG_FLOOR]
    if eigvals.size == 0:
        return 0.0
    entropy = -float(np.sum(eigvals * (np.log(eigvals) / np.log(base))))
    return max(entropy, 0.0)


def entanglement_entropy(
    state: np.ndarray,
    qubits: list[int],
    num_qubits: int | None = None,
    base: float = 2.0,
) -> float:
    r"""Entanglement entropy of subsystem *qubits* versus the rest.

    For a pure global state this is the von Neumann entropy of the reduced
    density matrix on *qubits*; it is ``0`` iff the bipartition is a product
    state (no entanglement across the cut) and positive otherwise.
    """
    reduced = partial_trace(state, qubits, num_qubits)
    return von_neumann_entropy(reduced, base=base)
