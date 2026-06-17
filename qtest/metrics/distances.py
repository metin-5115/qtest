"""Distance and divergence measures for distributions, states, and operators.

All distance functions are symmetric in their arguments, return a non-negative
``float``, and raise :class:`ValueError` for malformed input.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

# Numerical tolerances used when validating probability distributions.
_PROB_SUM_ATOL: float = 1e-6
_NEG_PROB_ATOL: float = 1e-9


def _validate_probability_distribution(name: str, dist: Mapping[str, float]) -> None:
    """Validate that *dist* is a non-empty discrete probability distribution.

    Allows tiny floating-point slack: probabilities must be in
    ``[-_NEG_PROB_ATOL, 1 + _PROB_SUM_ATOL]`` and sum to ``1`` within
    ``_PROB_SUM_ATOL``.

    Raises
    ------
    ValueError
        If ``dist`` is empty, contains non-finite or out-of-range values, or
        the entries do not sum to one.
    """
    if not dist:
        raise ValueError(f"{name} must be a non-empty probability distribution.")
    for key, value in dist.items():
        if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
            raise ValueError(f"{name}[{key!r}] = {value!r} is not a finite real number.")
        if value < -_NEG_PROB_ATOL or value > 1.0 + _PROB_SUM_ATOL:
            raise ValueError(f"{name}[{key!r}] = {value} is outside [0, 1].")
    total = float(sum(dist.values()))
    if not np.isclose(total, 1.0, atol=_PROB_SUM_ATOL):
        raise ValueError(
            f"{name} probabilities must sum to 1 (within {_PROB_SUM_ATOL}); got {total}."
        )


def total_variation_distance(
    p: Mapping[str, float],
    q: Mapping[str, float],
) -> float:
    r"""Total variation distance between two discrete distributions.

    .. math::

        \mathrm{TVD}(P, Q) \;=\; \tfrac{1}{2} \sum_{x} \left| P(x) - Q(x) \right|

    Missing keys are treated as zero probability. Both inputs must individually
    be valid probability distributions (non-negative, summing to one).

    Parameters
    ----------
    p, q : Mapping[str, float]
        Probability mass functions keyed by outcome label.

    Returns
    -------
    float
        TVD in :math:`[0, 1]`.

    Raises
    ------
    ValueError
        If either input is empty or not a valid probability distribution.
    """
    _validate_probability_distribution("p", p)
    _validate_probability_distribution("q", q)

    keys = set(p.keys()) | set(q.keys())
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def hellinger_distance(
    p: Mapping[str, float],
    q: Mapping[str, float],
) -> float:
    r"""Hellinger distance between two discrete distributions.

    .. math::

        H(P, Q) \;=\; \frac{1}{\sqrt{2}} \sqrt{ \sum_{x} \left( \sqrt{P(x)} - \sqrt{Q(x)} \right)^{2} }

    Missing keys are treated as zero probability. The result lies in
    :math:`[0, 1]`, with ``0`` iff :math:`P = Q` and ``1`` iff the supports
    are disjoint.

    Parameters
    ----------
    p, q : Mapping[str, float]
        Probability mass functions keyed by outcome label.

    Returns
    -------
    float
        Hellinger distance in :math:`[0, 1]`.

    Raises
    ------
    ValueError
        If either input is empty or not a valid probability distribution.
    """
    _validate_probability_distribution("p", p)
    _validate_probability_distribution("q", q)

    keys = set(p.keys()) | set(q.keys())
    squared_diff_sum = sum(
        (np.sqrt(max(p.get(k, 0.0), 0.0)) - np.sqrt(max(q.get(k, 0.0), 0.0))) ** 2 for k in keys
    )
    return float(np.sqrt(squared_diff_sum) / np.sqrt(2.0))


def _as_density_matrix(arr: np.ndarray, name: str) -> np.ndarray:
    r"""Coerce a state vector or square 2-D array into a density matrix.

    A 1-D input ``|psi>`` is lifted to the rank-1 projector
    :math:`|\psi\rangle\langle\psi|`. A square 2-D input is returned unchanged
    (cast to ``complex``).

    Raises
    ------
    ValueError
        If ``arr`` is neither 1-D nor a square 2-D array.
    """
    a = np.asarray(arr)
    if a.ndim == 1:
        psi = a.astype(complex)
        return np.outer(psi, psi.conj())
    if a.ndim == 2 and a.shape[0] == a.shape[1]:
        return a.astype(complex)
    raise ValueError(
        f"{name} must be a 1-D state vector or a square 2-D density matrix; "
        f"got shape {a.shape}."
    )


def _matrix_sqrt_psd(mat: np.ndarray) -> np.ndarray:
    """Principal square root of a Hermitian positive semi-definite matrix.

    Computed via eigendecomposition. Eigenvalues that fall slightly below
    zero from floating-point noise are clipped to zero before being rooted.
    """
    # Symmetrise to suppress non-Hermitian noise from upstream arithmetic.
    sym = 0.5 * (mat + mat.conj().T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals_clipped = np.clip(eigvals.real, 0.0, None)
    sqrt_mat: np.ndarray = (eigvecs * np.sqrt(eigvals_clipped)) @ eigvecs.conj().T
    return sqrt_mat


def fidelity(state1: np.ndarray, state2: np.ndarray) -> float:
    r"""Quantum fidelity between two states.

    For pure state vectors :math:`|\psi\rangle, |\phi\rangle`:

    .. math::

        F(\psi, \phi) \;=\; |\langle \psi | \phi \rangle|^{2}.

    For general (mixed) density matrices :math:`\rho, \sigma` the
    Uhlmann–Jozsa fidelity is used:

    .. math::

        F(\rho, \sigma) \;=\; \left( \mathrm{tr}\, \sqrt{ \sqrt{\rho}\, \sigma\, \sqrt{\rho} } \right)^{2}.

    The two arguments may independently be state vectors or density matrices;
    state vectors are lifted to rank-1 projectors on the fly. The output is
    clipped to :math:`[0, 1]` to absorb floating-point overshoot.

    Parameters
    ----------
    state1, state2 : np.ndarray
        State vectors (1-D) or density matrices (2-D, square).

    Returns
    -------
    float
        Fidelity in :math:`[0, 1]`.

    Raises
    ------
    ValueError
        If shapes are incompatible or inputs are neither 1-D nor square 2-D.
    """
    a = np.asarray(state1)
    b = np.asarray(state2)

    # Fast path: both arguments are state vectors.
    if a.ndim == 1 and b.ndim == 1:
        if a.shape != b.shape:
            raise ValueError(
                f"State vectors must have the same length; got {a.shape} and {b.shape}."
            )
        inner = np.vdot(a, b)
        return float(np.clip(np.abs(inner) ** 2, 0.0, 1.0))

    # Lift state vectors to density matrices for the general path.
    rho = _as_density_matrix(a, "state1")
    sigma = _as_density_matrix(b, "state2")
    if rho.shape != sigma.shape:
        raise ValueError(
            f"States must have compatible dimension; got {rho.shape} and {sigma.shape}."
        )

    # Pure-state shortcuts: F(|psi>, sigma) = <psi| sigma |psi>.
    if a.ndim == 1:
        psi = a.astype(complex)
        return float(np.clip(np.real(psi.conj() @ sigma @ psi), 0.0, 1.0))
    if b.ndim == 1:
        phi = b.astype(complex)
        return float(np.clip(np.real(phi.conj() @ rho @ phi), 0.0, 1.0))

    # General mixed–mixed case via Uhlmann's formula.
    sqrt_rho = _matrix_sqrt_psd(rho)
    middle = sqrt_rho @ sigma @ sqrt_rho
    middle_eigs = np.linalg.eigvalsh(0.5 * (middle + middle.conj().T)).real
    trace_sqrt = float(np.sum(np.sqrt(np.clip(middle_eigs, 0.0, None))))
    return float(np.clip(trace_sqrt**2, 0.0, 1.0))


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    r"""Trace distance between two quantum states.

    .. math::

        D(\rho, \sigma) \;=\; \tfrac{1}{2}\, \mathrm{tr}\, |\rho - \sigma|

    where :math:`|A| \equiv \sqrt{A^{\dagger} A}`. For Hermitian
    ``rho - sigma`` this reduces to half the sum of the absolute eigenvalues.

    Either argument may be a state vector (lifted to a rank-1 projector) or
    a density matrix.

    Parameters
    ----------
    rho, sigma : np.ndarray
        State vectors (1-D) or density matrices (2-D, square).

    Returns
    -------
    float
        Trace distance in :math:`[0, 1]`.

    Raises
    ------
    ValueError
        If shapes are incompatible.
    """
    rho_m = _as_density_matrix(rho, "rho")
    sigma_m = _as_density_matrix(sigma, "sigma")
    if rho_m.shape != sigma_m.shape:
        raise ValueError(
            f"Operators must have the same shape; got {rho_m.shape} and {sigma_m.shape}."
        )

    diff = rho_m - sigma_m
    # diff is Hermitian by construction; symmetrise to tame numerical noise.
    diff_sym = 0.5 * (diff + diff.conj().T)
    if np.allclose(diff, diff_sym, atol=1e-10):
        eigvals = np.linalg.eigvalsh(diff_sym).real
        return float(np.clip(0.5 * float(np.sum(np.abs(eigvals))), 0.0, 1.0))
    # Fallback for non-Hermitian operands: singular values.
    svals = np.linalg.svd(diff, compute_uv=False)
    return float(np.clip(0.5 * float(np.sum(svals)), 0.0, 1.0))


def hilbert_schmidt_distance(op1: np.ndarray, op2: np.ndarray) -> float:
    r"""Hilbert–Schmidt distance between two operators.

    .. math::

        d_{\mathrm{HS}}(A, B) \;=\; \sqrt{ \mathrm{tr}\!\left( (A - B)^{\dagger} (A - B) \right) }
                          \;=\; \| A - B \|_{\mathrm{F}}

    i.e. the Frobenius norm of the difference.

    Parameters
    ----------
    op1, op2 : np.ndarray
        Two-dimensional operators of identical shape.

    Returns
    -------
    float
        Non-negative HS distance.

    Raises
    ------
    ValueError
        If either input is not 2-D, or if shapes differ.
    """
    a = np.asarray(op1)
    b = np.asarray(op2)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"Operators must be 2-D; got shapes {a.shape} and {b.shape}.")
    if a.shape != b.shape:
        raise ValueError(f"Operators must have the same shape; got {a.shape} and {b.shape}.")
    return float(np.linalg.norm(a - b, ord="fro"))
