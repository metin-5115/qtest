"""Hypothesis strategies for quantum state vectors and density matrices.

The strategies use the standard "draw a seed, then numpy-sample"
pattern: Hypothesis controls the integer seed (so examples are
reproducible and shrink toward zero) while heavy numeric work happens
in NumPy. This keeps strategy generation O(1) draws regardless of the
dimension and avoids burning Hypothesis's entropy budget on thousands
of float draws for an 8-qubit (256-dim) state.

What the strategies guarantee
-----------------------------
* :func:`random_states` -- complex 1-D vector of length :math:`2^{n}`,
  unit Euclidean norm. Sampled by drawing independent complex standard
  normals and normalising, which is the standard recipe for a
  **Haar-uniform pure state** (up to a global phase).
* :func:`random_density_matrices` -- complex 2-D array of shape
  :math:`(2^{n}, 2^{n})`, Hermitian, positive semi-definite, unit trace.
  Constructed as :math:`\\rho = AA^{\\dagger} / \\mathrm{tr}(AA^{\\dagger})`
  with ``A`` a random Gaussian matrix; the rank controls how mixed the
  state is.
* :func:`product_states` -- tensor product of *n* independent random
  single-qubit pure states. The resulting :math:`2^{n}`-vector is by
  construction separable across the qubit cut chosen by the Kronecker
  ordering.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from hypothesis import strategies as st

# Hypothesis-controlled integer seeds. Using a 32-bit window keeps the
# numbers easy to reproduce by hand while still giving > 4 billion
# distinct examples per call site.
_MAX_SEED = 2**32 - 1
_SEEDS = st.integers(min_value=0, max_value=_MAX_SEED)

IntLike = Union[int, st.SearchStrategy[int]]


def _resolve_int(
    draw: st.DrawFn, value: IntLike, *, name: str, minimum: int = 1
) -> int:
    resolved = draw(value) if isinstance(value, st.SearchStrategy) else value
    if not isinstance(resolved, int) or isinstance(resolved, bool) or resolved < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {resolved!r}")
    return resolved


def _haar_pure_state(rng: np.random.Generator, dim: int) -> np.ndarray:
    """Sample a Haar-uniform pure state of dimension *dim*.

    The construction "complex standard normal + normalise" is the
    well-known shortcut for the uniform measure on the unit sphere in
    :math:`\\mathbb{C}^{d}`, which projects to the Haar measure on
    pure states (up to a global phase that has no physical meaning).
    """
    vec = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    vec /= np.linalg.norm(vec)
    return vec


# --------------------------------------------------------------------------- #
# Public strategies                                                           #
# --------------------------------------------------------------------------- #


@st.composite
def random_states(draw: st.DrawFn, n_qubits: IntLike) -> np.ndarray:
    """Strategy yielding Haar-random pure state vectors on *n_qubits*.

    Parameters
    ----------
    n_qubits
        Positive integer or a Hypothesis strategy of positive integers.

    Returns
    -------
    ndarray
        Complex 1-D array of length :math:`2^{n_{\\text{qubits}}}` with
        unit Euclidean norm.
    """
    n = _resolve_int(draw, n_qubits, name="n_qubits", minimum=1)
    seed = draw(_SEEDS)
    rng = np.random.default_rng(seed)
    return _haar_pure_state(rng, 2**n)


@st.composite
def product_states(draw: st.DrawFn, n_qubits: IntLike) -> np.ndarray:
    """Strategy yielding tensor-product (separable) pure states.

    Each of the *n_qubits* single-qubit factors is drawn Haar-uniformly
    and the result is their Kronecker product, taken in qubit-index
    order. The output is therefore guaranteed to be separable across
    every cut; entanglement-detection tests can use this as a "should
    not be entangled" oracle.
    """
    n = _resolve_int(draw, n_qubits, name="n_qubits", minimum=1)
    # A single seed per call: a per-qubit seed would shrink less cleanly
    # and offers no real benefit since one rng can sample n factors.
    seed = draw(_SEEDS)
    rng = np.random.default_rng(seed)

    state = np.array([1.0 + 0j])
    for _ in range(n):
        state = np.kron(state, _haar_pure_state(rng, 2))
    return state


@st.composite
def random_density_matrices(
    draw: st.DrawFn,
    n_qubits: IntLike,
    rank: Optional[IntLike] = None,
) -> np.ndarray:
    """Strategy yielding valid density matrices on *n_qubits*.

    The matrix is constructed via the Ginibre ensemble: draw a complex
    Gaussian :math:`A \\in \\mathbb{C}^{d \\times r}` and form
    :math:`\\rho = AA^{\\dagger}/\\mathrm{tr}(AA^{\\dagger})`. The result
    is Hermitian, positive semi-definite, and has unit trace; its rank
    equals ``rank``.

    Parameters
    ----------
    n_qubits
        Positive integer or strategy. Dimension is :math:`d = 2^{n}`.
    rank
        Target rank of the density matrix. ``None`` (default) uses the
        full dimension, giving a generically full-rank mixed state.
        ``rank = 1`` produces pure states.
    """
    n = _resolve_int(draw, n_qubits, name="n_qubits", minimum=1)
    dim = 2**n

    if rank is None:
        r = dim
    else:
        r = _resolve_int(draw, rank, name="rank", minimum=1)
        if r > dim:
            raise ValueError(f"rank ({r}) cannot exceed dimension 2**n_qubits ({dim})")

    seed = draw(_SEEDS)
    rng = np.random.default_rng(seed)

    a = rng.standard_normal((dim, r)) + 1j * rng.standard_normal((dim, r))
    rho = a @ a.conj().T
    rho /= np.trace(rho).real
    return rho


__all__ = [
    "product_states",
    "random_density_matrices",
    "random_states",
]
