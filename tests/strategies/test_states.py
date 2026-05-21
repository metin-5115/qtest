"""Tests for :mod:`qtest.strategies.states`.

Covers structural invariants (norm, hermiticity, trace), separability
of :func:`product_states`, and reproducibility under Hypothesis's
``@seed`` decorator.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, seed, settings
from hypothesis import strategies as st

from qtest.strategies import (
    product_states,
    random_density_matrices,
    random_states,
)

_FAST_SETTINGS = settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


# --------------------------------------------------------------------------- #
# random_states                                                               #
# --------------------------------------------------------------------------- #


@_FAST_SETTINGS
@given(psi=random_states(n_qubits=3))
def test_random_states_have_unit_norm(psi: np.ndarray) -> None:
    assert psi.shape == (8,)
    assert np.isclose(np.linalg.norm(psi), 1.0, atol=1e-12)


@_FAST_SETTINGS
@given(psi=random_states(n_qubits=1))
def test_random_states_complex_dtype(psi: np.ndarray) -> None:
    assert np.iscomplexobj(psi)


@_FAST_SETTINGS
@given(
    psi=random_states(n_qubits=st.integers(min_value=1, max_value=4)),
)
def test_random_states_dimension_matches_qubits(psi: np.ndarray) -> None:
    dim = psi.shape[0]
    # dim must be a power of two between 2 and 16
    assert dim in (2, 4, 8, 16)


# --------------------------------------------------------------------------- #
# product_states                                                              #
# --------------------------------------------------------------------------- #


@_FAST_SETTINGS
@given(psi=product_states(n_qubits=3))
def test_product_states_have_unit_norm(psi: np.ndarray) -> None:
    assert psi.shape == (8,)
    assert np.isclose(np.linalg.norm(psi), 1.0, atol=1e-12)


def _bipartite_schmidt_rank(psi: np.ndarray, n_left: int, n_right: int) -> int:
    """Return the Schmidt rank of *psi* across a ``(n_left | n_right)`` cut.

    A product state has Schmidt rank 1 across every cut; entangled
    states have rank >= 2. We round small singular values to zero to
    absorb double-precision round-off (~1e-16) introduced by the
    Kronecker product chain.
    """
    matrix = psi.reshape(2**n_left, 2**n_right)
    s = np.linalg.svd(matrix, compute_uv=False)
    return int(np.sum(s > 1e-10))


@_FAST_SETTINGS
@given(psi=product_states(n_qubits=3))
def test_product_states_are_separable(psi: np.ndarray) -> None:
    """Schmidt rank across **every** bipartition must equal 1."""
    for cut in (1, 2):
        assert _bipartite_schmidt_rank(psi, cut, 3 - cut) == 1


@_FAST_SETTINGS
@given(psi=product_states(n_qubits=4))
def test_product_states_are_separable_4q(psi: np.ndarray) -> None:
    for cut in (1, 2, 3):
        assert _bipartite_schmidt_rank(psi, cut, 4 - cut) == 1


# --------------------------------------------------------------------------- #
# random_density_matrices                                                     #
# --------------------------------------------------------------------------- #


@_FAST_SETTINGS
@given(rho=random_density_matrices(n_qubits=2))
def test_density_matrix_is_hermitian(rho: np.ndarray) -> None:
    assert rho.shape == (4, 4)
    assert np.allclose(rho, rho.conj().T, atol=1e-12)


@_FAST_SETTINGS
@given(rho=random_density_matrices(n_qubits=2))
def test_density_matrix_has_unit_trace(rho: np.ndarray) -> None:
    assert np.isclose(np.trace(rho), 1.0, atol=1e-12)


@_FAST_SETTINGS
@given(rho=random_density_matrices(n_qubits=2))
def test_density_matrix_is_positive_semidefinite(rho: np.ndarray) -> None:
    """All eigenvalues should be >= 0 (within numerical tolerance)."""
    eigvals = np.linalg.eigvalsh(rho)
    assert np.all(eigvals > -1e-10)


@_FAST_SETTINGS
@given(rho=random_density_matrices(n_qubits=2, rank=1))
def test_rank_one_density_matrix_is_pure(rho: np.ndarray) -> None:
    """Pure state: :math:`\\mathrm{tr}(\\rho^2) = 1` to within round-off."""
    assert np.isclose(np.trace(rho @ rho).real, 1.0, atol=1e-10)


@settings(max_examples=1, deadline=None)
@given(data=st.data())
def test_rank_above_dimension_raises(data) -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        data.draw(random_density_matrices(n_qubits=2, rank=10))


# --------------------------------------------------------------------------- #
# Reproducibility (same Hypothesis seed -> same draws)                        #
# --------------------------------------------------------------------------- #


def test_random_states_reproducible_under_hypothesis_seed() -> None:
    """Two ``@given`` runs with the same ``@seed`` see the same examples.

    The strategy uses ``draw`` for all entropy, so Hypothesis fully
    determines the sequence of generated arrays. We capture the first
    drawn vector from each run and compare bit-for-bit.
    """
    runs: list[np.ndarray] = []

    @seed(12345)
    @settings(max_examples=1, deadline=None)
    @given(psi=random_states(n_qubits=2))
    def capture(psi: np.ndarray) -> None:
        runs.append(psi.copy())

    capture()
    capture()
    assert len(runs) == 2
    np.testing.assert_array_equal(runs[0], runs[1])


def test_product_states_reproducible_under_hypothesis_seed() -> None:
    runs: list[np.ndarray] = []

    @seed(98765)
    @settings(max_examples=1, deadline=None)
    @given(psi=product_states(n_qubits=2))
    def capture(psi: np.ndarray) -> None:
        runs.append(psi.copy())

    capture()
    capture()
    np.testing.assert_array_equal(runs[0], runs[1])
