"""Tests for ``qtest.metrics.distances``."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qtest.metrics import (
    fidelity,
    hellinger_distance,
    hilbert_schmidt_distance,
    total_variation_distance,
    trace_distance,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def uniform_2() -> dict[str, float]:
    """Uniform distribution over two outcomes."""
    return {"0": 0.5, "1": 0.5}


@pytest.fixture
def biased_2() -> dict[str, float]:
    """A 70/30 biased distribution over two outcomes."""
    return {"0": 0.7, "1": 0.3}


@pytest.fixture
def zero_state() -> np.ndarray:
    return np.array([1.0, 0.0], dtype=complex)


@pytest.fixture
def one_state() -> np.ndarray:
    return np.array([0.0, 1.0], dtype=complex)


@pytest.fixture
def plus_state() -> np.ndarray:
    return np.array([1.0, 1.0], dtype=complex) / math.sqrt(2.0)


@pytest.fixture
def maximally_mixed_2() -> np.ndarray:
    return 0.5 * np.eye(2, dtype=complex)


# ---------------------------------------------------------------------------
# total_variation_distance
# ---------------------------------------------------------------------------


class TestTotalVariationDistance:
    def test_identical_distributions_zero(self, uniform_2: dict[str, float]) -> None:
        assert total_variation_distance(uniform_2, uniform_2) == pytest.approx(0.0)

    def test_disjoint_support_one(self) -> None:
        assert total_variation_distance({"0": 1.0}, {"1": 1.0}) == pytest.approx(1.0)

    def test_missing_keys_treated_as_zero(self) -> None:
        # P = {0: 0.5, 1: 0.5}, Q = {0: 1.0}  ->  TVD = 0.5
        assert total_variation_distance({"0": 0.5, "1": 0.5}, {"0": 1.0}) == pytest.approx(0.5)

    @pytest.mark.parametrize(
        ("p", "q", "expected"),
        [
            ({"0": 0.5, "1": 0.5}, {"0": 0.5, "1": 0.5}, 0.0),
            ({"0": 0.7, "1": 0.3}, {"0": 0.3, "1": 0.7}, 0.4),
            (
                {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25},
                {"a": 1.0},
                0.75,
            ),
        ],
    )
    def test_parametrized_known_values(
        self, p: dict[str, float], q: dict[str, float], expected: float
    ) -> None:
        assert total_variation_distance(p, q) == pytest.approx(expected)

    def test_symmetry(self, uniform_2: dict[str, float], biased_2: dict[str, float]) -> None:
        assert total_variation_distance(uniform_2, biased_2) == pytest.approx(
            total_variation_distance(biased_2, uniform_2)
        )

    def test_range_in_unit_interval(
        self, uniform_2: dict[str, float], biased_2: dict[str, float]
    ) -> None:
        d = total_variation_distance(uniform_2, biased_2)
        assert 0.0 <= d <= 1.0

    def test_triangle_inequality(self) -> None:
        p = {"0": 0.8, "1": 0.2}
        q = {"0": 0.5, "1": 0.5}
        r = {"0": 0.1, "1": 0.9}
        assert total_variation_distance(p, r) <= (
            total_variation_distance(p, q) + total_variation_distance(q, r) + 1e-12
        )

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            total_variation_distance({}, {"0": 1.0})

    def test_negative_probability_raises(self) -> None:
        with pytest.raises(ValueError):
            total_variation_distance({"0": -0.1, "1": 1.1}, {"0": 1.0})

    def test_unnormalized_raises(self) -> None:
        with pytest.raises(ValueError):
            total_variation_distance({"0": 0.5, "1": 0.4}, {"0": 1.0})

    def test_nan_value_raises(self) -> None:
        with pytest.raises(ValueError):
            total_variation_distance({"0": float("nan")}, {"0": 1.0})


# ---------------------------------------------------------------------------
# hellinger_distance
# ---------------------------------------------------------------------------


class TestHellingerDistance:
    def test_identical_distributions_zero(self, uniform_2: dict[str, float]) -> None:
        assert hellinger_distance(uniform_2, uniform_2) == pytest.approx(0.0, abs=1e-12)

    def test_disjoint_support_one(self) -> None:
        assert hellinger_distance({"0": 1.0}, {"1": 1.0}) == pytest.approx(1.0)

    def test_symmetry(self, uniform_2: dict[str, float], biased_2: dict[str, float]) -> None:
        assert hellinger_distance(uniform_2, biased_2) == pytest.approx(
            hellinger_distance(biased_2, uniform_2)
        )

    def test_range_in_unit_interval(
        self, uniform_2: dict[str, float], biased_2: dict[str, float]
    ) -> None:
        d = hellinger_distance(uniform_2, biased_2)
        assert 0.0 <= d <= 1.0

    @pytest.mark.parametrize(
        ("p", "q", "expected"),
        [
            ({"0": 0.5, "1": 0.5}, {"0": 0.5, "1": 0.5}, 0.0),
            ({"0": 1.0}, {"1": 1.0}, 1.0),
            # Closed-form: delta vs uniform_2
            (
                {"0": 1.0},
                {"0": 0.5, "1": 0.5},
                math.sqrt((1.0 - math.sqrt(0.5)) ** 2 + 0.5) / math.sqrt(2.0),
            ),
        ],
    )
    def test_parametrized_known_values(
        self, p: dict[str, float], q: dict[str, float], expected: float
    ) -> None:
        assert hellinger_distance(p, q) == pytest.approx(expected, abs=1e-12)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            hellinger_distance({}, {"0": 1.0})


# ---------------------------------------------------------------------------
# fidelity
# ---------------------------------------------------------------------------


class TestFidelity:
    def test_same_state_vector_is_one(self, zero_state: np.ndarray) -> None:
        assert fidelity(zero_state, zero_state) == pytest.approx(1.0)

    def test_orthogonal_state_vectors_zero(
        self, zero_state: np.ndarray, one_state: np.ndarray
    ) -> None:
        assert fidelity(zero_state, one_state) == pytest.approx(0.0)

    def test_plus_vs_zero_is_half(self, zero_state: np.ndarray, plus_state: np.ndarray) -> None:
        # |<+|0>|^2 = 1/2
        assert fidelity(plus_state, zero_state) == pytest.approx(0.5)

    def test_global_phase_invariance(self, plus_state: np.ndarray) -> None:
        phase = np.exp(1j * 1.2345)
        assert fidelity(plus_state, phase * plus_state) == pytest.approx(1.0)

    def test_density_matrix_self_fidelity_is_one(self, maximally_mixed_2: np.ndarray) -> None:
        # F(rho, rho) = (tr sqrt(rho^2))^2 = (tr rho)^2 = 1.
        assert fidelity(maximally_mixed_2, maximally_mixed_2) == pytest.approx(1.0)

    def test_pure_vs_maximally_mixed_is_half(
        self, zero_state: np.ndarray, maximally_mixed_2: np.ndarray
    ) -> None:
        # F(|0>, I/2) = <0|I/2|0> = 1/2.
        assert fidelity(zero_state, maximally_mixed_2) == pytest.approx(0.5)

    def test_symmetry(self, plus_state: np.ndarray, zero_state: np.ndarray) -> None:
        assert fidelity(plus_state, zero_state) == pytest.approx(fidelity(zero_state, plus_state))

    def test_random_pure_pairs_in_unit_interval(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(20):
            psi = rng.standard_normal(4) + 1j * rng.standard_normal(4)
            psi /= np.linalg.norm(psi)
            phi = rng.standard_normal(4) + 1j * rng.standard_normal(4)
            phi /= np.linalg.norm(phi)
            f = fidelity(psi, phi)
            assert 0.0 <= f <= 1.0

    def test_incompatible_shapes_raise(self) -> None:
        with pytest.raises(ValueError):
            fidelity(np.array([1.0, 0.0]), np.array([1.0, 0.0, 0.0]))

    def test_mixed_input_types(self, zero_state: np.ndarray, maximally_mixed_2: np.ndarray) -> None:
        # Caller can mix a state vector and a density matrix in either order.
        a = fidelity(zero_state, maximally_mixed_2)
        b = fidelity(maximally_mixed_2, zero_state)
        assert a == pytest.approx(b)
        assert a == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# trace_distance
# ---------------------------------------------------------------------------


class TestTraceDistance:
    def test_same_density_matrix_zero(self, maximally_mixed_2: np.ndarray) -> None:
        assert trace_distance(maximally_mixed_2, maximally_mixed_2) == pytest.approx(0.0)

    def test_orthogonal_pure_states_one(
        self, zero_state: np.ndarray, one_state: np.ndarray
    ) -> None:
        assert trace_distance(zero_state, one_state) == pytest.approx(1.0)

    def test_symmetry(self, zero_state: np.ndarray, plus_state: np.ndarray) -> None:
        assert trace_distance(zero_state, plus_state) == pytest.approx(
            trace_distance(plus_state, zero_state)
        )

    def test_pure_state_relation_to_fidelity(
        self, zero_state: np.ndarray, plus_state: np.ndarray
    ) -> None:
        # For pure states: D = sqrt(1 - F).
        f = fidelity(zero_state, plus_state)
        d = trace_distance(zero_state, plus_state)
        assert d == pytest.approx(math.sqrt(1.0 - f))

    def test_pure_vs_maximally_mixed_is_half(
        self, zero_state: np.ndarray, maximally_mixed_2: np.ndarray
    ) -> None:
        # D(|0><0|, I/2) = 1/2.
        assert trace_distance(zero_state, maximally_mixed_2) == pytest.approx(0.5)

    def test_range_in_unit_interval(
        self, zero_state: np.ndarray, maximally_mixed_2: np.ndarray
    ) -> None:
        d = trace_distance(zero_state, maximally_mixed_2)
        assert 0.0 <= d <= 1.0

    def test_incompatible_shapes_raise(self) -> None:
        with pytest.raises(ValueError):
            trace_distance(np.eye(2), np.eye(3))


# ---------------------------------------------------------------------------
# hilbert_schmidt_distance
# ---------------------------------------------------------------------------


class TestHilbertSchmidtDistance:
    def test_identical_matrices_zero(self) -> None:
        m = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
        assert hilbert_schmidt_distance(m, m) == pytest.approx(0.0)

    def test_known_value_pauli_x_vs_identity(self) -> None:
        pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
        identity = np.eye(2, dtype=complex)
        # X - I = [[-1, 1], [1, -1]], Frobenius norm = 2.
        assert hilbert_schmidt_distance(pauli_x, identity) == pytest.approx(2.0)

    def test_symmetry(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
        b = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
        assert hilbert_schmidt_distance(a, b) == pytest.approx(hilbert_schmidt_distance(b, a))

    def test_triangle_inequality(self) -> None:
        rng = np.random.default_rng(1)
        a = rng.standard_normal((4, 4))
        b = rng.standard_normal((4, 4))
        c = rng.standard_normal((4, 4))
        assert hilbert_schmidt_distance(a, c) <= (
            hilbert_schmidt_distance(a, b) + hilbert_schmidt_distance(b, c) + 1e-12
        )

    def test_non_2d_raises(self) -> None:
        with pytest.raises(ValueError):
            hilbert_schmidt_distance(np.array([1.0, 0.0]), np.array([0.0, 1.0]))

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            hilbert_schmidt_distance(np.eye(2), np.eye(3))
