"""Tests for :mod:`qtest.metrics.entanglement`."""

from __future__ import annotations

import numpy as np
import pytest

from qtest.metrics import entanglement_entropy, partial_trace, purity, von_neumann_entropy

_INV_SQRT2 = 1.0 / np.sqrt(2.0)
BELL = np.array([_INV_SQRT2, 0.0, 0.0, _INV_SQRT2])
PRODUCT_00 = np.array([1.0, 0.0, 0.0, 0.0])
GHZ3 = np.zeros(8)
GHZ3[0] = GHZ3[7] = _INV_SQRT2


def test_partial_trace_bell_is_maximally_mixed() -> None:
    rho_a = partial_trace(BELL, keep=[0], num_qubits=2)
    np.testing.assert_allclose(rho_a, 0.5 * np.eye(2), atol=1e-12)


def test_partial_trace_infers_num_qubits() -> None:
    rho_a = partial_trace(BELL, keep=[0])
    assert rho_a.shape == (2, 2)


def test_partial_trace_keep_all_returns_full_state() -> None:
    rho = partial_trace(PRODUCT_00, keep=[0, 1], num_qubits=2)
    expected = np.outer(PRODUCT_00, PRODUCT_00.conj())
    np.testing.assert_allclose(rho, expected, atol=1e-12)


@pytest.mark.parametrize("bad_keep", [[2], [-1], [0, 5]])
def test_partial_trace_out_of_range_raises(bad_keep: list[int]) -> None:
    with pytest.raises(ValueError):
        partial_trace(BELL, keep=bad_keep, num_qubits=2)


def test_partial_trace_dim_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        partial_trace(BELL, keep=[0], num_qubits=3)


def test_purity_pure_vs_mixed() -> None:
    assert purity(PRODUCT_00) == pytest.approx(1.0)
    assert purity(0.5 * np.eye(2)) == pytest.approx(0.5)


def test_von_neumann_entropy_bits() -> None:
    assert von_neumann_entropy(0.5 * np.eye(2)) == pytest.approx(1.0)
    assert von_neumann_entropy(PRODUCT_00) == pytest.approx(0.0, abs=1e-12)


def test_von_neumann_entropy_base_e() -> None:
    # ln(2) nats == 1 bit for the maximally mixed qubit.
    assert von_neumann_entropy(0.5 * np.eye(2), base=np.e) == pytest.approx(np.log(2))


def test_entanglement_entropy_values() -> None:
    assert entanglement_entropy(BELL, [0], 2) == pytest.approx(1.0)
    assert entanglement_entropy(PRODUCT_00, [0], 2) == pytest.approx(0.0, abs=1e-12)
    assert entanglement_entropy(GHZ3, [0], 3) == pytest.approx(1.0)
    assert entanglement_entropy(GHZ3, [0, 1], 3) == pytest.approx(1.0)


def test_invalid_shape_raises() -> None:
    with pytest.raises(ValueError):
        purity(np.zeros((2, 3)))
