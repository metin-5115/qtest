"""Tests for :func:`qtest.assertions.assert_state_close`."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from qtest.assertions import assert_state_close
from qtest.assertions._state_library import (
    _ghz_state,
    _parse_qubit_count,
    _w_state,
    get_state,
    list_known_states,
)
from qtest.assertions.state import (
    _coerce_expected_state,
    _format_amplitude,
    _format_state,
)
from qtest.backends.base import Backend

# --------------------------------------------------------------------------- #
# Mock backends                                                               #
# --------------------------------------------------------------------------- #


class _StateBackend(Backend):
    """Mock backend that returns a fixed state vector."""

    def __init__(self, state: np.ndarray, *, name: str = "mock_state") -> None:
        self._state = np.asarray(state, dtype=complex)
        self._name = name

    def run_circuit(
        self, circuit: Any, shots: int | None = None, seed: int | None = None
    ) -> dict[str, int]:
        raise NotImplementedError

    def get_statevector(self, circuit: Any) -> np.ndarray:
        return self._state.copy()

    def get_unitary(self, circuit: Any) -> np.ndarray:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self._name

    @property
    def supports_statevector(self) -> bool:
        return True


class _NoStateBackend(_StateBackend):
    """Backend that explicitly refuses state-vector extraction."""

    @property
    def supports_statevector(self) -> bool:
        return False


class _FakeCircuit:
    def __init__(self, name: str = "fake", num_qubits: int | None = None) -> None:
        self.name = name
        if num_qubits is not None:
            self.num_qubits = num_qubits


SQRT2 = float(np.sqrt(2.0))


# --------------------------------------------------------------------------- #
# _state_library — get_state / parametric families                            #
# --------------------------------------------------------------------------- #


def test_get_state_plus() -> None:
    s = get_state("plus")
    expected = np.array([1.0, 1.0], dtype=complex) / SQRT2
    assert np.allclose(s, expected)


def test_get_state_minus() -> None:
    assert np.allclose(get_state("minus"), np.array([1.0, -1.0]) / SQRT2)


def test_get_state_i_plus() -> None:
    assert np.allclose(get_state("i_plus"), np.array([1.0, 1.0j]) / SQRT2)


def test_get_state_i_minus() -> None:
    assert np.allclose(get_state("i_minus"), np.array([1.0, -1.0j]) / SQRT2)


def test_get_state_bell() -> None:
    expected = np.array([1.0, 0.0, 0.0, 1.0], dtype=complex) / SQRT2
    assert np.allclose(get_state("bell"), expected)


def test_get_state_ghz_3() -> None:
    s = get_state("ghz_3")
    assert s.size == 8
    expected = np.zeros(8, dtype=complex)
    expected[0] = expected[7] = 1.0 / SQRT2
    assert np.allclose(s, expected)


def test_get_state_w_3() -> None:
    s = get_state("w_3")
    expected = np.zeros(8, dtype=complex)
    for k in (1, 2, 4):
        expected[k] = 1.0 / float(np.sqrt(3))
    assert np.allclose(s, expected)


def test_get_state_w_4() -> None:
    s = get_state("w_4")
    assert s.size == 16
    nonzero_indices = np.where(np.abs(s) > 1e-9)[0]
    assert set(nonzero_indices.tolist()) == {1, 2, 4, 8}
    assert np.isclose(np.linalg.norm(s), 1.0)


def test_get_state_is_case_insensitive_and_strips_whitespace() -> None:
    assert np.allclose(get_state("  BELL  "), get_state("bell"))
    assert np.allclose(get_state("GHZ_3"), get_state("ghz_3"))


def test_get_state_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match="Unknown state name"):
        get_state("bogus")


def test_get_state_non_string_raises() -> None:
    with pytest.raises(ValueError, match="must be a string"):
        get_state(42)  # type: ignore[arg-type]


def test_get_state_ghz_zero_qubits_raises() -> None:
    with pytest.raises(ValueError, match="positive qubit count"):
        get_state("ghz_0")


def test_get_state_ghz_bad_suffix_raises() -> None:
    with pytest.raises(ValueError, match="Cannot parse qubit count"):
        get_state("ghz_abc")


def test_get_state_returns_unit_norm() -> None:
    for name in ("plus", "minus", "i_plus", "i_minus", "bell", "ghz_4", "w_5"):
        assert np.isclose(np.linalg.norm(get_state(name)), 1.0), name


def test_get_state_returns_independent_copy() -> None:
    """Mutating the returned array must not affect future lookups."""
    a = get_state("bell")
    a[0] = 0.0
    b = get_state("bell")
    assert not np.allclose(a, b)


def test_list_known_states_returns_sorted_list() -> None:
    names = list_known_states()
    assert names == sorted(names)
    assert "bell" in names
    assert "ghz_<n>" in names
    assert "w_<n>" in names


def test_parse_qubit_count_helper() -> None:
    assert _parse_qubit_count("ghz_5", "ghz_") == 5
    with pytest.raises(ValueError):
        _parse_qubit_count("ghz_-1", "ghz_")


def test_ghz_helper_n_equals_1_is_plus() -> None:
    """For n=1 the GHZ definition reduces to |+>."""
    assert np.allclose(_ghz_state(1), get_state("plus"))


def test_w_helper_n_equals_1_is_one_state() -> None:
    """For n=1 the W state is just |1>."""
    assert np.allclose(_w_state(1), np.array([0.0, 1.0], dtype=complex))


# --------------------------------------------------------------------------- #
# assert_state_close — happy paths                                            #
# --------------------------------------------------------------------------- #


def test_perfect_match_against_named_state() -> None:
    bell = np.array([1.0, 0.0, 0.0, 1.0], dtype=complex) / SQRT2
    assert_state_close(_FakeCircuit(), "bell", backend=_StateBackend(bell))


def test_perfect_match_against_ndarray() -> None:
    plus = np.array([1.0, 1.0], dtype=complex) / SQRT2
    assert_state_close(_FakeCircuit(), plus, backend=_StateBackend(plus))


def test_perfect_match_against_list() -> None:
    plus_list = [1.0 / SQRT2, 1.0 / SQRT2]
    plus = np.asarray(plus_list, dtype=complex)
    assert_state_close(_FakeCircuit(), plus_list, backend=_StateBackend(plus))


def test_ghz_3_match() -> None:
    state = get_state("ghz_3")
    assert_state_close(_FakeCircuit(num_qubits=3), "ghz_3", backend=_StateBackend(state))


# --------------------------------------------------------------------------- #
# Global phase handling                                                       #
# --------------------------------------------------------------------------- #


def test_global_phase_ignored_by_default() -> None:
    """Multiplying state by exp(i*theta) should still pass with global_phase=True."""
    bell = get_state("bell")
    phased = bell * np.exp(1j * 1.2345)
    assert_state_close(_FakeCircuit(), "bell", backend=_StateBackend(phased))


def test_global_phase_false_rejects_phase_shifted_state() -> None:
    bell = get_state("bell")
    phased = bell * np.exp(1j * 1.2345)
    with pytest.raises(AssertionError):
        assert_state_close(
            _FakeCircuit(),
            "bell",
            global_phase=False,
            backend=_StateBackend(phased),
        )


def test_global_phase_false_accepts_identical_state() -> None:
    bell = get_state("bell")
    assert_state_close(
        _FakeCircuit(),
        "bell",
        global_phase=False,
        tolerance=1e-9,
        backend=_StateBackend(bell),
    )


# --------------------------------------------------------------------------- #
# Failure paths                                                               #
# --------------------------------------------------------------------------- #


def test_orthogonal_state_fails() -> None:
    """Bell state vs |00> -> fidelity 0.5, should fail at tight tolerance."""
    bell = get_state("bell")
    zero_zero = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
    with pytest.raises(AssertionError, match="State vector mismatch"):
        assert_state_close(_FakeCircuit(), bell, backend=_StateBackend(zero_zero))


def test_failure_message_contains_diagnostics() -> None:
    bell = get_state("bell")
    wrong = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
    with pytest.raises(AssertionError) as exc:
        assert_state_close(
            _FakeCircuit(name="prep_bell", num_qubits=2),
            bell,
            backend=_StateBackend(wrong, name="mock_X"),
            msg="Bell prep failed",
        )
    text = str(exc.value)
    assert text.startswith("Bell prep failed")
    assert "prep_bell" in text
    assert "Tolerance" in text
    assert "fidelity:" in text
    assert "Expected state:" in text
    assert "Measured state:" in text
    assert "|00>" in text
    assert "|11>" in text


def test_failure_message_global_phase_false_shows_l2_distance() -> None:
    bell = get_state("bell")
    other = np.array([0.5, 0.5, 0.5, 0.5], dtype=complex)
    with pytest.raises(AssertionError) as exc:
        assert_state_close(
            _FakeCircuit(),
            bell,
            global_phase=False,
            backend=_StateBackend(other),
        )
    assert "L2 distance:" in str(exc.value)


# --------------------------------------------------------------------------- #
# Tolerance edge cases                                                        #
# --------------------------------------------------------------------------- #


def test_tolerance_zero_strict_match_passes() -> None:
    bell = get_state("bell")
    assert_state_close(_FakeCircuit(), bell, tolerance=0.0, backend=_StateBackend(bell))


def test_tolerance_loose_accepts_far_state() -> None:
    """Very loose tolerance permits a substantially wrong state."""
    bell = get_state("bell")
    wrong = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)  # fidelity = 0.5
    # tolerance = 0.6 -> requires fidelity >= 0.4, which 0.5 satisfies.
    assert_state_close(_FakeCircuit(), bell, tolerance=0.6, backend=_StateBackend(wrong))


def test_negative_tolerance_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        assert_state_close(
            _FakeCircuit(), "bell", tolerance=-0.1, backend=_StateBackend(get_state("bell"))
        )


def test_non_numeric_tolerance_raises() -> None:
    with pytest.raises(ValueError, match="real number"):
        assert_state_close(
            _FakeCircuit(),
            "bell",
            tolerance="loose",  # type: ignore[arg-type]
            backend=_StateBackend(get_state("bell")),
        )


# --------------------------------------------------------------------------- #
# Input validation                                                            #
# --------------------------------------------------------------------------- #


def test_unknown_state_name_raises() -> None:
    with pytest.raises(ValueError, match="Unknown state name"):
        assert_state_close(_FakeCircuit(), "not_a_state", backend=_StateBackend(get_state("plus")))


def test_non_unit_norm_array_raises() -> None:
    not_normed = np.array([1.0, 1.0], dtype=complex)  # norm sqrt(2), not 1
    with pytest.raises(ValueError, match="unit Euclidean norm"):
        assert_state_close(_FakeCircuit(), not_normed, backend=_StateBackend(get_state("plus")))


def test_non_power_of_two_length_raises() -> None:
    weird = np.array([1.0, 0.0, 0.0], dtype=complex)  # length 3 — not 2**n
    with pytest.raises(ValueError, match="power of 2"):
        assert_state_close(_FakeCircuit(), weird, backend=_StateBackend(get_state("plus")))


def test_unsupported_type_raises() -> None:
    with pytest.raises(ValueError, match="must be str"):
        assert_state_close(
            _FakeCircuit(),
            42,  # type: ignore[arg-type]
            backend=_StateBackend(get_state("plus")),
        )


def test_empty_list_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        assert_state_close(_FakeCircuit(), [], backend=_StateBackend(get_state("plus")))


def test_2d_array_raises() -> None:
    matrix = np.eye(2, dtype=complex) / np.sqrt(2)
    with pytest.raises(ValueError, match="1-D"):
        assert_state_close(_FakeCircuit(), matrix, backend=_StateBackend(get_state("plus")))


def test_shape_mismatch_between_circuit_and_expected_raises() -> None:
    """Backend returns 2-dim state, but expected is 4-dim."""
    backend = _StateBackend(np.array([1.0, 0.0], dtype=complex))
    with pytest.raises(ValueError, match="shape mismatch"):
        assert_state_close(_FakeCircuit(), "bell", backend=backend)


# --------------------------------------------------------------------------- #
# Backend support checks                                                      #
# --------------------------------------------------------------------------- #


def test_backend_without_statevector_support_raises() -> None:
    backend = _NoStateBackend(get_state("bell"), name="no_sv")
    with pytest.raises(ValueError, match="does not support state-vector"):
        assert_state_close(_FakeCircuit(), "bell", backend=backend)


# --------------------------------------------------------------------------- #
# Integration with the real Qiskit backend                                    #
# --------------------------------------------------------------------------- #


def test_hadamard_circuit_matches_plus_state() -> None:
    qiskit = pytest.importorskip("qiskit")
    from qtest.backends.qiskit_backend import QiskitBackend

    qc = qiskit.QuantumCircuit(1)
    qc.h(0)
    assert_state_close(qc, "plus", backend=QiskitBackend())


def test_bell_circuit_matches_bell_named_state() -> None:
    qiskit = pytest.importorskip("qiskit")
    from qtest.backends.qiskit_backend import QiskitBackend

    qc = qiskit.QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    assert_state_close(qc, "bell", backend=QiskitBackend())


def test_ghz_3_circuit_matches_named_state() -> None:
    qiskit = pytest.importorskip("qiskit")
    from qtest.backends.qiskit_backend import QiskitBackend

    qc = qiskit.QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    assert_state_close(qc, "ghz_3", backend=QiskitBackend())


def test_qiskit_default_backend_path() -> None:
    """When backend is None, the default Qiskit backend is selected via config."""
    qiskit = pytest.importorskip("qiskit")
    qc = qiskit.QuantumCircuit(1)
    qc.h(0)
    assert_state_close(qc, "plus")


# --------------------------------------------------------------------------- #
# Helper unit tests                                                           #
# --------------------------------------------------------------------------- #


def test_coerce_expected_state_accepts_ndarray() -> None:
    v = np.array([1.0, 0.0], dtype=complex)
    out = _coerce_expected_state(v)
    assert np.allclose(out, v)


def test_format_amplitude_real_only() -> None:
    s = _format_amplitude(0.5 + 0j)
    assert "+0.5000" in s
    assert "+0.0000j" in s


def test_format_amplitude_negative_imaginary() -> None:
    s = _format_amplitude(0.3 - 0.7j)
    assert "+0.3000" in s
    assert "-0.7000j" in s


def test_format_state_small() -> None:
    bell = get_state("bell")
    text = _format_state(bell)
    assert "|00>" in text
    assert "|01>" in text
    assert "|10>" in text
    assert "|11>" in text


def test_format_state_truncates_large_vectors() -> None:
    """For >max_entries amplitudes, only the largest are shown."""
    vec = np.zeros(64, dtype=complex)
    vec[0] = 1.0  # only one non-zero
    text = _format_state(vec, max_entries=4)
    assert "more amplitudes hidden" in text
