"""Backend-level tests for noise support and density-matrix extraction."""

from __future__ import annotations

import numpy as np
import pytest

from qtest.backends import Backend, QiskitBackend

pytest.importorskip("qiskit")

from qiskit import QuantumCircuit  # noqa: E402


def test_base_get_density_matrix_default_not_implemented() -> None:
    class _Dummy(Backend):
        def run_circuit(self, circuit, shots=None, seed=None, noise_model=None):  # type: ignore[no-untyped-def]
            return {}

        def get_statevector(self, circuit):  # type: ignore[no-untyped-def]
            return np.zeros(2, dtype=complex)

        def get_unitary(self, circuit):  # type: ignore[no-untyped-def]
            return np.eye(2, dtype=complex)

        @property
        def name(self) -> str:
            return "dummy"

        @property
        def supports_statevector(self) -> bool:
            return True

    with pytest.raises(NotImplementedError):
        _Dummy().get_density_matrix(object())


def test_qiskit_density_matrix_noiseless_is_pure_projector() -> None:
    qc = QuantumCircuit(1)
    qc.h(0)
    rho = QiskitBackend().get_density_matrix(qc)
    assert rho.shape == (2, 2)
    # |+><+| has all entries equal to 0.5.
    np.testing.assert_allclose(rho, np.full((2, 2), 0.5), atol=1e-8)
    # Pure state: trace(rho^2) == 1.
    assert np.isclose(np.trace(rho @ rho).real, 1.0, atol=1e-8)


def test_qiskit_density_matrix_under_noise_is_mixed() -> None:
    pytest.importorskip("qiskit_aer")
    from qtest.noise import depolarizing

    qc = QuantumCircuit(1)
    qc.h(0)
    rho = QiskitBackend().get_density_matrix(qc, noise_model=depolarizing(0.2))
    # Under depolarizing noise the state is mixed: trace(rho^2) < 1.
    assert np.trace(rho @ rho).real < 1.0


def test_run_circuit_accepts_noise_model() -> None:
    pytest.importorskip("qiskit_aer")
    from qtest.noise import depolarizing

    qc = QuantumCircuit(1)
    qc.x(0)
    qc.measure_all()
    counts = QiskitBackend().run_circuit(qc, shots=512, seed=1, noise_model=depolarizing(0.01))
    assert sum(counts.values()) == 512
