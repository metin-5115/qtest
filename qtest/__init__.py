"""qtest — Statistical, pytest-native testing for quantum circuits.

qtest brings the discipline of modern software testing to quantum programs.
It plugs into pytest, provides statistical assertions designed for the noisy,
probabilistic outputs of quantum measurements, and integrates with Hypothesis
to enable property-based testing for quantum circuits.

Public API:
    - assert_distribution_close
    - assert_state_close
    - assert_unitary
    - assert_circuit_equivalent
    - assert_robust_to_noise
    - assert_entangled / assert_separable
    - assert_measurement_probabilities
    - assert_phase
    - assert_commutes
    - assert_max_depth / assert_max_gate_count
    - assert_max_two_qubit_count / assert_max_t_count
    - load_qasm / load_qasm_file

See https://qtest-quantum.readthedocs.io for full documentation.
"""

from qtest.assertions import (
    assert_circuit_equivalent,
    assert_commutes,
    assert_distribution_close,
    assert_entangled,
    assert_max_depth,
    assert_max_gate_count,
    assert_max_t_count,
    assert_max_two_qubit_count,
    assert_measurement_probabilities,
    assert_phase,
    assert_robust_to_noise,
    assert_separable,
    assert_state_close,
    assert_unitary,
)
from qtest.qasm import load_qasm, load_qasm_file

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "assert_circuit_equivalent",
    "assert_commutes",
    "assert_distribution_close",
    "assert_entangled",
    "assert_max_depth",
    "assert_max_gate_count",
    "assert_max_t_count",
    "assert_max_two_qubit_count",
    "assert_measurement_probabilities",
    "assert_phase",
    "assert_robust_to_noise",
    "assert_separable",
    "assert_state_close",
    "assert_unitary",
    "load_qasm",
    "load_qasm_file",
]
