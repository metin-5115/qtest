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

See https://qtest.readthedocs.io for full documentation.
"""

from qtest.assertions import (
    assert_circuit_equivalent,
    assert_distribution_close,
    assert_state_close,
    assert_unitary,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "assert_circuit_equivalent",
    "assert_distribution_close",
    "assert_state_close",
    "assert_unitary",
]
