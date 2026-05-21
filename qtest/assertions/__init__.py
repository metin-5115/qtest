"""qtest's pytest-friendly assertion functions.

Public API:

* :func:`assert_distribution_close` — compare a circuit's measurement
  distribution to an expected one using a configurable statistical metric.
* :func:`assert_state_close` — compare a circuit's state vector to an
  expected state (named, list, or array) using fidelity or raw L2 distance.
* :func:`assert_unitary` — verify that an operation is unitary by checking
  :math:`U^{\\dagger}U \\approx I`.
* :func:`assert_circuit_equivalent` — verify two circuits implement
  equivalent unitaries via process fidelity, Hilbert-Schmidt distance, or
  random-state sampling.
"""

from qtest.assertions.distribution import assert_distribution_close
from qtest.assertions.equivalence import assert_circuit_equivalent
from qtest.assertions.state import assert_state_close
from qtest.assertions.unitary import assert_unitary

__all__ = [
    "assert_circuit_equivalent",
    "assert_distribution_close",
    "assert_state_close",
    "assert_unitary",
]
