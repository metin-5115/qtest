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
* :func:`assert_robust_to_noise` — sweep increasing noise strengths and
  verify the output distribution stays within a tolerance of the ideal.
* :func:`assert_entangled` / :func:`assert_separable` — entanglement-aware
  state assertions based on entanglement entropy.
* :func:`assert_measurement_probabilities` — compare a marginal measurement
  distribution over a subset of bits.
* :func:`assert_phase` — check the relative phase between two basis amplitudes.
* :func:`assert_commutes` — check operator (anti)commutation.
* :func:`assert_max_depth` / :func:`assert_max_gate_count` /
  :func:`assert_max_two_qubit_count` / :func:`assert_max_t_count` — guard a
  circuit's resource cost (depth, gate counts, T-count) to catch optimisation
  and transpilation regressions.
"""

from qtest.assertions.commutation import assert_commutes
from qtest.assertions.distribution import assert_distribution_close
from qtest.assertions.entanglement import assert_entangled, assert_separable
from qtest.assertions.equivalence import assert_circuit_equivalent
from qtest.assertions.marginals import assert_measurement_probabilities
from qtest.assertions.phase import assert_phase
from qtest.assertions.resources import (
    assert_max_depth,
    assert_max_gate_count,
    assert_max_t_count,
    assert_max_two_qubit_count,
)
from qtest.assertions.robustness import assert_robust_to_noise
from qtest.assertions.state import assert_state_close
from qtest.assertions.unitary import assert_unitary

__all__ = [
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
]
