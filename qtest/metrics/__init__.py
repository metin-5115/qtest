"""Metric and statistical-test functions used by qtest's assertions.

This package gathers two complementary toolkits:

* :mod:`qtest.metrics.distances` — deterministic distance / divergence
  functions over probability distributions, quantum states, and operators
  (TVD, Hellinger, fidelity, trace distance, Hilbert–Schmidt distance).

* :mod:`qtest.metrics.statistical_tests` — frequentist hypothesis tests and
  shot-noise tolerance helpers consumed by ``assert_distribution_close``
  and friends (Pearson :math:`\\chi^2`, Kolmogorov–Smirnov, ``auto_tolerance``).

All functions are pure: they perform input validation, never mutate their
arguments, and raise :class:`ValueError` on malformed input.
"""

from qtest.metrics.distances import (
    fidelity,
    hellinger_distance,
    hilbert_schmidt_distance,
    total_variation_distance,
    trace_distance,
)
from qtest.metrics.statistical_tests import (
    auto_tolerance,
    chi_square_test,
    kolmogorov_smirnov_test,
)

__all__ = [
    "auto_tolerance",
    "chi_square_test",
    "fidelity",
    "hellinger_distance",
    "hilbert_schmidt_distance",
    "kolmogorov_smirnov_test",
    "total_variation_distance",
    "trace_distance",
]
