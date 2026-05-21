Metrics
=======

The :mod:`qtest.metrics` subpackage groups two complementary toolkits
used internally by the assertions:

* :mod:`qtest.metrics.distances` — deterministic distance / divergence
  functions over probability distributions, quantum states, and
  operators (total variation, Hellinger, fidelity, trace distance,
  Hilbert–Schmidt distance).

* :mod:`qtest.metrics.statistical_tests` — frequentist hypothesis tests
  and shot-noise tolerance helpers (Pearson :math:`\chi^2`,
  Kolmogorov–Smirnov, :func:`auto_tolerance`).

All functions are pure: they validate input, never mutate their
arguments, and raise :class:`ValueError` on malformed input.

.. currentmodule:: qtest.metrics

Public API
----------

.. autosummary::
   :nosignatures:

   total_variation_distance
   hellinger_distance
   fidelity
   trace_distance
   hilbert_schmidt_distance
   chi_square_test
   kolmogorov_smirnov_test
   auto_tolerance

Reference
---------

.. automodule:: qtest.metrics
   :members:
   :show-inheritance:
   :member-order: bysource

Distances
~~~~~~~~~

.. automodule:: qtest.metrics.distances
   :members:
   :show-inheritance:
   :member-order: bysource

Statistical tests
~~~~~~~~~~~~~~~~~

.. automodule:: qtest.metrics.statistical_tests
   :members:
   :show-inheritance:
   :member-order: bysource
