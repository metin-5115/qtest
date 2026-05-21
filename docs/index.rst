qtest — statistical, pytest-native testing for quantum circuits
================================================================

.. image:: https://img.shields.io/pypi/v/qtest.svg
   :target: https://pypi.org/project/qtest/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/qtest.svg
   :target: https://pypi.org/project/qtest/
   :alt: Python versions

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

.. image:: https://readthedocs.org/projects/qtest/badge/?version=latest
   :target: https://qtest.readthedocs.io
   :alt: Documentation Status

**qtest** is an open-source Python library that brings the discipline of
modern software testing to quantum programs. It plugs straight into
:mod:`pytest`, gives you statistical assertions designed for noisy and
probabilistic outputs, and integrates with `Hypothesis
<https://hypothesis.readthedocs.io/>`_ so you can do property-based testing
on quantum circuits — without writing a single line of plumbing.

Why qtest?
----------

Testing quantum software is hard for reasons classical testing tools were
never designed to handle:

- **Outputs are distributions, not values.** A measurement gives you counts
  that are *statistically close* to the truth, never exact. ``assertEqual``
  is the wrong tool.
- **State and unitary comparisons are global-phase-insensitive.** Two
  correct implementations of the same gate can differ by a phase that
  doesn't matter physically — but does matter to ``numpy.allclose``.
- **Shot noise and seed drift make tests flaky.** Without a principled
  tolerance and a controlled seed, your CI lights up red at random.
- **Property-based testing fits quantum perfectly** — laws like unitarity,
  reversibility, and Bell-state symmetry are universal — but the standard
  Hypothesis strategies don't know about ``QuantumCircuit``.

``qtest`` solves these directly: tolerance-aware statistical assertions,
a Hypothesis strategy set tuned for circuits, gates, and states, and a
``pytest`` plugin that exposes ``--qtest-shots``, ``--qtest-tolerance``,
and ``--qtest-seed`` as first-class CLI flags.

Quickstart
----------

Install:

.. code-block:: bash

   pip install qtest

Write your first quantum test — a Bell state should produce a 50/50
distribution over ``00`` and ``11``:

.. code-block:: python

   from qiskit import QuantumCircuit
   from qtest import assert_distribution_close

   def test_bell_state_is_balanced(qtest_backend):
       qc = QuantumCircuit(2, 2)
       qc.h(0)
       qc.cx(0, 1)
       qc.measure([0, 1], [0, 1])

       counts = qtest_backend.run(qc, shots=4096)

       assert_distribution_close(
           counts,
           expected={"00": 0.5, "11": 0.5},
           tolerance=0.03,
       )

Run it:

.. code-block:: bash

   pytest test_bell.py -v

That's it — no mocking, no manual shot-count math, no hand-rolled
tolerance checks.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User guide

   guides/writing_assertions
   guides/property_testing
   guides/pytest_integration

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api/assertions
   api/strategies
   api/fixtures
   api/backends
   api/metrics

.. toctree::
   :maxdepth: 1
   :caption: Project

   contributing
   changelog
   GitHub repository <https://github.com/metin-5115/qtest>

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
