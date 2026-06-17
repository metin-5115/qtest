Snapshot & golden-file testing
==============================

Sometimes you don't have a hand-derived expected distribution — you just want
to **lock in** a circuit's current behaviour and be told when a refactor
changes it. That's what snapshot testing is for.

The ``qtest_snapshot`` fixture
------------------------------

Request the ``qtest_snapshot`` fixture and call
:meth:`~qtest.snapshot.Snapshot.assert_distribution_close`:

.. code-block:: python

   from qiskit import QuantumCircuit

   def test_grover_distribution(qtest_snapshot):
       qc = build_grover()       # your circuit under test
       qc.measure_all()
       qtest_snapshot.assert_distribution_close(qc, shots=8192, tolerance=0.1)

What happens:

* **First run** (no golden file yet): qtest samples the circuit and writes the
  distribution to ``__qtest_snapshots__/<test-name>.json`` next to your test,
  then passes.
* **Later runs**: qtest re-samples and compares against the stored baseline
  using the configured metric and ``tolerance``; the test fails if the output
  has drifted.

Because both the baseline and each later run are *sampled*, pick a ``tolerance``
that absorbs two-sample shot noise (or raise ``shots``).

Multiple snapshots per test
---------------------------

Pass ``name=`` to keep several independent baselines in one test:

.. code-block:: python

   def test_two_stages(qtest_snapshot):
       qtest_snapshot.assert_distribution_close(stage_a(), name="stage_a", shots=4096)
       qtest_snapshot.assert_distribution_close(stage_b(), name="stage_b", shots=4096)

Updating snapshots
------------------

When a behaviour change is intentional, refresh the golden files:

.. code-block:: bash

   pytest --qtest-snapshot-update

Commit the regenerated ``__qtest_snapshots__/`` files alongside your code so CI
compares against the reviewed baseline.

Visualising distributions
-------------------------

When a snapshot (or any :func:`~qtest.assert_distribution_close`) check fails,
:mod:`qtest.viz` helps you see the drift (requires ``pip install 'qtest-quantum[viz]'``):

.. code-block:: python

   from qtest.viz import plot_distribution_comparison

   ax = plot_distribution_comparison(measured, expected, title="drifted?")
   ax.figure.savefig("diff.png")

The ``--qtest-summary`` report
------------------------------

Run with ``--qtest-summary`` for an end-of-session digest. Every
distribution/marginal assertion records the distance it measured, so the report
shows the sample count and the mean / min / max measured distance across the
run — a quick health check on how close your circuits are tracking their
targets.
