Snapshot testing
================

Snapshot (golden-file) testing locks in a circuit's measured distribution: the
first run records it to a JSON file, later runs re-sample and compare against
that baseline with the usual statistical metric. Use it through the
``qtest_snapshot`` pytest fixture::

    def test_my_circuit(qtest_snapshot):
        qc = build_circuit()
        qc.measure_all()
        qtest_snapshot.assert_distribution_close(qc, shots=4096, tolerance=0.1)

Golden files are written to a ``__qtest_snapshots__/`` directory beside the
test file and should be committed. Refresh them deliberately::

    pytest --qtest-snapshot-update

.. currentmodule:: qtest.snapshot

Reference
---------

.. automodule:: qtest.snapshot
   :members:
   :show-inheritance:
   :member-order: bysource
