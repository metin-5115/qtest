Visualisation
=============

The :mod:`qtest.viz` module provides convenience matplotlib plots for
measurement distributions — useful in notebooks and when triaging a failed
:func:`~qtest.assert_distribution_close`. It requires the optional ``viz``
extra::

   pip install 'qtest-quantum[viz]'

matplotlib is imported lazily, so importing :mod:`qtest` does not require it.

.. currentmodule:: qtest.viz

Public API
----------

.. autosummary::
   :nosignatures:

   plot_distribution
   plot_distribution_comparison

Example
-------

.. code-block:: python

   from qtest.viz import plot_distribution_comparison

   measured = {"00": 0.48, "01": 0.01, "10": 0.02, "11": 0.49}
   expected = {"00": 0.5, "11": 0.5}
   ax = plot_distribution_comparison(measured, expected, title="Bell state")
   ax.figure.savefig("bell.png")

Reference
---------

.. automodule:: qtest.viz
   :members:
   :show-inheritance:
   :member-order: bysource
