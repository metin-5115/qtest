OpenQASM
========

Load circuits from OpenQASM 2.0 / 3.0 so they can be fed straight into qtest's
assertions — handy when a circuit arrives as a compiler artifact, an export
from another tool, or a checked-in fixture.

.. code-block:: python

   from qtest import load_qasm, assert_state_close

   program = '''
   OPENQASM 2.0;
   include "qelib1.inc";
   qreg q[2];
   h q[0];
   cx q[0], q[1];
   '''
   qc = load_qasm(program)          # version auto-detected from the header
   assert_state_close(qc, "bell")

OpenQASM 2.0 uses Qiskit's built-in parser. OpenQASM 3.0 additionally requires
the ``qiskit-qasm3-import`` package — install the optional extra::

   pip install 'qtest-quantum[qasm3]'

.. currentmodule:: qtest

Public API
----------

.. autosummary::
   :nosignatures:

   load_qasm
   load_qasm_file

Reference
---------

.. automodule:: qtest.qasm
   :members:
   :show-inheritance:
   :member-order: bysource
