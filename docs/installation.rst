Installation
============

``qtest`` is published on PyPI and supports Python 3.9 – 3.12 on Linux,
macOS, and Windows.

From PyPI
---------

The base install gives you the assertions, the pytest plugin, and the
Qiskit backend:

.. code-block:: bash

   pip install qtest-quantum

To pull in *every* optional dependency in one go — Hypothesis strategies,
Qiskit Aer for fast simulation, matplotlib visualisation, and the docs
toolchain — use the ``all`` extra:

.. code-block:: bash

   pip install "qtest-quantum[all]"

Optional extras
---------------

``qtest`` is intentionally lean by default. Each optional feature is
gated behind a named extra so you only install what you actually use:

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - Extra
     - Adds
     - Install with
   * - ``hypothesis``
     - Property-based testing strategies (:mod:`qtest.strategies`).
     - ``pip install "qtest-quantum[hypothesis]"``
   * - ``aer``
     - The high-performance Qiskit Aer simulator backend.
     - ``pip install "qtest-quantum[aer]"``
   * - ``viz``
     - Matplotlib-based plotting helpers.
     - ``pip install "qtest-quantum[viz]"``
   * - ``dev``
     - Lint, type-check, test, build, and release tooling.
     - ``pip install "qtest-quantum[dev]"``
   * - ``docs``
     - Sphinx, Furo, MyST, and the rest of the docs toolchain.
     - ``pip install "qtest-quantum[docs]"``
   * - ``all``
     - All of the above.
     - ``pip install "qtest-quantum[all]"``

Development install
-------------------

Clone the repository and install in editable mode with the ``dev`` extra
so you get the formatter, linter, type checker, and test runner:

.. code-block:: bash

   git clone https://github.com/metin-5115/qtest.git
   cd qtest
   python -m venv .venv
   source .venv/bin/activate           # Linux / macOS
   .venv\Scripts\activate              # Windows PowerShell
   pip install -e ".[dev]"
   pre-commit install

Run the test suite to verify the install:

.. code-block:: bash

   pytest -q

Building the docs locally
-------------------------

If you plan to work on documentation, install the ``docs`` extra and use
``sphinx-autobuild`` for a live-reload preview:

.. code-block:: bash

   pip install -e ".[docs]"
   sphinx-autobuild docs docs/_build/html

The site will be served at http://127.0.0.1:8000 and rebuilt on every
file save.

Dependency matrix
-----------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Component
     - Minimum version
     - Notes
   * - Python
     - 3.9
     - 3.9 / 3.10 / 3.11 / 3.12 are CI-tested.
   * - Qiskit
     - 1.0
     - Required core dependency.
   * - NumPy
     - 1.24
     - Required core dependency.
   * - SciPy
     - 1.10
     - Required core dependency.
   * - pytest
     - 7.0
     - Required core dependency.
   * - Hypothesis
     - 6.0
     - Optional, enables :mod:`qtest.strategies`.
   * - Qiskit Aer
     - 0.13
     - Optional, enables the high-performance simulator backend.
   * - matplotlib
     - 3.7
     - Optional, enables visualisation helpers.

Verifying the install
---------------------

Run the bundled smoke test from a Python REPL:

.. code-block:: pycon

   >>> import qtest
   >>> qtest.__version__
   '0.1.0'
   >>> from qtest import assert_distribution_close
   >>> assert_distribution_close({"0": 512, "1": 512}, {"0": 0.5, "1": 0.5}, tolerance=0.05)

A clean run with no exceptions means qtest is wired up correctly.
