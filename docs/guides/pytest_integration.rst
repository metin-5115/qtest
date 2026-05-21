pytest integration
==================

qtest ships as a real :mod:`pytest` plugin. Installing ``qtest`` is the
*only* thing you need to do — the plugin auto-registers via its
``pytest11`` entry point, three CLI flags appear under ``pytest --help``,
and the bundled circuit fixtures become available once you enable
them in your ``conftest.py``.

.. contents:: On this page
   :local:
   :depth: 1

CLI flags
---------

The plugin exposes three knobs that apply to the entire test session:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Flag
     - Default
     - Effect
   * - ``--qtest-shots``
     - 1024
     - Default shot count for the assertion-internal backend run.
   * - ``--qtest-tolerance``
     - 0.05
     - Default total-variation tolerance for
       :func:`~qtest.assert_distribution_close`.
   * - ``--qtest-seed``
     - System entropy
     - Master seed for the backend RNG; identical seeds → identical
       sample sequences.

A typical CI invocation:

.. code-block:: bash

   pytest -v \
       --qtest-shots 8192 \
       --qtest-tolerance 0.02 \
       --qtest-seed 12345

You can also set those values once-per-project in ``pyproject.toml``:

.. code-block:: toml

   [tool.qtest]
   shots = 4096
   tolerance = 0.03
   seed = 42

The plugin reads ``[tool.qtest]`` at session start and CLI flags
override file settings.

Built-in circuit fixtures
-------------------------

The fixtures live in two modules — ``qtest.fixtures.common_states``
for state-preparation circuits and ``qtest.fixtures.common_gates`` for
gate-layer factories. They are **plain pytest plugins**, so you enable
them by listing the modules under ``pytest_plugins`` in your top-level
``conftest.py``:

.. code-block:: python

   # tests/conftest.py
   pytest_plugins = [
       "qtest.fixtures.common_states",
       "qtest.fixtures.common_gates",
   ]

Once that one line is in place, every test in the suite can pull in
any of these fixtures by simply naming them as a parameter:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Fixture
     - Yields
   * - ``bell_state``
     - A 2-qubit Bell-state preparation circuit.
   * - ``plus_state``
     - A 1-qubit :math:`|+\rangle` preparation circuit.
   * - ``minus_state``
     - A 1-qubit :math:`|-\rangle` preparation circuit.
   * - ``ghz_state``
     - Factory ``n -> QuantumCircuit`` for an :math:`n`-qubit GHZ state.
   * - ``ghz_3`` / ``ghz_4`` / ``ghz_5``
     - Shortcuts for 3-, 4-, and 5-qubit GHZ circuits.
   * - ``w_state``
     - Factory ``n -> QuantumCircuit`` for an :math:`n`-qubit W state.
   * - ``hadamard_circuit``
     - Factory ``n -> QuantumCircuit`` for an :math:`n`-qubit
       all-Hadamard circuit.
   * - ``random_clifford``
     - Factory yielding a seeded random Clifford circuit.

Using the fixtures together
---------------------------

.. code-block:: python

   from qtest import assert_distribution_close

   def test_bell_is_balanced(bell_state):
       # bell_state already has H + CNOT applied; we just need to measure.
       qc = bell_state.copy()
       qc.measure_all()
       assert_distribution_close(
           qc,
           expected={"00": 0.5, "11": 0.5},
           shots=4096,
           tolerance=0.03,
       )

   def test_ghz_is_balanced(ghz_5):
       qc = ghz_5.copy()
       qc.measure_all()
       assert_distribution_close(
           qc,
           expected={"00000": 0.5, "11111": 0.5},
           shots=4096,
           tolerance=0.03,
       )

   def test_random_clifford_is_unitary(random_clifford):
       from qtest import assert_unitary
       qc = random_clifford(num_qubits=4, depth=10, seed=42)
       assert_unitary(qc, tolerance=1e-9)

Markers
-------

qtest registers two pytest markers you can use to gate tests:

.. code-block:: python

   import pytest

   @pytest.mark.slow
   def test_high_shot_property(): ...

   @pytest.mark.hardware
   def test_real_qpu(): ...

Run only the fast tests during local development:

.. code-block:: bash

   pytest -m "not slow"

Or filter out hardware tests when you're offline:

.. code-block:: bash

   pytest -m "not hardware"

Reproducibility
---------------

The single most useful thing the plugin does for you is make runs
reproducible. When ``--qtest-seed`` is set, every assertion that
needs randomness (the default backend's RNG, the sampling-based
modes of :func:`~qtest.assert_circuit_equivalent`) is seeded from
the same root. The net effect: a passing run on your laptop is
bit-for-bit identical to the CI run, so flakes are real flakes,
not seed drift.

CI configuration recipes
------------------------

A pragmatic split:

.. code-block:: yaml

   # .github/workflows/ci.yml — excerpt
   - name: Fast tests
     run: pytest -m "not slow" --qtest-shots 1024 --qtest-tolerance 0.05

   - name: Slow tests
     run: pytest -m "slow" --qtest-shots 8192 --qtest-tolerance 0.02
     if: github.event_name == 'schedule'

PR runs stay snappy; the nightly build runs the precise, shot-heavy
properties.

Where to go next
----------------

* :doc:`writing_assertions` — the assertions these fixtures feed into.
* :doc:`property_testing` — how the slow markers and high shot counts
  play with Hypothesis profiles.
* :doc:`../api/fixtures` — the full autogenerated reference for every
  bundled fixture.
