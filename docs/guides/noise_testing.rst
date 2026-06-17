Noise-aware testing
===================

Real quantum hardware is noisy. A circuit that is perfect on an ideal
simulator can still drift once depolarizing error, readout error, and
T1/T2 relaxation enter the picture. qtest lets you test *under noise* by
passing a :class:`~qtest.noise.NoiseModel` to the same assertions you
already use.

Requires the Aer simulator:

.. code-block:: bash

   pip install 'qtest-quantum[aer]'

Applying a noise model
----------------------

Build a model with one of the constructors in :mod:`qtest.noise` and pass it
via ``noise_model``. Noise spreads probability mass, so loosen the tolerance
relative to an ideal run:

.. code-block:: python

   from qiskit import QuantumCircuit
   from qtest import assert_distribution_close
   from qtest.noise import depolarizing

   def test_bell_under_noise():
       qc = QuantumCircuit(2)
       qc.h(0); qc.cx(0, 1); qc.measure_all()

       assert_distribution_close(
           qc,
           expected={"00": 0.5, "11": 0.5},
           shots=4096,
           tolerance=0.15,
           noise_model=depolarizing(0.01),
       )

Available channels
-------------------

- :func:`~qtest.noise.depolarizing` — depolarizing error on every gate.
- :func:`~qtest.noise.bit_flip` / :func:`~qtest.noise.phase_flip` — Pauli
  X / Z errors after single-qubit gates.
- :func:`~qtest.noise.thermal_relaxation` — T1/T2 relaxation.
- :func:`~qtest.noise.readout_error` — measurement error.

Layer several channels with ``+``:

.. code-block:: python

   from qtest.noise import depolarizing, readout_error

   model = depolarizing(0.005) + readout_error(0.01)

Noisy state comparison
----------------------

Under noise a circuit produces a *mixed* state. With ``noise_model`` set,
:func:`~qtest.assert_state_close` compares the resulting density matrix to the
ideal target via fidelity (``global_phase=True`` is required):

.. code-block:: python

   from qtest import assert_state_close
   from qtest.noise import thermal_relaxation

   def test_plus_fidelity_under_noise():
       qc = QuantumCircuit(1); qc.h(0)
       assert_state_close(
           qc, "plus", tolerance=0.05,
           noise_model=thermal_relaxation(t1=200.0, t2=120.0, time=1.0),
       )

Robustness sweeps
-----------------

:func:`~qtest.assert_robust_to_noise` sweeps a ladder of increasing noise
strengths and asserts the output never drifts further than ``max_distance``
from the ideal — the natural shape of an error-mitigation regression test:

.. code-block:: python

   from qtest import assert_robust_to_noise

   def test_bell_is_robust():
       qc = QuantumCircuit(2)
       qc.h(0); qc.cx(0, 1); qc.measure_all()

       assert_robust_to_noise(
           qc,
           expected={"00": 0.5, "11": 0.5},
           noise_levels=[0.001, 0.005, 0.02],
           max_distance=0.2,
           noise_type="depolarizing",
           shots=4096,
       )

Setting noise globally
----------------------

Apply a built-in preset to every sampling assertion without touching test
code, either from the command line:

.. code-block:: bash

   pytest --qtest-noise=depolarizing

or in ``pyproject.toml``:

.. code-block:: toml

   [tool.qtest]
   default_noise = "depolarizing"

An explicit ``noise_model=`` argument on an individual assertion always wins
over the global setting. Available presets are listed by
:func:`~qtest.noise.available_presets`.
