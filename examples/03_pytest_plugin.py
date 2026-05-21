"""Using the qtest pytest plugin: config, CLI flags, and markers.

Run: pytest examples/03_pytest_plugin.py -v --qtest-summary

After ``pip install qtest`` the plugin is auto-discovered (entry point
``pytest11/qtest``). It gives you three things:

    1. A configuration cascade -- pyproject.toml < CLI flags < programmatic
       qtest.configure() calls. The plugin applies (1) and (2) at startup.
    2. Quantum-aware markers: @pytest.mark.quantum, @pytest.mark.hardware,
       @pytest.mark.slow_quantum. Pre-registered, so --strict-markers
       won't complain.
    3. An opt-in end-of-session report with ``--qtest-summary``.

Quick CLI tour
--------------
    # Override shots / tolerance / metric without touching code:
    pytest examples/03_pytest_plugin.py --qtest-shots=8000 --qtest-tolerance=0.02

    # Pick a different statistical metric:
    pytest examples/03_pytest_plugin.py --qtest-metric=hellinger

    # Compute tolerance from the shot count automatically:
    pytest examples/03_pytest_plugin.py --qtest-auto-tolerance

    # Skip slow quantum tests in CI:
    pytest examples/03_pytest_plugin.py -m "not slow_quantum"

    # Run only hardware tests:
    pytest examples/03_pytest_plugin.py -m hardware

    # Print a summary block at the end:
    pytest examples/03_pytest_plugin.py --qtest-summary

    # Disable the plugin (e.g. while debugging a clash):
    pytest examples/03_pytest_plugin.py -p no:qtest

Equivalent pyproject.toml configuration
---------------------------------------
    [tool.qtest]
    default_shots = 4000
    default_tolerance = 0.05
    default_backend = "qiskit"
    statistical_metric = "tv"
    auto_tolerance = false
    verbose_failures = true

A matching conftest.py for a project-wide setup
-----------------------------------------------
    # tests/conftest.py
    import qtest

    def pytest_configure(config):
        # Anything you can't express in pyproject.toml goes here, e.g.
        # registering a custom backend:
        # qtest.backends.register_backend("mybackend", MyBackend)
        qtest.configure(default_seed=42)
"""

from __future__ import annotations

import pytest
from qiskit import QuantumCircuit

from qtest.assertions import assert_distribution_close, assert_state_close
from qtest.plugin import record_distance


# --------------------------------------------------------------------------- #
# Marker: @pytest.mark.quantum                                                #
# --------------------------------------------------------------------------- #
#
# Any test marked ``quantum`` is counted in the ``--qtest-summary``
# block at the end of the run. The marker has no other side effects --
# use it generously on any test that exercises a quantum circuit.


@pytest.mark.quantum
def test_plus_state_via_hadamard() -> None:
    qc = QuantumCircuit(1)
    qc.h(0)
    assert_state_close(qc, expected_state="plus", tolerance=1e-9)


# --------------------------------------------------------------------------- #
# Marker: @pytest.mark.slow_quantum                                           #
# --------------------------------------------------------------------------- #
#
# Tag tests that take more than ~1s so CI can deselect them with
# ``-m "not slow_quantum"``. We mark ``quantum`` too so the summary
# counts it.


@pytest.mark.quantum
@pytest.mark.slow_quantum
def test_high_shot_count_bell_distribution() -> None:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    # 20_000 shots is overkill for two outcomes; here just to show the
    # marker makes the test deselectable in fast CI lanes.
    assert_distribution_close(
        qc,
        expected={"00": 0.5, "11": 0.5},
        shots=20_000,
        tolerance=0.02,
        seed=0,
    )


# --------------------------------------------------------------------------- #
# Marker: @pytest.mark.hardware                                               #
# --------------------------------------------------------------------------- #
#
# Marks tests that require credentials / real QPU access. Always
# skipped here because no hardware backend is wired up; the marker
# itself is just a filter.


@pytest.mark.hardware
@pytest.mark.skip(reason="needs an IBM Quantum account in this demo")
def test_runs_on_real_hardware() -> None:
    raise NotImplementedError  # would call a hardware backend


# --------------------------------------------------------------------------- #
# Parametrisation + record_distance => richer --qtest-summary                 #
# --------------------------------------------------------------------------- #
#
# ``record_distance`` lets a test feed its measured distance into the
# summary block so the "Average distance" line reports something
# meaningful. Combine with @pytest.mark.parametrize to scan a knob.


@pytest.mark.quantum
@pytest.mark.parametrize("shots", [500, 2000, 8000])
def test_distribution_distance_vs_shots(shots: int) -> None:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    # Run the circuit ourselves so we can measure the empirical TV
    # distance and report it via record_distance. assert_distribution_close
    # would compute the same number internally, but doesn't expose it.
    from qtest.backends.registry import get_default_backend
    from qtest.metrics import total_variation_distance

    counts = get_default_backend().run_circuit(qc, shots=shots, seed=123)
    total = sum(counts.values())
    measured = {label: count / total for label, count in counts.items()}
    expected = {"00": 0.5, "11": 0.5}
    distance = total_variation_distance(measured, expected)
    record_distance(distance)

    # And the actual assertion:
    assert_distribution_close(
        qc,
        expected={"00": 0.5, "11": 0.5},
        shots=shots,
        tolerance=0.1,
        seed=123,
    )
