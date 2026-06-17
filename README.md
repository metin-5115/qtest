# qtest

**Statistical, pytest-native testing for quantum circuits.**

[![CI](https://github.com/metin-5115/qtest/actions/workflows/ci.yml/badge.svg)](https://github.com/metin-5115/qtest/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/qtest-quantum.svg)](https://pypi.org/project/qtest-quantum/)
[![Python versions](https://img.shields.io/pypi/pyversions/qtest-quantum.svg)](https://pypi.org/project/qtest-quantum/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/metin-5115/qtest/branch/main/graph/badge.svg)](https://codecov.io/gh/metin-5115/qtest)
[![Documentation Status](https://readthedocs.org/projects/qtest-quantum/badge/?version=latest)](https://qtest-quantum.readthedocs.io)

`qtest` is an open-source Python library that brings the discipline of modern software testing to quantum programs. It plugs straight into `pytest`, gives you statistical assertions designed for noisy and probabilistic outputs, and integrates with Hypothesis so you can do property-based testing on quantum circuits — without writing a single line of plumbing.

---

## Why qtest?

Testing quantum software is hard for reasons classical testing tools were never designed to handle:

- **Outputs are distributions, not values.** A measurement gives you counts that are _statistically close_ to the truth, never exact. `assertEqual` is the wrong tool.
- **State and unitary comparisons are global-phase-insensitive.** Two correct implementations of the same gate can differ by a phase that doesn't matter physically — but does matter to `numpy.allclose`.
- **Shot noise and seed drift make tests flaky.** Without a principled tolerance and a controlled seed, your CI lights up red at random.
- **Property-based testing fits quantum perfectly** — laws like unitarity, reversibility, and Bell-state symmetry are universal — but the standard Hypothesis strategies don't know about `QuantumCircuit`.

`qtest` solves these directly. It gives you tolerance-aware statistical assertions, a Hypothesis strategy set tuned for circuits, gates, and states, and a `pytest` plugin that exposes `--qtest-shots`, `--qtest-tolerance`, and `--qtest-seed` as first-class CLI flags.

---

## Quickstart

Install:

```bash
pip install qtest-quantum
```

Write your first quantum test — a Bell state should produce a 50/50 distribution over `00` and `11`:

```python
from qiskit import QuantumCircuit
from qtest import assert_distribution_close

def test_bell_state_is_balanced():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    assert_distribution_close(
        qc,
        expected={"00": 0.5, "11": 0.5},
        shots=4096,
        tolerance=0.03,
    )
```

Run it:

```bash
pytest test_bell.py -v
```

Expected output:

```
test_bell.py::test_bell_state_is_balanced PASSED                         [100%]

============================== 1 passed in 0.42s ===============================
```

That's it. No mocking, no manual shot-count math, no hand-rolled tolerance checks.

---

## Features

- **Statistical assertions built for quantum.** `assert_distribution_close`, `assert_state_close`, `assert_unitary`, and `assert_circuit_equivalent` — each one understands shot noise, fidelity, trace distance, and global phase.
- **Resource/cost assertions.** `assert_max_depth`, `assert_max_gate_count`, `assert_max_two_qubit_count`, and `assert_max_t_count` lock in the gains of an optimisation or transpiler pass and fail the build when depth, two-qubit-gate count, or T-count regress — pure static analysis, no simulation.
- **Noise-aware testing.** Inject realistic error channels (depolarizing, bit/phase flip, T1/T2 relaxation, readout) into any assertion with `noise_model=`, or sweep increasing noise with `assert_robust_to_noise` to guard error-mitigation regressions.
- **Native pytest plugin.** Drop `qtest` into any existing test suite. CLI flags `--qtest-shots`, `--qtest-tolerance`, and `--qtest-seed` let you tune every run without touching code.
- **Property-based testing for circuits.** Ready-made Hypothesis strategies for random circuits, gates, state vectors, and Haar-random unitaries.
- **Backend-agnostic by design.** Ships with Qiskit (default), Cirq, and PennyLane backends behind one `Backend` protocol — switch per call or globally with `--qtest-backend`. Install extras with `qtest-quantum[cirq]` / `qtest-quantum[pennylane]`.
- **OpenQASM interop.** `load_qasm` / `load_qasm_file` parse OpenQASM 2.0/3.0 into circuits you can assert on directly (OpenQASM 3.0 via `qtest-quantum[qasm3]`).
- **Snapshot (golden-file) testing.** Lock in a circuit's distribution with the `qtest_snapshot` fixture; refresh deliberately with `--qtest-snapshot-update`.
- **Visualisation & reporting.** `qtest.viz` plots expected-vs-measured distributions (`qtest-quantum[viz]`), and `--qtest-summary` reports measured-distance statistics across the run.
- **Deterministic by default.** A controlled seed pipeline means a test that passes locally passes on CI.
- **Zero ceremony fixtures.** `bell_state`, `plus_state`, `minus_state`, `ghz_3` / `ghz_4` / `ghz_5`, `hadamard_circuit`, and `random_clifford` come included.

---

## More examples

### State-vector assertions with global-phase tolerance

```python
import numpy as np
from qiskit import QuantumCircuit
from qtest import assert_state_close

def test_hadamard_produces_plus_state():
    qc = QuantumCircuit(1)
    qc.h(0)

    expected = np.array([1, 1]) / np.sqrt(2)
    assert_state_close(qc, expected_state=expected, tolerance=1e-9)

    # qtest also ships a named state library, so the line below is
    # equivalent — global phase is ignored by default.
    assert_state_close(qc, expected_state="plus", tolerance=1e-9)
```

### Property-based testing with Hypothesis

```python
from hypothesis import given, settings
from qtest import assert_unitary
from qtest.strategies import quantum_circuits

@given(circuit=quantum_circuits(min_qubits=2, max_qubits=4, max_depth=10))
@settings(deadline=None, max_examples=50)
def test_any_circuit_is_unitary(circuit):
    assert_unitary(circuit, tolerance=1e-9)
```

### Custom tolerance per test

```python
from qtest import assert_circuit_equivalent

def test_optimizer_preserves_semantics(original_circuit, optimized_circuit):
    assert_circuit_equivalent(
        original_circuit,
        optimized_circuit,
        tolerance=1e-7,
    )
```

`assert_circuit_equivalent` compares the two circuits' unitaries via
process fidelity, which is global-phase-insensitive by default — no extra
flag needed.

### Guarding circuit cost across a transpiler pass

```python
from qiskit import transpile
from qtest import assert_circuit_equivalent, assert_max_two_qubit_count

def test_transpilation_is_correct_and_cheaper(original_circuit):
    optimized = transpile(original_circuit, optimization_level=3)

    # 1. The optimiser must not change what the circuit computes.
    assert_circuit_equivalent(original_circuit, optimized, tolerance=1e-9)

    # 2. ...and it must not regress the expensive two-qubit-gate count.
    assert_max_two_qubit_count(optimized, max_count=12)
```

You can also set tolerance globally for an entire run:

```bash
pytest --qtest-shots=8192 --qtest-tolerance=0.01 --qtest-seed=42
```

---

## Installation

Standard install from PyPI:

```bash
pip install qtest-quantum
```

With Hypothesis (property-based testing) and Aer (high-performance simulator):

```bash
pip install "qtest-quantum[hypothesis,aer]"
```

Or pull in every optional dependency in one go:

```bash
pip install "qtest-quantum[all]"
```

For development — clone the repo and install in editable mode with the dev extras:

```bash
git clone https://github.com/metin-5115/qtest.git
cd qtest
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
pytest
```

Supported Python versions: **3.9, 3.10, 3.11, 3.12**.

---

## Documentation

Full documentation, API reference, and guides live at **[qtest-quantum.readthedocs.io](https://qtest-quantum.readthedocs.io)**.

Highlights worth bookmarking:

- [Quickstart guide](https://qtest-quantum.readthedocs.io/en/latest/quickstart.html)
- [Writing custom assertions](https://qtest-quantum.readthedocs.io/en/latest/guides/writing_assertions.html)
- [Property-based testing for quantum circuits](https://qtest-quantum.readthedocs.io/en/latest/guides/property_testing.html)
- [API reference](https://qtest-quantum.readthedocs.io/en/latest/api/)

---

## Contributing

Contributions are welcome and appreciated — bug reports, feature requests, documentation fixes, and pull requests of any size. Start with [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow, coding conventions, and how to run the test suite locally. By participating you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

If you'd like to discuss a larger change before opening a PR, open a [GitHub discussion](https://github.com/metin-5115/qtest/discussions) or an issue first.

---

## License

`qtest` is released under the [MIT License](LICENSE). You are free to use it in commercial and non-commercial projects — attribution is appreciated.

### Citation

If `qtest` supports your research, please cite it:

```bibtex
@software{qtest,
  author  = {Tuncbilek, Metin},
  title   = {qtest: Statistical, pytest-native testing for quantum circuits},
  year    = {2026},
  url     = {https://github.com/metin-5115/qtest}
}
```

---

## Acknowledgments

`qtest` stands on the shoulders of remarkable open-source work:

- The **[Qiskit](https://qiskit.org/)** team at IBM Quantum, whose simulators, circuit model, and quantum-information utilities make this library possible.
- The **[pytest](https://pytest.org/)** maintainers, whose plugin architecture is so clean that bolting on a quantum testing layer felt natural.
- The **[Hypothesis](https://hypothesis.works/)** team, who proved that property-based testing belongs in every serious Python project — including quantum ones.

Thank you to everyone who files an issue, opens a PR, or simply tries `qtest` on their own circuits. Quantum software deserves the same testing rigor as the classical software we ship every day, and you are helping make that happen.
