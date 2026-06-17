# Changelog

All notable changes to **qtest** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Noise-aware testing** (`qtest.noise`):
  - `NoiseModel` wrapper plus constructors `depolarizing`, `bit_flip`,
    `phase_flip`, `thermal_relaxation`, `readout_error`; channels compose
    with `+`. Named presets via `from_preset` / `available_presets`.
  - `noise_model=` argument on `assert_distribution_close` (noisy sampling)
    and `assert_state_close` (density-matrix fidelity).
  - New assertion `assert_robust_to_noise` — sweep increasing noise levels
    and bound the output's drift from the ideal distribution.
  - `--qtest-noise=<preset>` CLI flag and `[tool.qtest] default_noise`
    config field.
  - Noise fixtures (`qtest.fixtures.noise`): `light_noise`, `heavy_noise`,
    and the `depolarizing_noise` / `readout_noise` / `thermal_noise`
    factory fixtures.
- **New assertion family**:
  - `assert_entangled` / `assert_separable` — entanglement-aware state
    assertions based on entanglement entropy across a bipartition.
  - `assert_measurement_probabilities` — compare a marginal measurement
    distribution over a subset of measured bits.
  - `assert_phase` — verify the relative phase between two computational-basis
    amplitudes (modulo 2π).
  - `assert_commutes` — verify operator commutation / anticommutation.
- **Resource / cost assertions** (`qtest.assertions.resources`):
  `assert_max_depth`, `assert_max_gate_count` (total or per-gate),
  `assert_max_two_qubit_count`, and `assert_max_t_count` guard a circuit's
  structural cost so optimisation / transpilation regressions fail the build.
  Backed by the new `Backend.get_resources` method (implemented for the Qiskit
  and Cirq backends) returning a `CircuitResources` record. Pure static
  analysis — no simulation required.
- **Entanglement metrics** (`qtest.metrics`): `partial_trace`, `purity`,
  `von_neumann_entropy`, `entanglement_entropy`.
- **Cirq and PennyLane backends** (`qtest.backends`): `CirqBackend` and
  `PennyLaneBackend` now fully implement the `Backend` protocol
  (state vector, unitary, sampling) and are auto-registered as `"cirq"` and
  `"pennylane"`. Install via the new `qtest-quantum[cirq]` / `qtest-quantum[pennylane]`
  extras. Noise models (Aer-based) are not supported on these backends.
- **Visualisation** (`qtest.viz`, optional `[viz]` extra): `plot_distribution`
  and `plot_distribution_comparison` matplotlib helpers for triaging
  distribution mismatches.
- **Snapshot testing** (`qtest.snapshot`): the `qtest_snapshot` pytest fixture
  records a circuit's distribution to a golden file and compares later runs
  against it; refresh with the new `--qtest-snapshot-update` flag.
- **Richer `--qtest-summary`**: distribution / marginal assertions now record
  the distance they measure, and the summary reports sample count and
  mean / min / max measured distance (plus the active noise preset, if any).
- **OpenQASM interop** (`qtest.qasm`): `load_qasm` / `load_qasm_file` parse
  OpenQASM 2.0 / 3.0 into circuits usable with any assertion (version
  auto-detected from the header). OpenQASM 3.0 needs the new `qtest-quantum[qasm3]`
  extra.
- **Developer tooling**: filled in `tox.ini` (test / lint / type / docs
  environments mirroring CI) and `.pre-commit-config.yaml` (whitespace hygiene,
  ruff, black).

### Changed
- `Backend.run_circuit` gains an optional `noise_model` parameter and a new
  `Backend.get_density_matrix` method (implemented by `QiskitBackend`,
  default `NotImplementedError` elsewhere).

### Deprecated
- *(reserved for the next release)*

### Removed
- *(reserved for the next release)*

### Fixed
- *(reserved for the next release)*

### Security
- *(reserved for the next release)*

## [0.1.0] - 2026-05-21

The first public alpha of qtest.

### Added
- **Core assertions** (`qtest.assertions`):
  - `assert_distribution_close` — statistical comparison of measurement
    distributions with a tolerance-aware, shot-noise-friendly check.
  - `assert_state_close` — state-vector closeness with global-phase
    insensitivity and named reference states.
  - `assert_unitary` — verify that an operator is unitary via
    `U^†U ≈ I`.
  - `assert_circuit_equivalent` — verify two circuits implement the
    same unitary (process fidelity / Hilbert-Schmidt / sampling).
- **Pytest plugin** (`qtest.plugin`):
  - CLI flags `--qtest-shots`, `--qtest-tolerance`, `--qtest-seed`.
  - Markers `slow` and `hardware`.
- **Bundled pytest fixtures** (`qtest.fixtures.common_states`,
  `qtest.fixtures.common_gates`):
  - `bell_state`, `plus_state`, `minus_state`, `ghz_state` factory,
    `ghz_3`, `ghz_4`, `ghz_5`, `w_state` factory, `hadamard_circuit`
    factory, `random_clifford` factory.
- **Hypothesis strategies** (`qtest.strategies`, optional extra):
  - `quantum_circuits`, `random_gates`, `pauli_strings`,
    `random_states`, `product_states`, `random_density_matrices`.
- **Metrics** (`qtest.metrics`):
  - Distances: total variation, Hellinger, fidelity, trace distance,
    Hilbert-Schmidt distance.
  - Statistical tests: Pearson χ², Kolmogorov–Smirnov, plus
    `auto_tolerance` for shot-noise-aware bounds.
- **Backend abstraction** (`qtest.backends`):
  - `Backend` protocol and `QiskitBackend` implementation.
  - Registry helpers (`get_backend`, `register_backend`, …).
- **Documentation** (Sphinx + Furo, hosted on ReadTheDocs):
  - Installation guide, quickstart, user-guide chapters, full API
    reference.
- **Project plumbing**:
  - MIT licence, Code of Conduct (Contributor Covenant 2.1),
    contributor guide, security policy.
  - CI on Linux / macOS / Windows × Python 3.9 – 3.12.

### Known limitations
- Visualisation helpers are minimal.

[Unreleased]: https://github.com/metin-5115/qtest/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/metin-5115/qtest/releases/tag/v0.1.0
