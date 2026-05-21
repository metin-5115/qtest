# Changelog

All notable changes to **qtest** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- *(reserved for the next release)*

### Changed
- *(reserved for the next release)*

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
- **Bundled fixtures** (`qtest.fixtures`):
  - `bell_circuit`, `ghz_circuit`, `plus_circuit`, `minus_circuit`,
    `w_circuit`, `hadamards`, `random_clifford_circuit`.
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
- Only the Qiskit backend is shipped. Cirq and PennyLane adapters are
  scaffolded but not yet implemented.
- Visualisation helpers are minimal.

[Unreleased]: https://github.com/metin-5115/qtest/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/metin-5115/qtest/releases/tag/v0.1.0
