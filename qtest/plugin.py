"""pytest plugin for qtest.

This module is registered as a pytest plugin via the ``pytest11`` entry
point in ``pyproject.toml``::

    [project.entry-points.pytest11]
    qtest = "qtest.plugin"

so it is **auto-discovered** whenever qtest is installed alongside pytest.
The plugin can be disabled per-run with ``-p no:qtest``.

Responsibilities
----------------
1. **CLI flags** — expose ``--qtest-*`` options under a dedicated
   pytest group. Each flag whose default is ``None`` is treated as
   "unset" so that *only* explicitly-passed flags override values from
   ``pyproject.toml``.
2. **Configuration cascade** — at ``pytest_configure`` time, merge the
   ``[tool.qtest]`` section of the rootdir's ``pyproject.toml`` with the
   CLI overrides (CLI wins) and apply the result via
   :func:`qtest.configure`.
3. **Markers** — register ``quantum``, ``hardware``, and
   ``slow_quantum`` markers so ``--strict-markers`` users can apply them
   without warnings.
4. **Summary** — when ``--qtest-summary`` is passed, print a short
   end-of-session report (quantum tests run, backend, total shots,
   average distance). Tests can feed distance samples to the report via
   :func:`record_distance`.
"""

from __future__ import annotations

import statistics
from typing import Any

import pytest

from qtest.config import configure, get_config, load_from_pyproject

# --------------------------------------------------------------------------- #
# Per-session statistics                                                      #
# --------------------------------------------------------------------------- #


class _RunStats:
    """Accumulator for quantum-test statistics shown in ``--qtest-summary``.

    A single module-level instance, :data:`_STATS`, is mutated by the
    plugin's hooks and by :func:`record_distance`. It is reset at the
    start of every ``pytest_configure`` so re-running pytest within the
    same Python process (notably via :class:`pytest.Pytester`) yields
    clean numbers.
    """

    def __init__(self) -> None:
        self.quantum_tests_run: int = 0
        self.distances: list[float] = []

    def reset(self) -> None:
        self.quantum_tests_run = 0
        self.distances.clear()


_STATS = _RunStats()


def record_distance(distance: float) -> None:
    """Record a measured distance for the optional ``--qtest-summary`` report.

    Assertion helpers (or user tests) may call this to feed samples into
    the end-of-session average. Values are stored as plain floats; no
    aggregation happens until the summary is rendered.
    """
    _STATS.distances.append(distance)


# --------------------------------------------------------------------------- #
# CLI options                                                                 #
# --------------------------------------------------------------------------- #


# Map CLI dest names to ``QtestConfig`` field names. Boolean flags handled
# separately so that "flag not passed" stays distinguishable from "flag
# explicitly set to False".
_CLI_TO_FIELD: dict[str, str] = {
    "qtest_shots": "default_shots",
    "qtest_tolerance": "default_tolerance",
    "qtest_backend": "default_backend",
    "qtest_seed": "default_seed",
    "qtest_metric": "statistical_metric",
}

_BOOL_CLI_TO_FIELD: dict[str, str] = {
    "qtest_auto_tolerance": "auto_tolerance",
    "qtest_verbose_failures": "verbose_failures",
}


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("qtest", "qtest — statistical quantum testing")
    group.addoption(
        "--qtest-shots",
        action="store",
        dest="qtest_shots",
        type=int,
        default=None,
        help="Default number of shots for sampling-based assertions.",
    )
    group.addoption(
        "--qtest-tolerance",
        action="store",
        dest="qtest_tolerance",
        type=float,
        default=None,
        help="Default tolerance for distribution / state comparisons "
        "(must lie in [0, 1]).",
    )
    group.addoption(
        "--qtest-backend",
        action="store",
        dest="qtest_backend",
        default=None,
        help="Name of the qtest backend to use (must be registered).",
    )
    group.addoption(
        "--qtest-seed",
        action="store",
        dest="qtest_seed",
        type=int,
        default=None,
        help="Default seed for reproducible sampling.",
    )
    group.addoption(
        "--qtest-metric",
        action="store",
        dest="qtest_metric",
        choices=["tv", "chi_square", "hellinger"],
        default=None,
        help="Default statistical metric for distribution comparisons.",
    )
    # Boolean flags: default=None so that "not passed" can be detected and
    # left to pyproject.toml / dataclass defaults rather than overriding to
    # False.
    group.addoption(
        "--qtest-auto-tolerance",
        action="store_true",
        dest="qtest_auto_tolerance",
        default=None,
        help="Compute tolerance automatically from shots.",
    )
    group.addoption(
        "--qtest-verbose-failures",
        action="store_true",
        dest="qtest_verbose_failures",
        default=None,
        help="Include extended diagnostics in assertion failure messages.",
    )
    group.addoption(
        "--qtest-summary",
        action="store_true",
        dest="qtest_summary",
        default=False,
        help="Print a quantum-test summary at the end of the test session.",
    )


# --------------------------------------------------------------------------- #
# Configuration cascade                                                       #
# --------------------------------------------------------------------------- #


def _collect_cli_overrides(config: pytest.Config) -> dict[str, Any]:
    """Return only those CLI flags that were *explicitly* passed.

    Each ``--qtest-*`` flag uses ``default=None``; a value of ``None``
    therefore means "not passed" and is filtered out so the value from
    ``pyproject.toml`` (or the dataclass default) survives.
    """
    overrides: dict[str, Any] = {}
    for cli_name, field_name in _CLI_TO_FIELD.items():
        value = config.getoption(cli_name, default=None)
        if value is not None:
            overrides[field_name] = value
    for cli_name, field_name in _BOOL_CLI_TO_FIELD.items():
        value = config.getoption(cli_name, default=None)
        if value is not None:
            overrides[field_name] = bool(value)
    return overrides


def _load_pyproject_settings(config: pytest.Config) -> dict[str, Any]:
    """Load ``[tool.qtest]`` from the rootdir's ``pyproject.toml``.

    Missing files / sections yield ``{}``; malformed TOML or a non-table
    ``[tool.qtest]`` section propagates the underlying exception so the
    user sees the misconfiguration loudly at startup.
    """
    rootpath = getattr(config, "rootpath", None)
    if rootpath is None:
        return {}
    return load_from_pyproject(rootpath / "pyproject.toml")


def pytest_configure(config: pytest.Config) -> None:
    # Markers first so they are registered even if config application
    # below fails for some reason — strict-markers users still see them.
    config.addinivalue_line(
        "markers",
        "quantum: quantum-circuit test (counted by --qtest-summary)",
    )
    config.addinivalue_line(
        "markers",
        "hardware: test that requires real quantum hardware access",
    )
    config.addinivalue_line(
        "markers",
        "slow_quantum: quantum test that is slow "
        "(deselect with -m 'not slow_quantum')",
    )

    settings: dict[str, Any] = {}
    settings.update(_load_pyproject_settings(config))
    settings.update(_collect_cli_overrides(config))

    if settings:
        configure(**settings)

    _STATS.reset()


# --------------------------------------------------------------------------- #
# Summary tracking                                                            #
# --------------------------------------------------------------------------- #


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Count quantum-marked tests that completed their call phase.

    We count once per test (the ``call`` phase report) and only when the
    test actually ran (passed or failed). Skipped tests are excluded.
    """
    if report.when != "call":
        return
    if report.outcome not in {"passed", "failed"}:
        return
    if "quantum" in report.keywords:
        _STATS.quantum_tests_run += 1


def pytest_terminal_summary(
    terminalreporter: pytest.TerminalReporter,
    config: pytest.Config,
) -> None:
    if not config.getoption("qtest_summary", default=False):
        return

    cfg = get_config()
    n = _STATS.quantum_tests_run
    total_shots = n * cfg.default_shots
    avg_distance = statistics.fmean(_STATS.distances) if _STATS.distances else None

    terminalreporter.write_sep("=", "qtest summary")
    terminalreporter.write_line(f"Quantum tests run    : {n}")
    terminalreporter.write_line(f"Backend              : {cfg.default_backend}")
    terminalreporter.write_line(f"Total shots          : {total_shots}")
    if avg_distance is None:
        terminalreporter.write_line("Average distance     : n/a")
    else:
        terminalreporter.write_line(f"Average distance     : {avg_distance:.6f}")


__all__ = ["record_distance"]
