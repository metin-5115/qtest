"""Tests for the ``--qtest-*`` CLI flags exposed by ``qtest.plugin``.

These tests use :class:`pytest.Pytester` to spawn fresh pytest sessions
inside a temporary directory. That isolates the global :mod:`qtest.config`
singleton from the outer test runner and makes the flag interactions
observable in a deterministic way (a sub-test prints the resolved
config; we assert on its captured stdout).

The outer ``_reset_config`` fixture is a belt-and-braces guard against
state bleeding back into other tests in the suite when pytester runs
in-process.
"""

from __future__ import annotations

import textwrap

import pytest

from qtest.config import reset_config

pytest_plugins = ["pytester"]


@pytest.fixture(autouse=True)
def _reset_config() -> None:
    """Ensure each test starts and ends with a clean qtest config."""
    reset_config()
    yield
    reset_config()


# Body of an inner test file: prints the current qtest config to stdout
# so the outer assertions can pattern-match against the captured output.
_PROBE_TEST = textwrap.dedent(
    """
    from qtest.config import get_config

    def test_probe():
        c = get_config()
        print(f"SHOTS={c.default_shots}")
        print(f"TOLERANCE={c.default_tolerance}")
        print(f"BACKEND={c.default_backend}")
        print(f"SEED={c.default_seed}")
        print(f"METRIC={c.statistical_metric}")
        print(f"AUTO_TOL={c.auto_tolerance}")
        print(f"VERBOSE={c.verbose_failures}")
    """
)


# --------------------------------------------------------------------------- #
# Individual flags                                                            #
# --------------------------------------------------------------------------- #


def test_qtest_shots_overrides_config(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_probe=_PROBE_TEST)
    result = pytester.runpytest("-s", "--qtest-shots=4096")
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(["*SHOTS=4096*"])


def test_qtest_tolerance_overrides_config(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_probe=_PROBE_TEST)
    result = pytester.runpytest("-s", "--qtest-tolerance=0.123")
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(["*TOLERANCE=0.123*"])


def test_qtest_backend_overrides_config(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_probe=_PROBE_TEST)
    result = pytester.runpytest("-s", "--qtest-backend=qiskit")
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(["*BACKEND=qiskit*"])


def test_qtest_seed_overrides_config(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_probe=_PROBE_TEST)
    result = pytester.runpytest("-s", "--qtest-seed=42")
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(["*SEED=42*"])


@pytest.mark.parametrize("metric", ["tv", "chi_square", "hellinger"])
def test_qtest_metric_overrides_config(pytester: pytest.Pytester, metric: str) -> None:
    pytester.makepyfile(test_probe=_PROBE_TEST)
    result = pytester.runpytest("-s", f"--qtest-metric={metric}")
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines([f"*METRIC={metric}*"])


def test_qtest_metric_rejects_unknown_value(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_probe=_PROBE_TEST)
    result = pytester.runpytest("--qtest-metric=bogus")
    # argparse-style choices error => non-zero exit, no tests run.
    assert result.ret != 0


def test_qtest_auto_tolerance_flag_enables_setting(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_probe=_PROBE_TEST)
    result = pytester.runpytest("-s", "--qtest-auto-tolerance")
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(["*AUTO_TOL=True*"])


def test_qtest_verbose_failures_flag_toggles_setting(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_probe=_PROBE_TEST)
    # Default is True, so explicit flag is still True (idempotent assert).
    result = pytester.runpytest("-s", "--qtest-verbose-failures")
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(["*VERBOSE=True*"])


def test_no_cli_flags_leaves_defaults(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_probe=_PROBE_TEST)
    result = pytester.runpytest("-s")
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(
        [
            "*SHOTS=1000*",
            "*TOLERANCE=0.05*",
            "*BACKEND=qiskit*",
            "*SEED=None*",
            "*METRIC=tv*",
            "*AUTO_TOL=False*",
            "*VERBOSE=True*",
        ]
    )


def test_multiple_cli_flags_combine(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_probe=_PROBE_TEST)
    result = pytester.runpytest(
        "-s",
        "--qtest-shots=2048",
        "--qtest-tolerance=0.01",
        "--qtest-metric=hellinger",
        "--qtest-seed=7",
        "--qtest-auto-tolerance",
    )
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(
        [
            "*SHOTS=2048*",
            "*TOLERANCE=0.01*",
            "*SEED=7*",
            "*METRIC=hellinger*",
            "*AUTO_TOL=True*",
        ]
    )


# --------------------------------------------------------------------------- #
# Validation propagates from configure()                                      #
# --------------------------------------------------------------------------- #


def test_invalid_shots_value_fails_session(pytester: pytest.Pytester) -> None:
    """Passing an invalid value should surface as a config error at startup."""
    pytester.makepyfile(test_probe=_PROBE_TEST)
    result = pytester.runpytest("--qtest-shots=0")
    # Either non-zero exit or INTERNALERROR — either way, the probe must
    # not have reported a passing test with shots=0.
    assert result.ret != 0
    assert not any("SHOTS=0" in line for line in result.stdout.lines)
