"""Integration tests for the ``qtest`` pytest plugin.

Covers plugin discovery, ``pyproject.toml`` loading, the CLI > pyproject
> dataclass precedence chain, marker registration, the optional
``--qtest-summary`` terminal report, and the ``-p no:qtest`` opt-out.
"""

from __future__ import annotations

import textwrap

import pytest

from qtest.config import reset_config

pytest_plugins = ["pytester"]


@pytest.fixture(autouse=True)
def _reset_config() -> None:
    reset_config()
    yield
    reset_config()


# --------------------------------------------------------------------------- #
# Plugin discovery / opt-out                                                  #
# --------------------------------------------------------------------------- #


def test_plugin_is_loaded(pytester: pytest.Pytester) -> None:
    """qtest should appear in ``pytest --trace-config`` output."""
    pytester.makepyfile(test_x="def test_a(): pass\n")
    result = pytester.runpytest("--trace-config")
    result.assert_outcomes(passed=1)
    joined = "\n".join(result.stdout.lines)
    assert "qtest" in joined


def test_plugin_exposes_qtest_group_in_help(pytester: pytest.Pytester) -> None:
    result = pytester.runpytest("--help")
    assert result.ret == 0
    joined = "\n".join(result.stdout.lines)
    assert "--qtest-shots" in joined
    assert "--qtest-tolerance" in joined
    assert "--qtest-backend" in joined
    assert "--qtest-summary" in joined


def test_plugin_can_be_disabled(pytester: pytest.Pytester) -> None:
    """With ``-p no:qtest`` the ``--qtest-*`` flags must no longer exist."""
    pytester.makepyfile(test_x="def test_a(): pass\n")
    result = pytester.runpytest("-p", "no:qtest", "--qtest-shots=1000")
    # Unrecognised argument => pytest usage error (exit 4).
    assert result.ret != 0


# --------------------------------------------------------------------------- #
# pyproject.toml loading                                                      #
# --------------------------------------------------------------------------- #


_PROBE_TEST = textwrap.dedent(
    """
    from qtest.config import get_config

    def test_probe():
        c = get_config()
        print(f"SHOTS={c.default_shots}")
        print(f"TOLERANCE={c.default_tolerance}")
        print(f"METRIC={c.statistical_metric}")
    """
)


def test_pyproject_values_are_applied(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_probe=_PROBE_TEST)
    pytester.makepyprojecttoml(
        textwrap.dedent(
            """
            [tool.qtest]
            default_shots = 8192
            default_tolerance = 0.02
            statistical_metric = "hellinger"
            """
        )
    )
    result = pytester.runpytest("-s")
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(
        [
            "*SHOTS=8192*",
            "*TOLERANCE=0.02*",
            "*METRIC=hellinger*",
        ]
    )


def test_cli_flag_overrides_pyproject(pytester: pytest.Pytester) -> None:
    """CLI flag must win over a value set in ``[tool.qtest]``."""
    pytester.makepyfile(test_probe=_PROBE_TEST)
    pytester.makepyprojecttoml(
        textwrap.dedent(
            """
            [tool.qtest]
            default_shots = 8192
            default_tolerance = 0.02
            statistical_metric = "hellinger"
            """
        )
    )
    result = pytester.runpytest(
        "-s",
        "--qtest-shots=128",
        "--qtest-metric=tv",
    )
    result.assert_outcomes(passed=1)
    # CLI wins for these two; pyproject still wins for tolerance.
    result.stdout.fnmatch_lines(
        [
            "*SHOTS=128*",
            "*TOLERANCE=0.02*",
            "*METRIC=tv*",
        ]
    )


def test_missing_pyproject_falls_back_to_defaults(pytester: pytest.Pytester) -> None:
    """No pyproject.toml at all => config should match dataclass defaults."""
    pytester.makepyfile(test_probe=_PROBE_TEST)
    result = pytester.runpytest("-s")
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(
        [
            "*SHOTS=1000*",
            "*TOLERANCE=0.05*",
            "*METRIC=tv*",
        ]
    )


# --------------------------------------------------------------------------- #
# Marker registration                                                         #
# --------------------------------------------------------------------------- #


def test_markers_are_registered(pytester: pytest.Pytester) -> None:
    """``--markers`` should list all three qtest-owned markers."""
    result = pytester.runpytest("--markers")
    assert result.ret == 0
    joined = "\n".join(result.stdout.lines)
    assert "@pytest.mark.quantum" in joined
    assert "@pytest.mark.hardware" in joined
    assert "@pytest.mark.slow_quantum" in joined


def test_markers_pass_strict_markers(pytester: pytest.Pytester) -> None:
    """A test marked ``@pytest.mark.quantum`` must not trigger a warning
    even under ``--strict-markers``."""
    pytester.makepyfile(
        test_marked=textwrap.dedent(
            """
            import pytest

            @pytest.mark.quantum
            def test_q(): pass

            @pytest.mark.hardware
            def test_h(): pass

            @pytest.mark.slow_quantum
            def test_s(): pass
            """
        )
    )
    result = pytester.runpytest("--strict-markers")
    result.assert_outcomes(passed=3)


# --------------------------------------------------------------------------- #
# --qtest-summary                                                             #
# --------------------------------------------------------------------------- #


def test_summary_flag_prints_report(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        test_marked=textwrap.dedent(
            """
            import pytest

            @pytest.mark.quantum
            def test_q1(): pass

            @pytest.mark.quantum
            def test_q2(): pass

            def test_plain(): pass
            """
        )
    )
    result = pytester.runpytest("--qtest-summary")
    result.assert_outcomes(passed=3)
    joined = "\n".join(result.stdout.lines)
    assert "qtest summary" in joined
    assert "Quantum tests run    : 2" in joined
    assert "Backend              : qiskit" in joined
    # default_shots=1000, two quantum tests => 2000 total shots.
    assert "Total shots          : 2000" in joined
    assert "Average distance     : n/a" in joined


def test_summary_includes_recorded_distances(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        test_marked=textwrap.dedent(
            """
            import pytest
            from qtest.plugin import record_distance

            @pytest.mark.quantum
            def test_q1():
                record_distance(0.10)

            @pytest.mark.quantum
            def test_q2():
                record_distance(0.20)
            """
        )
    )
    result = pytester.runpytest("--qtest-summary")
    result.assert_outcomes(passed=2)
    joined = "\n".join(result.stdout.lines)
    assert "Quantum tests run    : 2" in joined
    # mean(0.10, 0.20) = 0.15
    assert "Average distance     : 0.150000" in joined


def test_summary_not_printed_without_flag(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        test_marked=textwrap.dedent(
            """
            import pytest

            @pytest.mark.quantum
            def test_q(): pass
            """
        )
    )
    result = pytester.runpytest()
    result.assert_outcomes(passed=1)
    joined = "\n".join(result.stdout.lines)
    assert "qtest summary" not in joined


def test_summary_counts_failing_quantum_tests(pytester: pytest.Pytester) -> None:
    """Failed quantum tests still count toward the run total."""
    pytester.makepyfile(
        test_marked=textwrap.dedent(
            """
            import pytest

            @pytest.mark.quantum
            def test_pass(): pass

            @pytest.mark.quantum
            def test_fail(): assert False
            """
        )
    )
    result = pytester.runpytest("--qtest-summary")
    result.assert_outcomes(passed=1, failed=1)
    joined = "\n".join(result.stdout.lines)
    assert "Quantum tests run    : 2" in joined


def test_summary_excludes_skipped_quantum_tests(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        test_marked=textwrap.dedent(
            """
            import pytest

            @pytest.mark.quantum
            def test_ok(): pass

            @pytest.mark.quantum
            @pytest.mark.skip(reason="x")
            def test_skipped(): pass
            """
        )
    )
    result = pytester.runpytest("--qtest-summary")
    result.assert_outcomes(passed=1, skipped=1)
    joined = "\n".join(result.stdout.lines)
    assert "Quantum tests run    : 1" in joined
