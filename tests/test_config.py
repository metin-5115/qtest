"""Tests for :mod:`qtest.config`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from qtest.config import (
    QtestConfig,
    _resolve_value,
    configure,
    get_config,
    load_from_pyproject,
    reset_config,
)

# --------------------------------------------------------------------------- #
# Autouse fixture: ensure tests do not leak config state                      #
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _isolate_config() -> Any:
    """Snapshot config before each test, restore after."""
    yield
    reset_config()


# --------------------------------------------------------------------------- #
# Defaults                                                                    #
# --------------------------------------------------------------------------- #


def test_defaults() -> None:
    c = get_config()
    assert c.default_shots == 1000
    assert c.default_tolerance == 0.05
    assert c.default_backend == "qiskit"
    assert c.default_seed is None
    assert c.auto_tolerance is False
    assert c.verbose_failures is True
    assert c.statistical_metric == "tv"


def test_dataclass_can_be_constructed_with_overrides() -> None:
    c = QtestConfig(default_shots=8000, statistical_metric="hellinger")
    assert c.default_shots == 8000
    assert c.statistical_metric == "hellinger"
    assert c.default_tolerance == 0.05  # untouched


# --------------------------------------------------------------------------- #
# Singleton behaviour                                                         #
# --------------------------------------------------------------------------- #


def test_get_config_returns_same_instance_across_calls() -> None:
    assert get_config() is get_config()


def test_get_config_reference_survives_reset() -> None:
    """reset_config() must mutate the singleton in place, not replace it."""
    ref = get_config()
    configure(default_shots=5000)
    reset_config()
    assert get_config() is ref
    assert ref.default_shots == 1000


# --------------------------------------------------------------------------- #
# configure()                                                                 #
# --------------------------------------------------------------------------- #


def test_configure_updates_single_field() -> None:
    configure(default_shots=5000)
    assert get_config().default_shots == 5000


def test_configure_updates_multiple_fields() -> None:
    configure(default_shots=2000, default_tolerance=0.1, statistical_metric="hellinger")
    c = get_config()
    assert c.default_shots == 2000
    assert c.default_tolerance == 0.1
    assert c.statistical_metric == "hellinger"


def test_configure_unknown_param_raises() -> None:
    with pytest.raises(ValueError, match="Unknown qtest config parameter"):
        configure(nonexistent_param=1)


def test_configure_is_atomic_on_failure() -> None:
    """If any value fails validation, no field should be updated."""
    original_shots = get_config().default_shots
    with pytest.raises(ValueError):
        configure(default_shots=8000, default_tolerance=99.0)
    assert get_config().default_shots == original_shots


# --------------------------------------------------------------------------- #
# Validation                                                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad", [0, -1, -1000])
def test_validation_rejects_non_positive_shots(bad: int) -> None:
    with pytest.raises(ValueError):
        configure(default_shots=bad)


def test_validation_rejects_non_int_shots() -> None:
    with pytest.raises(ValueError):
        configure(default_shots=1000.0)


def test_validation_rejects_bool_as_shots() -> None:
    with pytest.raises(ValueError):
        configure(default_shots=True)


@pytest.mark.parametrize("bad", [-0.1, 1.1, 2.0, -1.0])
def test_validation_rejects_out_of_range_tolerance(bad: float) -> None:
    with pytest.raises(ValueError):
        configure(default_tolerance=bad)


@pytest.mark.parametrize("ok", [0.0, 0.05, 0.5, 1.0])
def test_validation_accepts_valid_tolerance(ok: float) -> None:
    configure(default_tolerance=ok)
    assert get_config().default_tolerance == ok


@pytest.mark.parametrize("metric", ["tv", "chi_square", "hellinger"])
def test_validation_accepts_whitelisted_metrics(metric: str) -> None:
    configure(statistical_metric=metric)
    assert get_config().statistical_metric == metric


@pytest.mark.parametrize("bad", ["TV", "kl", "bogus", "", None])
def test_validation_rejects_unknown_metric(bad: Any) -> None:
    with pytest.raises(ValueError):
        configure(statistical_metric=bad)


def test_validation_rejects_empty_backend_name() -> None:
    with pytest.raises(ValueError):
        configure(default_backend="")


def test_validation_rejects_non_int_seed() -> None:
    with pytest.raises(ValueError):
        configure(default_seed=3.14)


def test_validation_accepts_none_seed() -> None:
    configure(default_seed=None)
    assert get_config().default_seed is None


def test_validation_accepts_int_seed() -> None:
    configure(default_seed=42)
    assert get_config().default_seed == 42


@pytest.mark.parametrize("bad", [1, 0, "yes", None])
def test_validation_rejects_non_bool_auto_tolerance(bad: Any) -> None:
    with pytest.raises(ValueError):
        configure(auto_tolerance=bad)


# --------------------------------------------------------------------------- #
# reset_config()                                                              #
# --------------------------------------------------------------------------- #


def test_reset_restores_all_fields_to_defaults() -> None:
    configure(
        default_shots=9999,
        default_tolerance=0.5,
        statistical_metric="chi_square",
        default_seed=123,
        auto_tolerance=True,
        verbose_failures=False,
    )
    reset_config()
    c = get_config()
    assert c.default_shots == 1000
    assert c.default_tolerance == 0.05
    assert c.statistical_metric == "tv"
    assert c.default_seed is None
    assert c.auto_tolerance is False
    assert c.verbose_failures is True


# --------------------------------------------------------------------------- #
# _resolve_value                                                              #
# --------------------------------------------------------------------------- #


def test_resolve_value_returns_override_when_not_none() -> None:
    assert _resolve_value("default_shots", 4242) == 4242


def test_resolve_value_falls_back_to_config_when_none() -> None:
    configure(default_shots=2048)
    assert _resolve_value("default_shots", None) == 2048


def test_resolve_value_respects_literal_zero_override() -> None:
    """A literal 0 should be treated as an explicit override, not 'use default'."""
    assert _resolve_value("default_shots", 0) == 0


def test_resolve_value_respects_literal_false_override() -> None:
    assert _resolve_value("auto_tolerance", False) is False


def test_resolve_value_unknown_param_raises() -> None:
    with pytest.raises(ValueError):
        _resolve_value("not_a_real_field", 42)


# --------------------------------------------------------------------------- #
# load_from_pyproject                                                         #
# --------------------------------------------------------------------------- #


def test_load_from_pyproject_reads_tool_qtest(tmp_path: Path) -> None:
    pp = tmp_path / "pyproject.toml"
    pp.write_text(
        "[tool.qtest]\n"
        "default_shots = 8192\n"
        "default_tolerance = 0.02\n"
        'statistical_metric = "hellinger"\n',
        encoding="utf-8",
    )
    result = load_from_pyproject(pp)
    assert result == {
        "default_shots": 8192,
        "default_tolerance": 0.02,
        "statistical_metric": "hellinger",
    }


def test_load_from_pyproject_returns_empty_for_missing_file(tmp_path: Path) -> None:
    assert load_from_pyproject(tmp_path / "nope.toml") == {}


def test_load_from_pyproject_returns_empty_for_no_qtest_section(tmp_path: Path) -> None:
    pp = tmp_path / "pyproject.toml"
    pp.write_text("[tool.black]\nline-length = 100\n", encoding="utf-8")
    assert load_from_pyproject(pp) == {}


def test_load_from_pyproject_returns_empty_for_empty_qtest_section(tmp_path: Path) -> None:
    pp = tmp_path / "pyproject.toml"
    pp.write_text("[tool.qtest]\n", encoding="utf-8")
    assert load_from_pyproject(pp) == {}


def test_load_from_pyproject_with_path_object(tmp_path: Path) -> None:
    pp = tmp_path / "pyproject.toml"
    pp.write_text("[tool.qtest]\ndefault_shots = 333\n", encoding="utf-8")
    # Both Path and string should work
    assert load_from_pyproject(pp) == {"default_shots": 333}
    assert load_from_pyproject(str(pp)) == {"default_shots": 333}  # type: ignore[arg-type]


def test_load_from_pyproject_rejects_non_table_qtest_section(tmp_path: Path) -> None:
    pp = tmp_path / "pyproject.toml"
    pp.write_text('[tool]\nqtest = "oops"\n', encoding="utf-8")
    with pytest.raises(ValueError):
        load_from_pyproject(pp)


def test_load_from_pyproject_default_path_is_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pp = tmp_path / "pyproject.toml"
    pp.write_text("[tool.qtest]\ndefault_shots = 77\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert load_from_pyproject() == {"default_shots": 77}


# --------------------------------------------------------------------------- #
# Integration: load_from_pyproject + configure                                #
# --------------------------------------------------------------------------- #


def test_load_then_configure_round_trip(tmp_path: Path) -> None:
    pp = tmp_path / "pyproject.toml"
    pp.write_text(
        "[tool.qtest]\n" "default_shots = 4096\n" 'statistical_metric = "chi_square"\n',
        encoding="utf-8",
    )
    values = load_from_pyproject(pp)
    configure(**values)
    c = get_config()
    assert c.default_shots == 4096
    assert c.statistical_metric == "chi_square"
