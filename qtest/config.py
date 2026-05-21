"""Global configuration for qtest.

qtest's defaults (shot count, tolerance, backend, statistical metric, ...)
can be set in three ways, in order of decreasing precedence:

1. **Per-call overrides** passed directly to assertion functions, e.g.
   ``assert_distribution_close(p, q, shots=8000)``.
2. **Programmatic configuration** via :func:`configure`, e.g.
   ``qtest.configure(default_shots=5000)``.
3. **``pyproject.toml``** under ``[tool.qtest]`` (loaded by the pytest
   plugin at startup via :func:`load_from_pyproject`).
4. **Dataclass defaults** baked into :class:`QtestConfig`.

The :func:`_resolve_value` helper encapsulates rule 1 vs the rest:
assertion functions accept ``Optional`` overrides and fall back to the
current config when the caller passes ``None``.

Singleton lifetime
------------------
A single :class:`QtestConfig` instance backs the module. ``configure``
and ``reset_config`` mutate that instance in-place, so references
obtained from :func:`get_config` remain valid across resets.

Thread-safety
-------------
The module performs no locking. qtest configuration is expected to be
set once during pytest startup (or via ``configure`` in user fixtures);
concurrent mutation from worker threads is out of scope.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Dataclass                                                                   #
# --------------------------------------------------------------------------- #


_VALID_METRICS = frozenset({"tv", "chi_square", "hellinger"})


@dataclass
class QtestConfig:
    """User-tunable defaults for qtest.

    Attributes
    ----------
    default_shots
        Default number of shots used by sampling-based assertions.
    default_tolerance
        Default tolerance (distance threshold) for distribution / state
        comparisons. Must lie in ``[0, 1]``.
    default_backend
        Name of the :mod:`qtest.backends` registry entry to use when no
        backend is passed explicitly.
    default_seed
        Default seed for reproducible sampling (``None`` means
        non-deterministic).
    auto_tolerance
        If ``True``, ``assert_distribution_close`` will compute a tolerance
        from ``shots`` via :func:`qtest.metrics.auto_tolerance` when the
        caller does not pass one.
    verbose_failures
        Whether assertion failure messages include extended diagnostics.
    statistical_metric
        Name of the default distance / test for distribution comparisons.
        One of ``"tv"`` (total variation), ``"chi_square"``, ``"hellinger"``.
    """

    default_shots: int = 1000
    default_tolerance: float = 0.05
    default_backend: str = "qiskit"
    default_seed: int | None = None
    auto_tolerance: bool = False
    verbose_failures: bool = True
    statistical_metric: str = "tv"


# --------------------------------------------------------------------------- #
# Singleton state                                                             #
# --------------------------------------------------------------------------- #


_CONFIG: QtestConfig = QtestConfig()


def get_config() -> QtestConfig:
    """Return the current :class:`QtestConfig` singleton."""
    return _CONFIG


# --------------------------------------------------------------------------- #
# Validation                                                                  #
# --------------------------------------------------------------------------- #


def _validate(name: str, value: Any) -> None:
    """Raise :class:`ValueError` if *value* is invalid for *name*.

    Validation is per-field. Fields without explicit rules (e.g.
    ``verbose_failures``) only check their basic Python type.
    """
    if name == "default_shots":
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"default_shots must be a positive integer, got {value!r}")

    elif name == "default_tolerance":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"default_tolerance must be a real number, got {value!r}")
        if not (0.0 <= float(value) <= 1.0):
            raise ValueError(f"default_tolerance must be in [0, 1], got {value!r}")

    elif name == "default_backend":
        if not isinstance(value, str) or not value:
            raise ValueError(f"default_backend must be a non-empty string, got {value!r}")

    elif name == "default_seed":
        if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
            raise ValueError(f"default_seed must be an integer or None, got {value!r}")

    elif name == "auto_tolerance":
        if not isinstance(value, bool):
            raise ValueError(f"auto_tolerance must be a bool, got {value!r}")

    elif name == "verbose_failures":
        if not isinstance(value, bool):
            raise ValueError(f"verbose_failures must be a bool, got {value!r}")

    elif name == "statistical_metric" and value not in _VALID_METRICS:
        raise ValueError(
            f"statistical_metric must be one of {sorted(_VALID_METRICS)}, " f"got {value!r}"
        )


# --------------------------------------------------------------------------- #
# Public mutators                                                             #
# --------------------------------------------------------------------------- #


_FIELD_NAMES: frozenset[str] = frozenset(f.name for f in fields(QtestConfig))


def configure(**kwargs: Any) -> None:
    """Update one or more fields of the global :class:`QtestConfig`.

    Unknown keyword arguments raise :class:`ValueError` (no silent typos).
    Each value is validated; on any validation failure the singleton is
    left **unchanged** (all-or-nothing update).
    """
    unknown = set(kwargs) - _FIELD_NAMES
    if unknown:
        raise ValueError(
            f"Unknown qtest config parameter(s): {sorted(unknown)}. "
            f"Valid parameters: {sorted(_FIELD_NAMES)}"
        )

    for name, value in kwargs.items():
        _validate(name, value)

    for name, value in kwargs.items():
        setattr(_CONFIG, name, value)


def reset_config() -> None:
    """Restore every field of the global config to its dataclass default.

    The singleton instance is preserved; only its fields are mutated, so
    references obtained from :func:`get_config` remain valid.
    """
    defaults = QtestConfig()
    for field in fields(QtestConfig):
        setattr(_CONFIG, field.name, getattr(defaults, field.name))


def _resolve_value(param_name: str, override: Any) -> Any:
    """Resolve a per-call override against the current config.

    Returns *override* if it is not ``None``; otherwise returns
    ``getattr(get_config(), param_name)``.

    A literal override of ``0`` or ``False`` is **respected** (not
    treated as "use default") — only ``None`` triggers the fallback.
    """
    if param_name not in _FIELD_NAMES:
        raise ValueError(
            f"Unknown qtest config parameter: {param_name!r}. "
            f"Valid parameters: {sorted(_FIELD_NAMES)}"
        )
    if override is not None:
        return override
    return getattr(_CONFIG, param_name)


# --------------------------------------------------------------------------- #
# pyproject.toml loader                                                       #
# --------------------------------------------------------------------------- #


def _load_tomllib() -> Any:
    """Return a TOML parser module (stdlib ``tomllib`` on 3.11+, else ``tomli``)."""
    if sys.version_info >= (3, 11):
        import tomllib

        return tomllib
    try:
        import tomli  # type: ignore[import-not-found,unused-ignore]

        return tomli
    except ImportError as exc:
        raise ImportError(
            "Reading pyproject.toml on Python < 3.11 requires the 'tomli' "
            "package. Install with: pip install tomli"
        ) from exc


def load_from_pyproject(path: Path | None = None) -> dict[str, Any]:
    """Read the ``[tool.qtest]`` section of a ``pyproject.toml`` file.

    Parameters
    ----------
    path
        Explicit path to a ``pyproject.toml``. If ``None``, the current
        working directory's ``pyproject.toml`` is used.

    Returns
    -------
    dict
        The contents of ``[tool.qtest]`` as a plain dict, or ``{}`` if the
        file does not exist, contains no ``[tool.qtest]`` section, or has
        an empty section.

    Notes
    -----
    This function only **reads** — it does not apply the values to the
    config. Callers (e.g. the pytest plugin) typically pass the result to
    :func:`configure` after validation. The function returns an empty
    dict rather than raising for missing files so it is safe to call
    unconditionally during plugin startup.
    """
    path = Path.cwd() / "pyproject.toml" if path is None else Path(path)

    if not path.is_file():
        return {}

    tomllib = _load_tomllib()
    with path.open("rb") as f:
        data = tomllib.load(f)

    section = data.get("tool", {}).get("qtest", {})
    if not isinstance(section, dict):
        raise ValueError(f"[tool.qtest] in {path} must be a table, got {type(section).__name__}")
    return dict(section)


__all__ = [
    "QtestConfig",
    "configure",
    "get_config",
    "load_from_pyproject",
    "reset_config",
]
