"""Visualisation helpers for quantum measurement distributions.

These are convenience plotting functions for use in notebooks, debugging
sessions, and test-failure triage. They require matplotlib, installed via the
optional ``viz`` extra::

    pip install 'qtest[viz]'

matplotlib is imported lazily, so importing :mod:`qtest.viz` (or :mod:`qtest`)
does not require it — only calling a plotting function does. Each function
draws onto a provided ``Axes`` (or a freshly created one) and returns that
``Axes`` so callers can further customise or save the figure.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_MATPLOTLIB_MISSING_MSG = (
    "qtest.viz requires matplotlib. Install it with:\n  pip install 'qtest[viz]'"
)


def _require_pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised only without matplotlib
        raise ImportError(_MATPLOTLIB_MISSING_MSG) from exc
    return plt


def _validate_distribution(name: str, dist: Mapping[str, float]) -> None:
    if not isinstance(dist, Mapping) or not dist:
        raise ValueError(f"{name} must be a non-empty mapping of bitstring -> value")


def plot_distribution(
    distribution: Mapping[str, float],
    ax: Any | None = None,
    *,
    title: str | None = None,
    color: str | None = None,
    sort_keys: bool = True,
) -> Any:
    """Bar-plot a single measurement distribution.

    Parameters
    ----------
    distribution
        Mapping ``{bitstring: probability}`` (or counts).
    ax
        Existing matplotlib ``Axes`` to draw on. A new one is created if
        ``None``.
    title
        Optional axes title.
    color
        Optional bar colour.
    sort_keys
        Sort bitstrings lexicographically (default) for a stable x-axis.

    Returns
    -------
    matplotlib.axes.Axes
    """
    _validate_distribution("distribution", distribution)
    plt = _require_pyplot()
    if ax is None:
        _, ax = plt.subplots()

    keys = sorted(distribution) if sort_keys else list(distribution)
    values = [distribution[k] for k in keys]
    ax.bar(keys, values, color=color)
    ax.set_xlabel("outcome")
    ax.set_ylabel("probability")
    if title:
        ax.set_title(title)
    ax.tick_params(axis="x", rotation=90 if keys and len(keys[0]) > 4 else 0)
    return ax


def plot_distribution_comparison(
    measured: Mapping[str, float],
    expected: Mapping[str, float],
    ax: Any | None = None,
    *,
    title: str | None = None,
) -> Any:
    """Grouped bar-plot of *measured* vs *expected* distributions.

    Outcomes present in either distribution are shown side by side, making it
    easy to eyeball where a circuit's sampled output drifts from the target —
    handy when triaging an ``assert_distribution_close`` failure.

    Returns
    -------
    matplotlib.axes.Axes
    """
    _validate_distribution("measured", measured)
    _validate_distribution("expected", expected)
    plt = _require_pyplot()
    if ax is None:
        _, ax = plt.subplots()

    keys = sorted(set(measured) | set(expected))
    x = range(len(keys))
    width = 0.4
    ax.bar(
        [i - width / 2 for i in x], [expected.get(k, 0.0) for k in keys], width, label="expected"
    )
    ax.bar(
        [i + width / 2 for i in x], [measured.get(k, 0.0) for k in keys], width, label="measured"
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(keys, rotation=90 if keys and len(keys[0]) > 4 else 0)
    ax.set_xlabel("outcome")
    ax.set_ylabel("probability")
    ax.legend()
    if title:
        ax.set_title(title)
    return ax


__all__ = ["plot_distribution", "plot_distribution_comparison"]
