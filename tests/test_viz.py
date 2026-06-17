"""Tests for :mod:`qtest.viz` (require matplotlib)."""

from __future__ import annotations

import pytest

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")  # headless backend for CI

from qtest.viz import plot_distribution, plot_distribution_comparison  # noqa: E402


def test_plot_distribution_returns_axes() -> None:
    ax = plot_distribution({"00": 0.5, "11": 0.5}, title="Bell")
    assert ax.get_xlabel() == "outcome"
    assert ax.get_ylabel() == "probability"
    assert ax.get_title() == "Bell"
    assert len(ax.patches) == 2  # one bar per outcome


def test_plot_distribution_comparison_returns_axes() -> None:
    ax = plot_distribution_comparison({"00": 0.48, "11": 0.52}, {"00": 0.5, "11": 0.5})
    # expected + measured bars for each of the 2 outcomes.
    assert len(ax.patches) == 4
    assert ax.get_legend() is not None


def test_plot_distribution_empty_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        plot_distribution({})


def test_plot_comparison_empty_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        plot_distribution_comparison({}, {"0": 1.0})
