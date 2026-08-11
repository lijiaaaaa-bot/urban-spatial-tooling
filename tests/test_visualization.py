"""Tests for src/visualization.py — module surface only.

Heavy figure rendering is intentionally skipped; these tests only verify
that the module imports and exposes its key entry points.
"""

import matplotlib

matplotlib.use("Agg")  # headless backend, no display needed

import matplotlib.pyplot as plt  # noqa: E402

from src import visualization as viz  # noqa: E402


def test_module_imports():
    assert viz is not None


def test_plot_land_use_exists():
    assert callable(viz.plot_land_use)


def test_add_scale_bar_exists():
    assert callable(viz.add_scale_bar)


def test_add_north_arrow_exists():
    assert callable(viz.add_north_arrow)


def test_add_north_arrow_runs_without_rendering():
    """The north arrow is a single annotation — cheap and side-effect free."""
    fig, ax = plt.subplots()
    viz.add_north_arrow(ax)
    assert len(ax.texts) >= 1
    plt.close(fig)
