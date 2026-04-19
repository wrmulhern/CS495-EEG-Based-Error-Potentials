"""Tests for visualizer module."""

import matplotlib
matplotlib.use('Agg')  # non-GUI backend
import matplotlib.pyplot as plt
from src.data_visualization.visualizer import _apply_mpl_theme


def test_apply_mpl_theme():
    """Test theme application."""
    fig, ax = plt.subplots()
    _apply_mpl_theme(fig, [ax], "light")
    # Just check it doesn't crash
    assert True