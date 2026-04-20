"""
Unified QPalette theming for the Fusion style.

Replaces the separate ``light_theme`` and ``dark_theme`` modules with a
single :func:`apply_theme` that derives every ``QPalette`` role from the
centralized :class:`~src.gui.themes.colors.Palette`.
"""

from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtWidgets import QApplication

from src.gui.themes.colors import get_palette


def apply_theme(app: QApplication, is_dark: bool) -> None:
    """Set the Fusion style and a theme-appropriate ``QPalette`` on *app*."""
    app.setStyle("Fusion")

    p = get_palette(is_dark)
    pal = QPalette()

    pal.setColor(QPalette.Window, QColor(p.window))
    pal.setColor(QPalette.WindowText, QColor(p.text))

    pal.setColor(QPalette.Base, QColor(p.surface))
    pal.setColor(QPalette.AlternateBase, QColor(p.surface_alt))

    pal.setColor(QPalette.Text, QColor(p.text))
    pal.setColor(QPalette.Button, QColor(p.surface_elevated))
    pal.setColor(QPalette.ButtonText, QColor(p.text))

    pal.setColor(QPalette.ToolTipBase, QColor(p.surface_hover))
    pal.setColor(QPalette.ToolTipText, QColor(p.text))

    pal.setColor(QPalette.Highlight, QColor(p.accent))
    pal.setColor(QPalette.HighlightedText, QColor(p.window))

    app.setPalette(pal)
