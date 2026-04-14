"""
Dark-mode QPalette for the Fusion style.

Mirrors :func:`~src.gui.themes.light_theme.apply_light_theme` but
uses dark backgrounds and light text, consistent with the palette
defined in :mod:`src.gui.themes.colors`.
"""

from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtWidgets import QApplication

from src.gui.themes.colors import DARK as P


def apply_dark_theme(app: QApplication) -> None:
    """Set the Fusion style and a dark ``QPalette`` on *app*."""
    app.setStyle("Fusion")

    pal = QPalette()

    pal.setColor(QPalette.Window, QColor(P.window))
    pal.setColor(QPalette.Base, QColor(P.surface_alt))
    pal.setColor(QPalette.AlternateBase, QColor(P.surface_elevated))

    pal.setColor(QPalette.WindowText, QColor(P.text))
    pal.setColor(QPalette.Text, QColor(P.text))
    pal.setColor(QPalette.ToolTipBase, QColor(P.surface_hover))
    pal.setColor(QPalette.ToolTipText, QColor(P.text))

    pal.setColor(QPalette.Button, QColor(P.surface_elevated))
    pal.setColor(QPalette.ButtonText, QColor(P.text))

    pal.setColor(QPalette.Highlight, QColor(P.accent))
    pal.setColor(QPalette.HighlightedText, QColor("#000000"))

    app.setPalette(pal)
