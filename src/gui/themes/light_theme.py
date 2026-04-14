"""
Light-mode QPalette for the Fusion style.

Applies a Google-Material-inspired colour scheme as the application-wide
default, using colors from :mod:`src.gui.themes.colors`.
"""

from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtWidgets import QApplication

from src.gui.themes.colors import LIGHT as P


def apply_light_theme(app: QApplication) -> None:
    """Set the Fusion style and a light ``QPalette`` on *app*."""
    app.setStyle("Fusion")

    pal = QPalette()

    pal.setColor(QPalette.Window, QColor("#f5f5f5"))
    pal.setColor(QPalette.WindowText, QColor(P.text))

    pal.setColor(QPalette.Base, QColor(P.surface))
    pal.setColor(QPalette.AlternateBase, QColor("#f0f0f0"))

    pal.setColor(QPalette.Text, QColor(P.text))
    pal.setColor(QPalette.Button, QColor(P.surface))
    pal.setColor(QPalette.ButtonText, QColor(P.text))

    pal.setColor(QPalette.Highlight, QColor(P.accent))
    pal.setColor(QPalette.HighlightedText, QColor(P.surface))

    app.setPalette(pal)
