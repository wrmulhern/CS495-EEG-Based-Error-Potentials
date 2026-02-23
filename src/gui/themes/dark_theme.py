from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtWidgets import QApplication


def apply_dark_theme(app: QApplication) -> None:
    """
    Apply a high-contrast dark theme to the PyQt application.

    This mirrors the structure of the existing light theme function so that
    switching between themes simply swaps palette colors while keeping the
    same Qt "Fusion" style.
    """
    app.setStyle("Fusion")

    pal = QPalette()

    # Window / background
    pal.setColor(QPalette.Window, QColor(18, 18, 18))          # main window background
    pal.setColor(QPalette.Base, QColor(24, 24, 24))            # text entry backgrounds
    pal.setColor(QPalette.AlternateBase, QColor(32, 33, 36))   # alternating rows / frames

    # Text
    pal.setColor(QPalette.WindowText, QColor(232, 234, 237))   # primary text
    pal.setColor(QPalette.Text, QColor(232, 234, 237))
    pal.setColor(QPalette.ToolTipBase, QColor(60, 64, 67))
    pal.setColor(QPalette.ToolTipText, QColor(232, 234, 237))

    # Buttons
    pal.setColor(QPalette.Button, QColor(32, 33, 36))
    pal.setColor(QPalette.ButtonText, QColor(232, 234, 237))

    # Highlights / selection
    pal.setColor(QPalette.Highlight, QColor(138, 180, 248))    # blue accent
    pal.setColor(QPalette.HighlightedText, QColor(0, 0, 0))

    app.setPalette(pal)

