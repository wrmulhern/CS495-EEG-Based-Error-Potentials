from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtWidgets import (
    QApplication
)

def apply_light_theme(app: QApplication) -> None:
    app.setStyle("Fusion")

    pal = QPalette()

    pal.setColor(QPalette.Window, QColor(245, 245, 245))
    pal.setColor(QPalette.WindowText, QColor(32, 33, 36))

    pal.setColor(QPalette.Base, QColor(255, 255, 255))
    pal.setColor(QPalette.AlternateBase, QColor(240, 240, 240))

    pal.setColor(QPalette.Text, QColor(32, 33, 36))
    pal.setColor(QPalette.Button, QColor(255, 255, 255))
    pal.setColor(QPalette.ButtonText, QColor(32, 33, 36))

    pal.setColor(QPalette.Highlight, QColor(26, 115, 232))
    pal.setColor(QPalette.HighlightedText, QColor(255, 255, 255))

    app.setPalette(pal)
