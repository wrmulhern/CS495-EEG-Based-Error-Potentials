from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
)

class ToggleSwitch(QCheckBox):
    """
    A simple toggle-looking checkbox (still a QCheckBox under the hood).
    """

    def __init__(self, text=""):
        super().__init__(text)
        self.setCursor(Qt.PointingHandCursor)
        self.setChecked(False)
        self.setStyleSheet(
            """
            QCheckBox {
                spacing: 10px;
                color: #202124;
                font-size: 13px;
            }
            QCheckBox::indicator {
                width: 44px;
                height: 24px;
            }
            QCheckBox::indicator:unchecked {
                border-radius: 12px;
                background: #dadce0;
            }
            QCheckBox::indicator:unchecked:pressed {
                background: #c7c9cc;
            }
            QCheckBox::indicator:checked {
                border-radius: 12px;
                background: #1a73e8;
            }
            QCheckBox::indicator:checked:pressed {
                background: #1666c1;
            }
            """
        )
