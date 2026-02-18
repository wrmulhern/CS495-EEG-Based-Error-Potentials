import os
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QPalette, QColor
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
    QMessageBox,
)

class FileDropFrame(QFrame):
    filesDropped = pyqtSignal(list)  # List[str]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            """
            FileDropFrame {
                border: 2px dashed #9aa0a6;
                border-radius: 6px;
                background: #ffffff;
            }
            QLabel {
                background: transparent;
            }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(6)

        self.title = QLabel("Drag and drop one or more files here")
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setStyleSheet("color: #202124; font-size: 14px; border: none;")

        layout.addStretch(1)
        layout.addWidget(self.title)
        layout.addStretch(1)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet(
                """
                QFrame {
                    border: 2px dashed #1a73e8;
                    border-radius: 6px;
                    background: #e8f0fe;
                }
                """
            )
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.setStyleSheet(
            """
            QFrame {
                border: 2px dashed #9aa0a6;
                border-radius: 6px;
                background: #ffffff;
            }
            """
        )
        super().dragLeaveEvent(event)

    def dropEvent(self, event) -> None:
        self.setStyleSheet(
            """
            QFrame {
                border: 2px dashed #9aa0a6;
                border-radius: 6px;
                background: #ffffff;
            }
            """
        )

        urls = event.mimeData().urls()
        paths = []
        for u in urls:
            p = u.toLocalFile()
            if p:
                paths.append(os.path.abspath(p))

        if paths:
            self.filesDropped.emit(paths)

        event.acceptProposedAction()
