import os
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtWidgets import QFrame, QVBoxLayout, QLabel

class FileDropFrame(QFrame):
    filesDropped = pyqtSignal(list)  # List[str]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_dark_mode = False
        self.setAcceptDrops(True)
        self.setFrameShape(QFrame.StyledPanel)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(6)

        self.title = QLabel("Drag and drop one or more files here")
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setStyleSheet("color: #202124; font-size: 14px; border: none;")

        layout.addStretch(1)
        layout.addWidget(self.title)
        layout.addStretch(1)

        self._set_idle_style()

    def set_dark_mode(self, is_dark: bool) -> None:
        """
        Adjust the drop area styling for light vs dark mode while keeping
        the same layout (dashed border, rounded corners, centered label).
        """
        if self._is_dark_mode == is_dark:
            return
        self._is_dark_mode = is_dark
        self._set_idle_style()

        if self._is_dark_mode:
            self.title.setStyleSheet("color: #e8eaed; font-size: 14px; border: none;")
        else:
            self.title.setStyleSheet("color: #202124; font-size: 14px; border: none;")

    def _set_idle_style(self) -> None:
        if not self._is_dark_mode:
            self.setStyleSheet(
                """
                QFrame {
                    border: 2px dashed #9aa0a6;
                    border-radius: 6px;
                    background: #ffffff;
                }
                QLabel {
                    background: transparent;
                }
                """
            )
        else:
            self.setStyleSheet(
                """
                QFrame {
                    border: 2px dashed #5f6368;
                    border-radius: 6px;
                    background: #202124;
                }
                QLabel {
                    background: transparent;
                }
                """
            )

    def _set_hover_style(self) -> None:
        if not self._is_dark_mode:
            self.setStyleSheet(
                """
                QFrame {
                    border: 2px dashed #1a73e8;
                    border-radius: 6px;
                    background: #e8f0fe;
                }
                QLabel {
                    background: transparent;
                }
                """
            )
        else:
            self.setStyleSheet(
                """
                QFrame {
                    border: 2px dashed #8ab4f8;
                    border-radius: 6px;
                    background: #1e1e1e;
                }
                QLabel {
                    background: transparent;
                }
                """
            )

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._set_hover_style()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self._set_idle_style()
        super().dragLeaveEvent(event)

    def dropEvent(self, event) -> None:
        self._set_idle_style()

        urls = event.mimeData().urls()
        paths = []
        for u in urls:
            p = u.toLocalFile()
            if p:
                paths.append(os.path.abspath(p))

        if paths:
            self.filesDropped.emit(paths)

        event.acceptProposedAction()
