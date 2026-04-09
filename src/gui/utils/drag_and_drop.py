"""
Drag-and-drop file zone widget for the bottom bar.

Provides :class:`FileDropFrame`, a dashed-border ``QFrame`` that
accepts file drops and emits the resolved absolute paths via
:pyqt:`filesDropped(list[str])`.  Used by
:class:`~src.gui.file_window.FileWindow` to let users load ``.set``
and ``.csv`` files without a file dialog.
"""

import os
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtWidgets import QFrame, QVBoxLayout, QLabel


class FileDropFrame(QFrame):
    """Dashed-border drop zone that accepts file URLs and emits paths.

    Visual states:

    * **Idle** — grey dashed border with a centred instruction label.
    * **Hover** — blue dashed border with a tinted background (shown
      while a valid drag is over the frame).

    Both states have light and dark variants, toggled via
    :meth:`set_dark_mode`.

    Signals:
        filesDropped(list[str]): Emitted with a list of absolute file
            paths after a successful drop.
    """
    filesDropped = pyqtSignal(list)

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
        """Switch between light and dark idle / hover palettes.

        No-op if the mode hasn't actually changed.
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
        """Apply the grey dashed-border stylesheet (no drag in progress)."""
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
        """Apply the blue dashed-border stylesheet (valid drag hovering)."""
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
        """Accept the drag if it carries file URLs; switch to hover style."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._set_hover_style()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        """Revert to idle style when the drag leaves the frame."""
        self._set_idle_style()
        super().dragLeaveEvent(event)

    def dropEvent(self, event) -> None:
        """Resolve dropped URLs to absolute paths and emit :pyqt:`filesDropped`."""
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
