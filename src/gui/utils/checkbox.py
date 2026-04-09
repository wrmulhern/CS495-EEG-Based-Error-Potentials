"""
iOS-style toggle switch implemented as a styled ``QCheckBox``.

Used by :class:`~src.gui.file_window.FileWindow` for the dark-mode
toggle in the top bar.  The widget preserves the standard
``QCheckBox`` API (``isChecked()``, ``stateChanged`` signal, etc.)
while rendering as a 44×24 px rounded pill via Qt stylesheets.
"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QCheckBox


class ToggleSwitch(QCheckBox):
    """Pill-shaped toggle switch with light and dark theme variants.

    Internally just a ``QCheckBox`` with a large indicator and no text.
    Call :meth:`set_dark_mode` whenever the app theme changes to swap
    between the two colour palettes.

    Parameters:
        text (str): Optional label (usually ``""`` for a bare toggle).
    """

    def __init__(self, text: str = ""):
        super().__init__(text)
        self._is_dark_mode = False
        self.setCursor(Qt.PointingHandCursor)
        self.setChecked(False)
        self._apply_styles()

    def set_dark_mode(self, is_dark: bool) -> None:
        """Switch between light and dark colour palettes.

        No-op if the mode hasn't actually changed.
        """
        if self._is_dark_mode == is_dark:
            return
        self._is_dark_mode = is_dark
        self._apply_styles()

    def _apply_styles(self) -> None:
        """Re-apply the full stylesheet for the current theme."""
        if not self._is_dark_mode:
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
        else:
            # Dark mode: same structure, higher contrast colors
            self.setStyleSheet(
                """
                QCheckBox {
                    spacing: 10px;
                    color: #e8eaed;
                    font-size: 13px;
                }
                QCheckBox::indicator {
                    width: 44px;
                    height: 24px;
                }
                QCheckBox::indicator:unchecked {
                    border-radius: 12px;
                    background: #5f6368;
                }
                QCheckBox::indicator:unchecked:pressed {
                    background: #80868b;
                }
                QCheckBox::indicator:checked {
                    border-radius: 12px;
                    background: #8ab4f8;
                }
                QCheckBox::indicator:checked:pressed {
                    background: #669df6;
                }
                """
            )
