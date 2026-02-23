from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QCheckBox


class ToggleSwitch(QCheckBox):
    """
    A simple toggle-looking checkbox (still a QCheckBox under the hood).

    The widget keeps the same API as a normal `QCheckBox` but exposes a
    `set_dark_mode` helper so the containing window can restyle the control
    when the global theme changes, instead of hard-coding colors here.
    """

    def __init__(self, text: str = ""):
        super().__init__(text)
        self._is_dark_mode = False
        self.setCursor(Qt.PointingHandCursor)
        self.setChecked(False)
        self._apply_styles()

    def set_dark_mode(self, is_dark: bool) -> None:
        """
        Update the toggle's appearance for light vs dark mode while keeping
        the same structure (indicator sizes, border radius, etc.).
        """
        if self._is_dark_mode == is_dark:
            return
        self._is_dark_mode = is_dark
        self._apply_styles()

    def _apply_styles(self) -> None:
        if not self._is_dark_mode:
            # Light mode: matches the original styling
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
