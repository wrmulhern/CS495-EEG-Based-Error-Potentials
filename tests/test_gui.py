"""Tests for GUI modules."""

import os
import sys
from unittest.mock import Mock, patch

# Check if PyQt5 is available
try:
    from PyQt5.QtWidgets import QApplication, QWidget
    from PyQt5.QtCore import Qt
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False

def test_file_window():
    """Test FileWindow (skipped if PyQt5 not available)."""
    if not PYQT_AVAILABLE:
        print("PyQt5 not available, skipping GUI tests")
        return
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    try:
        from src.gui.file_window import FileWindow
        window = FileWindow()
        assert window.windowTitle() == "ErrP Visualizer"
        assert not window.is_dark_mode
        window.close()
        print("FileWindow test passed")
    finally:
        if app:
            app.quit()

def test_flanker_window():
    """Test FlankerWindow (skipped if PyQt5 not available)."""
    if not PYQT_AVAILABLE:
        return
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    try:
        from src.gui.flanker_window import FlankerWindow
        window = FlankerWindow()
        assert window.windowTitle() == "Record EEG — Flanker Task"
        assert window.is_dark == False
        window.close()
        print("FlankerWindow test passed")
    finally:
        if app:
            app.quit()

def test_help_dialog():
    """Test HelpDialog (skipped if PyQt5 not available)."""
    if not PYQT_AVAILABLE:
        return
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    try:
        from src.gui.help_dialog import HelpDialog
        dialog = HelpDialog()
        assert dialog.windowTitle() == "Help — ErrP Visualizer"
        dialog.close()
        print("HelpDialog test passed")
    finally:
        if app:
            app.quit()

def test_themes():
    """Test themes."""
    if not PYQT_AVAILABLE:
        return
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    try:
        from src.gui.themes.theme import apply_theme
        widget = QWidget()
        apply_theme(app, is_dark=False)
        print("Themes test passed")
    finally:
        if app:
            app.quit()

def test_colors():
    """Test color definitions."""
    if not PYQT_AVAILABLE:
        return
    from src.gui.themes.colors import get_palette
    palette = get_palette(is_dark=False)
    assert hasattr(palette, 'text')
    assert hasattr(palette, 'window')
    print("Colors test passed")

def test_utils_checkbox():
    """Test custom checkbox."""
    if not PYQT_AVAILABLE:
        return
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    try:
        from src.gui.utils.checkbox import ToggleSwitch
        cb = ToggleSwitch("Test")
        assert cb.text() == "Test"
        cb.close()
        print("ToggleSwitch test passed")
    finally:
        if app:
            app.quit()

def test_utils_drag_and_drop():
    """Test drag and drop widget."""
    if not PYQT_AVAILABLE:
        return
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    try:
        from src.gui.utils.drag_and_drop import FileDropFrame
        frame = FileDropFrame()
        assert isinstance(frame, QWidget)
        frame.close()
        print("FileDropFrame test passed")
    finally:
        if app:
            app.quit()

def test_utils_multi_select():
    """Test multi-select widget."""
    if not PYQT_AVAILABLE:
        return
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    try:
        from src.gui.utils.multi_select import MultiSelectDropdown
        combo = MultiSelectDropdown(['Ch1', 'Ch2', 'Ch3'])
        assert isinstance(combo, QWidget)
        combo.close()
        print("MultiSelectDropdown test passed")
    finally:
        if app:
            app.quit()