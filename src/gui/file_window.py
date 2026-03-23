import os
import logging
from typing import List, Optional

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from PyQt5.QtCore import Qt, QUrl, QTimer, pyqtSignal
from PyQt5.QtGui import QKeySequence, QIcon, QDesktopServices, QColor
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
    QTabWidget,
    QMessageBox,
    QShortcut,
    QSlider,
    QProgressDialog,
)

from src.data_processing.data_loader import read_epochs_eeglab_minimal, read_csv_data
from src.data_processing.file_validator import FileValidator, FileValidationError
from src.data_processing.data_processor import average_epochs, select_time_window
from src.data_visualization.visualizer import plot_evoked, plot_topomap, plot_joint, plot_topomap_frame
from .utils.drag_and_drop import FileDropFrame
from .utils.checkbox import ToggleSwitch
from .themes.light_theme import apply_light_theme
from .themes.dark_theme import apply_dark_theme

logger = logging.getLogger(__name__)

LIVE_RECORDING_URL = "https://google.com"  # swap in real URL

#
#
# MULTI-SELECT DROPDOWN WIDGET
#
#
class MultiSelectItemDelegate:
    """Helper to style selected items with blue background."""
    @staticmethod
    def update_item_style(item: QListWidgetItem, is_selected: bool):
        """Update item style based on selection state."""
        if is_selected:
            item.setBackground(QColor("#1a73e8"))
            item.setForeground(QColor("#ffffff"))
        else:
            item.setBackground(QColor(Qt.white))
            item.setForeground(QColor(Qt.black))


class MultiSelectDropdown(QWidget):
    """
    A custom multi-select dropdown widget with checkboxes.
    Supports "Select All" / "Deselect All" functionality.
    Stays open until clicking outside or pressing Enter.
    """
    selectionChanged = pyqtSignal(list)  # Emits list of selected items
    confirmed = pyqtSignal()  # Emits when Enter is pressed

    def __init__(self, items: List[str], parent=None):
        super().__init__(parent)
        self.items = items
        self.selected = set()
        self.is_open = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Button to show/hide dropdown
        self.button = QPushButton("All Channels")
        self.button.setStyleSheet("text-align: left; padding-left: 8px;")
        self.button.clicked.connect(self.toggle_dropdown)
        layout.addWidget(self.button)

        # Dropdown frame (initially hidden)
        self.dropdown_frame = QFrame()
        self.dropdown_frame.setFrameShape(QFrame.StyledPanel)
        self.dropdown_frame.setWindowFlags(Qt.Popup | Qt.FramelessWindowHint)

        dropdown_layout = QVBoxLayout(self.dropdown_frame)
        dropdown_layout.setContentsMargins(0, 0, 0, 0)
        dropdown_layout.setSpacing(0)

        # List widget with items
        self.list_widget = QListWidget()
        self.list_widget.setMaximumHeight(250)
        self.list_widget.setSelectionMode(QListWidget.NoSelection)  # Disable blue highlight
        self.list_widget.itemClicked.connect(self._on_item_clicked)

        # Store the original keyPressEvent method and override it
        self._original_list_keypress = self.list_widget.keyPressEvent
        self.list_widget.keyPressEvent = self._on_list_key_press

        for i, item_text in enumerate(items):
            item = QListWidgetItem(item_text)
            self.list_widget.addItem(item)

        dropdown_layout.addWidget(self.list_widget)
        self.dropdown_frame.setLayout(dropdown_layout)
        self.dropdown_frame.hide()

        # Install event filter to detect when dropdown loses focus
        self.dropdown_frame.installEventFilter(self)

    def toggle_dropdown(self):
        if self.dropdown_frame.isVisible():
            self.close_dropdown()
        else:
            self.open_dropdown()

    def open_dropdown(self):
        """Show the dropdown below the button."""
        # Position below button
        pos = self.button.mapToGlobal(self.button.rect().bottomLeft())
        self.dropdown_frame.move(pos)
        self.dropdown_frame.resize(self.button.width(), 250)
        self.dropdown_frame.show()
        self.list_widget.setFocus()
        self.is_open = True  # Set flag after showing

    def close_dropdown(self):
        """Hide the dropdown and confirm the selection."""
        self.dropdown_frame.hide()
        self.is_open = False  # Set flag after hiding
        self.confirmed.emit()

    def _on_item_clicked(self, item: QListWidgetItem):
        """Handle item click to toggle selection."""
        idx = self.list_widget.row(item)
        item_text = self.items[idx]

        # Special handling for "All Channels"
        if item_text == "All Channels":
            # If "All Channels" is currently selected
            if item_text in self.selected:
                # Deselect all items
                self.selected.clear()
                for i in range(self.list_widget.count()):
                    list_item = self.list_widget.item(i)
                    MultiSelectItemDelegate.update_item_style(list_item, False)
            else:
                # Select all items
                self.selected = set(self.items)
                for i in range(self.list_widget.count()):
                    list_item = self.list_widget.item(i)
                    MultiSelectItemDelegate.update_item_style(list_item, True)
        else:
            # Regular item clicked - toggle selection
            is_currently_selected = item_text in self.selected

            if is_currently_selected:
                self.selected.discard(item_text)
            else:
                self.selected.add(item_text)

            # Update styling for this item
            MultiSelectItemDelegate.update_item_style(item, not is_currently_selected)

            # Check if all items (excluding "All Channels") are selected
            all_items = set(self.items[1:])  # Skip "All Channels"
            individual_selected = self.selected.copy()
            individual_selected.discard("All Channels")

            all_checkbox = self.list_widget.item(0)
            if individual_selected == all_items:
                self.selected.add("All Channels")
                MultiSelectItemDelegate.update_item_style(all_checkbox, True)
            else:
                self.selected.discard("All Channels")
                MultiSelectItemDelegate.update_item_style(all_checkbox, False)

        self.selectionChanged.emit(list(self.selected))
        self._update_button_text()

    def _on_list_key_press(self, event):
        """Handle key press in list widget."""
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            self.close_dropdown()
        elif event.key() == Qt.Key_Escape:
            self.close_dropdown()
        else:
            self._original_list_keypress(event)

    def eventFilter(self, obj, event):
        """Handle events on the dropdown frame to close when it loses focus."""
        if obj == self.dropdown_frame:
            if event.type() == 3:  # QEvent.FocusOut
                if self.dropdown_frame.isVisible():
                    self.close_dropdown()
                    return True
        return super().eventFilter(obj, event)

    def _update_button_text(self):
        """Update button text to show current selection."""
        if not self.selected:
            text = "No Selection"
        elif "All Channels" in self.selected and len(self.selected) == len(self.items):
            text = "All Channels"
        else:
            # Show count or abbreviated list
            if len(self.selected) == 1:
                text = list(self.selected)[0]
            else:
                text = f"{len(self.selected)} selected"
        self.button.setText(text)

    def get_selected(self) -> List[str]:
        """Return list of selected items."""
        return list(self.selected)

    def set_items(self, items: List[str]):
        """Update the list of items."""
        self.items = items
        self.list_widget.clear()
        self.selected.clear()

        for item_text in items:
            item = QListWidgetItem(item_text)
            self.list_widget.addItem(item)

        self._update_button_text()

    def keyPressEvent(self, event):
        """Handle Escape key to close dropdown."""
        if event.key() == Qt.Key_Escape and self.dropdown_frame.isVisible():
            self.close_dropdown()
        else:
            super().keyPressEvent(event)

    def focusOutEvent(self, event):
        """Close dropdown when parent widget loses focus."""
        if self.dropdown_frame.isVisible():
            self.close_dropdown()
        super().focusOutEvent(event)

#
#
# HELP DIALOG
#
#
class HelpDialog(QDialog):
    """
    Modal help / how-to dialog accessible from the top bar.
    Uses QTextBrowser so the content is scrollable and supports basic HTML.
    """

    _CONTENT = """
<h2 style="margin-top:0;">ErrP Visualizer &mdash; Quick Guide</h2>

<h3>What is an ErrP?</h3>
<p>An <b>Error-Related Potential (ErrP)</b> is a brain signal that appears in EEG when
a person perceives or makes an error. Two main components:</p>
<ul>
  <li><b>ERN / Ne</b> (50&ndash;150 ms) &mdash; negative deflection shortly after the error,
      generated in the anterior cingulate cortex.</li>
  <li><b>Pe</b> (200&ndash;400 ms) &mdash; positive deflection reflecting conscious error awareness.</li>
</ul>

<h3>End-to-end workflow</h3>
<ol>
  <li>Click <b>Record EEG</b> in the top bar. This opens a web application where you can run
      a <b>Flanker Task</b> &mdash; a standard cognitive paradigm that reliably elicits ErrP
      signals using a connected BCI headset.</li>
  <li>Complete the task. The web app exports your session as a <b>.set</b> or <b>.csv</b> file.</li>
  <li>Drop that file into this app to visualize your ErrP.</li>
</ol>

<h3>Loading files</h3>
<ul>
  <li>Drag and drop one or more files onto the drop zone, or click <b>Browse (&hellip;)</b>.</li>
  <li><b>.set</b> files are loaded directly. <b>.csv</b> files (e.g. from OpenBCI Ganglion)
      are <b>automatically converted</b> to .set format &mdash; no manual steps required.
      A converted file is saved alongside the original CSV.</li>
  <li>Each file opens in its own <b>tab</b>. Tabs are fully independent.</li>
  <li>Files load <b>lazily</b>: data is only read when you first click <b>Visualize</b> on that tab.</li>
  <li>Close a single tab with its <b>&times;</b> button, or remove all tabs with <b>Clear All</b>.</li>
</ul>

<h3>Graph types</h3>
<ul>
  <li><b>ErrP Time Series</b> &mdash; averaged ERP waveform across all (or selected) channels.
      Best for inspecting the ERN and Pe components over time.</li>
  <li><b>Topographic Map</b> &mdash; scalp voltage map at up to three time points.
      Requires &ge;19 channels. The epoch window is fixed to the full range.</li>
  <li><b>Joint Maps</b> &mdash; time series and topomaps combined in one figure.
      Topomap times outside the epoch window show as <i>Out of range</i> placeholders.</li>
</ul>

<h3>Graph options</h3>
<ul>
  <li><b>Epoch (ms)</b> &mdash; crop the time axis. Leave blank for the full epoch.
      Disabled automatically for Topographic Map.</li>
  <li><b>Sensor</b> &mdash; plot a single channel instead of all channels (Time Series only).</li>
  <li><b>Topomap times (s)</b> &mdash; three time points (in seconds) for the scalp maps.</li>
  <li><b>Display Events and Responses</b> &mdash; overlays the ERN window (blue, 50&ndash;150 ms)
      and Pe window (green, 200&ndash;400 ms) with hover-activated labels.</li>
</ul>

<h3>Downloading a graph</h3>
<p>Click <b>Download Graph</b> in the bottom bar to save the currently displayed figure
as a high-resolution PNG (300&thinsp;dpi).</p>

<h3>Dark mode</h3>
<p>Toggle <b>Dark mode</b> in the top bar. The theme applies to both the Qt UI and the
embedded Matplotlib figures.</p>

<h3>Supported file formats</h3>
<ul>
  <li><b>EEGLAB .set</b> &mdash; epoched data with &ge;2 trials. Companion <b>.fdt</b> files
      are handled automatically.</li>
  <li><b>.csv</b> &mdash; OpenBCI Ganglion format. Automatically converted to .set on load.</li>
</ul>
"""

    def __init__(self, is_dark: bool = False, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Help — ErrP Visualizer")
        self.setMinimumSize(620, 540)
        self.resize(660, 580)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 16)
        layout.setSpacing(12)

        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(True)
        self.browser.setHtml(self._CONTENT)
        self.browser.setFrameShape(QFrame.NoFrame)
        layout.addWidget(self.browser, stretch=1)

        btn_box = QDialogButtonBox(QDialogButtonBox.Close)
        btn_box.rejected.connect(self.accept)
        layout.addWidget(btn_box)

        self._apply_theme(is_dark)

    def _apply_theme(self, is_dark: bool):
        if is_dark:
            self.setStyleSheet(
                "QDialog { background: #1e1e1e; }"
                "QTextBrowser { background: #1e1e1e; color: #e8eaed; border: none; font-size: 13px; }"
                "QPushButton { background: #303134; color: #e8eaed; border: 1px solid #5f6368;"
                " border-radius: 4px; padding: 4px 16px; }"
                "QPushButton:hover { background: #3c4043; }"
            )
        else:
            self.setStyleSheet(
                "QDialog { background: #ffffff; }"
                "QTextBrowser { background: #ffffff; color: #202124; border: none; font-size: 13px; }"
                "QPushButton { background: #ffffff; color: #202124; border: 1px solid #dadce0;"
                " border-radius: 4px; padding: 4px 16px; }"
                "QPushButton:hover { background: #f1f3f4; }"
            )

#
#
# FileTab: Class for each tab (file) that will be owned in FileWindow
#
#
class FileTab(QWidget):
    """
    Self-contained widget representing a single loaded .set file.

    Contained within FileTab:
      - The matplotlib canvas / figure
      - Graph Options controls
      - Visualize button
      - Cached EpochsData (not loaded until visualize clicked)

    Independent from FileTab:
      - Dark-mode state
      - The drag/drop zone and browse button
    """

    def __init__(self, filepath: str, is_dark_mode: bool = False, parent=None):
        super().__init__(parent)

        self.filepath = filepath
        self._is_dark_mode = is_dark_mode

        icon_path = os.path.join(os.path.dirname(__file__), "..", "assets", "icon.png")
        self.setWindowIcon(QIcon(icon_path))

        # Per tab EEG state, loaded once visualized clicked (lazy loading)
        self.current_epochs = None
        self._all_sensors: List[str] = ["All Channels"]
        self._last_time_series_selection: List[str] = ["All Channels"]
        self._last_graph_type: str = "ErrP Time Series"
        self.events_checkbox_checked: bool = False

        # Animation state (for animated topomap mode)
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._on_anim_tick)
        self._anim_playing = False
        self._anim_evoked = None
        self._anim_theme = "light"
        self._anim_global_vmax = None

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(18)

        # Graph
        self.graph_frame = self._build_graph_frame()
        root.addWidget(self.graph_frame, stretch=3)

        # Options
        options_box = self._build_options_panel()
        root.addWidget(options_box, stretch=1)

        # Apply initial theme
        self._apply_mpl_theme(self.figure)


    def _build_graph_frame(self) -> QFrame:
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(10, 10, 10, 10)

        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        self._draw_placeholder()

        layout.addWidget(self.canvas, stretch=1)
        return frame

    def _build_options_panel(self) -> QGroupBox:
        box = QGroupBox("Graph Options")
        self.options_box = box
        box.setStyleSheet(
            "QGroupBox { font-size: 13px; font-weight: 600; margin-top: 12px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }"
        )
        layout = QVBoxLayout(box)
        layout.setContentsMargins(14, 28, 14, 14)
        layout.setSpacing(14)

        # Epoch inputs
        self.epoch_container = QWidget()
        epoch_layout = QVBoxLayout(self.epoch_container)
        epoch_layout.setContentsMargins(0, 0, 0, 0)
        epoch_layout.setSpacing(6)

        self.epoch_label = QLabel("Epoch (in ms)")
        self.epoch_label.setStyleSheet("color: #202124; font-size: 12px;")

        epoch_row = QHBoxLayout()
        epoch_row.setSpacing(10)
        self.epoch_start = QLineEdit()
        self.epoch_start.setPlaceholderText("Start")
        self.epoch_start.setFixedWidth(110)
        self.epoch_start.textChanged.connect(self.mark_needs_update)

        self.epoch_end = QLineEdit()
        self.epoch_end.setPlaceholderText("End")
        self.epoch_end.setFixedWidth(110)
        self.epoch_end.textChanged.connect(self.mark_needs_update)

        dash = QLabel("—")
        dash.setAlignment(Qt.AlignCenter)
        dash.setFixedWidth(16)

        epoch_row.addWidget(self.epoch_start)
        epoch_row.addWidget(dash)
        epoch_row.addWidget(self.epoch_end)
        epoch_row.addStretch(1)

        epoch_layout.addWidget(self.epoch_label)
        epoch_layout.addLayout(epoch_row)
        layout.addWidget(self.epoch_container)

        # Sensor dropdown
        self.sensor_container = QWidget()
        sensor_layout = QVBoxLayout(self.sensor_container)
        sensor_layout.setContentsMargins(0, 0, 0, 0)
        sensor_layout.setSpacing(6)
        sensor_label = QLabel("Sensor(s)")
        sensor_label.setStyleSheet("color: #202124; font-size: 12px;")
        self.sensor_combo = MultiSelectDropdown(["All Channels"])
        self.sensor_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.sensor_combo.confirmed.connect(self.mark_needs_update)
        sensor_layout.addWidget(sensor_label)
        sensor_layout.addWidget(self.sensor_combo)
        layout.addWidget(self.sensor_container)

        # Graph type dropdown
        graph_type_label = QLabel("Graph Type")
        graph_type_label.setStyleSheet("color: #202124; font-size: 12px;")
        self.graph_type_combo = QComboBox()
        self.graph_type_combo.addItems(["ErrP Time Series", "Topographic Map", "Joint Maps"])
        self.graph_type_combo.currentTextChanged.connect(self._on_graph_type_changed)
        self.graph_type_combo.currentTextChanged.connect(self.mark_needs_update)
        layout.addWidget(graph_type_label)
        layout.addWidget(self.graph_type_combo)

        # Topomap times
        self.topo_times_container = QWidget()
        topo_layout = QVBoxLayout(self.topo_times_container)
        topo_layout.setContentsMargins(0, 0, 0, 0)
        topo_layout.setSpacing(6)
        topo_label = QLabel("Topomap times (s)")
        topo_label.setStyleSheet("color: #202124; font-size: 12px;")
        topo_layout.addWidget(topo_label)

        topo_row = QHBoxLayout()
        topo_row.setSpacing(8)
        self.topo_time_1 = QLineEdit()
        self.topo_time_1.setPlaceholderText("0.1")
        self.topo_time_1.setFixedWidth(70)
        self.topo_time_1.textChanged.connect(self.mark_needs_update)
        self.topo_time_2 = QLineEdit()
        self.topo_time_2.setPlaceholderText("0.2")
        self.topo_time_2.setFixedWidth(70)
        self.topo_time_2.textChanged.connect(self.mark_needs_update)
        self.topo_time_3 = QLineEdit()
        self.topo_time_3.setPlaceholderText("0.3")
        self.topo_time_3.setFixedWidth(70)
        self.topo_time_3.textChanged.connect(self.mark_needs_update)
        topo_row.addWidget(self.topo_time_1)
        topo_row.addWidget(self.topo_time_2)
        topo_row.addWidget(self.topo_time_3)
        topo_row.addStretch(1)
        topo_layout.addLayout(topo_row)
        layout.addWidget(self.topo_times_container)
        self.topo_times_container.setVisible(False)

        # Topomap mode selector (Static / Animated) — visible for Topographic Map only
        self.topo_mode_container = QWidget()
        topo_mode_layout = QVBoxLayout(self.topo_mode_container)
        topo_mode_layout.setContentsMargins(0, 0, 0, 0)
        topo_mode_layout.setSpacing(6)
        self.topo_mode_label = QLabel("Topomap Mode")
        self.topo_mode_label.setStyleSheet("color: #202124; font-size: 12px;")
        self.topo_mode_combo = QComboBox()
        self.topo_mode_combo.addItems(["Static", "Animated"])
        self.topo_mode_combo.currentTextChanged.connect(self._on_topo_mode_changed)
        self.topo_mode_combo.currentTextChanged.connect(self.mark_needs_update)
        topo_mode_layout.addWidget(self.topo_mode_label)
        topo_mode_layout.addWidget(self.topo_mode_combo)
        layout.addWidget(self.topo_mode_container)
        self.topo_mode_container.setVisible(False)

        # Animation controls — visible when Topomap Mode is Animated
        self.anim_controls_container = QWidget()
        anim_layout = QVBoxLayout(self.anim_controls_container)
        anim_layout.setContentsMargins(0, 0, 0, 0)
        anim_layout.setSpacing(8)

        anim_btn_row = QHBoxLayout()
        anim_btn_row.setSpacing(8)
        self.anim_play_btn = QPushButton("▶  Play")
        self.anim_play_btn.setCursor(Qt.PointingHandCursor)
        self.anim_play_btn.setFixedHeight(30)
        self.anim_play_btn.clicked.connect(self._toggle_animation)
        anim_btn_row.addWidget(self.anim_play_btn)

        anim_speed_label = QLabel("Speed:")
        anim_speed_label.setStyleSheet("color: #202124; font-size: 12px;")
        self.anim_speed_combo = QComboBox()
        self.anim_speed_combo.addItems(["0.5x", "1x", "2x", "4x"])
        self.anim_speed_combo.setCurrentIndex(1)
        self.anim_speed_combo.setFixedWidth(60)
        anim_btn_row.addWidget(anim_speed_label)
        anim_btn_row.addWidget(self.anim_speed_combo)
        anim_btn_row.addStretch(1)
        anim_layout.addLayout(anim_btn_row)

        slider_row = QHBoxLayout()
        slider_row.setSpacing(8)
        self.anim_slider = QSlider(Qt.Horizontal)
        self.anim_slider.setRange(0, 100)
        self.anim_slider.setValue(0)
        self.anim_slider.valueChanged.connect(self._on_anim_slider_changed)
        slider_row.addWidget(self.anim_slider, stretch=1)

        self.anim_time_label = QLabel("0.0 ms")
        self.anim_time_label.setStyleSheet("color: #202124; font-size: 12px;")
        self.anim_time_label.setFixedWidth(80)
        self.anim_time_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        slider_row.addWidget(self.anim_time_label)
        anim_layout.addLayout(slider_row)

        layout.addWidget(self.anim_controls_container)
        self.anim_controls_container.setVisible(False)

        # Events checkbox
        self.events_checkbox_container = QWidget()
        ev_layout = QVBoxLayout(self.events_checkbox_container)
        ev_layout.setContentsMargins(0, 0, 0, 0)
        ev_layout.setSpacing(0)
        self.events_checkbox = QCheckBox("Display Events and Responses")
        self.events_checkbox.setStyleSheet("font-size: 12px; color: #202124;")
        self.events_checkbox.stateChanged.connect(self._on_events_checkbox_changed)
        self.events_checkbox.stateChanged.connect(self.mark_needs_update)
        ev_layout.addWidget(self.events_checkbox)
        layout.addWidget(self.events_checkbox_container)

        layout.addStretch(1)

        # Visualize button (lives inside the tab so each tab has its own)
        self.visualize_btn = QPushButton("Visualize")
        self.visualize_btn.setCursor(Qt.PointingHandCursor)
        self.visualize_btn.setFixedHeight(44)
        self.visualize_btn.clicked.connect(self.visualize)

        self._run_shortcut = QShortcut(QKeySequence(Qt.Key_Return), self)
        self._run_shortcut.activated.connect(self.visualize_btn.click)

        layout.addWidget(self.visualize_btn)

        self.reset_visualize_button()
        return box

    # Lazy loading -> once visualize is clicked
    def ensure_loaded(self) -> bool:
        """
        Load the .set file if not already loaded.
        Returns True on success, False on failure.
        """
        if self.current_epochs is not None:
            return True
        # Show a simple modal loading indicator so the user
        # gets feedback while large files are being read.
        app = QApplication.instance()
        progress = QProgressDialog("Loading EEG data…", "", 0, 0, self)
        progress.setWindowTitle("Loading")
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setCancelButton(None)
        progress.setMinimumDuration(0)
        progress.setAutoClose(True)
        progress.show()
        if app is not None:
            app.setOverrideCursor(Qt.WaitCursor)
            app.processEvents()
        try:
            logger.debug(f"[Tab] Loading {self.filepath} ...")
            self.current_epochs = read_epochs_eeglab_minimal(self.filepath)
            self._all_sensors = ["All Channels"] + list(self.current_epochs.ch_names)
            self.sensor_combo.set_items(self._all_sensors)
            logger.debug(f"[Tab] Loaded {len(self.current_epochs.ch_names)} channels")
            return True
        except Exception as e:
            QMessageBox.critical(
                self,
                "Load Error",
                f"Could not load file:\n{os.path.basename(self.filepath)}\n\n{e}",
            )
            return False
        finally:
            progress.close()
            if app is not None:
                app.restoreOverrideCursor()

    # Visualize
    def visualize(self):
        if not self.ensure_loaded():
            return

        graph_type = self.graph_type_combo.currentText()

        if graph_type in ("Topographic Map", "Joint Maps"):
                n_channels = len(self.current_epochs.ch_names)
                if n_channels < 19:
                    QMessageBox.warning(
                        self, "Insufficient Channels",
                        f"Topographic maps require at least 19 channels for reliable spatial interpolation.\n\n"
                        f"Your data has {n_channels} channels.\n\n"
                        f"Please use 'ErrP Time Series' visualization instead.",
                        QMessageBox.Ok
                    )
                    return

        opts = {
            "epoch_start": self.epoch_start.text().strip(),
            "epoch_end":   self.epoch_end.text().strip(),
            "sensors":     self.sensor_combo.get_selected(),
            "graph_type":  graph_type,
            "display_events_responses": self.events_checkbox.isChecked(),
        }

        try:
            epochs = self.current_epochs

            # Apply time window (skip for Topographic Map — needs full range)
            if graph_type != "Topographic Map" and (opts["epoch_start"] or opts["epoch_end"]):
                try:
                    tmin = self.current_epochs.tmin
                    tmax = self.current_epochs.tmax
                    if opts["epoch_start"]:
                        tmin = float(opts["epoch_start"]) / 1000
                    if opts["epoch_end"]:
                        tmax = float(opts["epoch_end"]) / 1000
                    logger.debug(f"[Tab] Time window: {tmin:.3f}–{tmax:.3f} s")
                    epochs = select_time_window(epochs, tmin, tmax)
                except ValueError:
                    logger.warning("[Tab] Invalid epoch times, using full range")

            # Channel picks (topo/joint always need all channels)
            channel_picks = None
            selected_sensors = opts["sensors"]
            needs_all = graph_type in ("Topographic Map", "Joint Maps")

            if not needs_all and selected_sensors and selected_sensors != ["All Channels"]:
                # Build picks from selected sensors
                channel_picks = []
                for sensor_name in selected_sensors:
                    if sensor_name in epochs.ch_names:
                        channel_picks.append(epochs.ch_names.index(sensor_name))
                if channel_picks:
                    logger.debug(f"[Tab] Channels: {len(channel_picks)} selected")
                else:
                    channel_picks = None

            logger.debug("[Tab] Averaging epochs...")
            evoked = average_epochs(epochs, picks=channel_picks)

            theme = "dark" if self._is_dark_mode else "light"

            # Animated topomap — set up the figure and slider, then return
            if (graph_type == "Topographic Map"
                    and self.topo_mode_combo.currentText() == "Animated"):
                self._setup_animated_topomap(evoked, theme)
                self.reset_visualize_button()
                return

            if graph_type == "ErrP Time Series":
                fig = plot_evoked(
                    evoked,
                    window_title="ErrP Time Series",
                    display_events_responses=opts["display_events_responses"],
                    show=False, theme=theme,
                    selected_sensors=selected_sensors,
                )
            elif graph_type == "Topographic Map":
                fig = plot_topomap(
                    evoked, times=self._parse_topomap_times(),
                    show=False, theme=theme,
                    selected_sensors=selected_sensors,
                )
            elif graph_type == "Joint Maps":
                fig = plot_joint(
                    evoked,
                    times=self._parse_topomap_times(),
                    title="ErrP Analysis",
                    display_events_responses=opts["display_events_responses"],
                    show=False, theme=theme,
                    selected_sensors=selected_sensors,
                )
            else:
                fig = plot_evoked(evoked, show=False, theme=theme, selected_sensors=selected_sensors)

            self._replace_canvas(fig)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Visualization failed:\n{e}")
            import traceback; traceback.print_exc()

        self.reset_visualize_button()

    def _replace_canvas(self, new_fig: Figure):
        layout = self.graph_frame.layout()
        layout.removeWidget(self.canvas)
        self.canvas.setParent(None)
        self.canvas.deleteLater()

        self.figure = new_fig
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, stretch=1)
        self.canvas.draw_idle()

    #placeholder before visualize clicked
    def _draw_placeholder(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, "Load data and click Visualize",
                ha="center", va="center", fontsize=16,
                color="#9aa0a6" if self._is_dark_mode else "#5f6368")
        ax.axis("off")

    # handle changing graph types
    def _on_graph_type_changed(self, graph_type: str):
        supports_events = graph_type in ("ErrP Time Series", "Joint Maps")
        self.events_checkbox_container.setVisible(supports_events)

        supports_topo = graph_type in ("Topographic Map", "Joint Maps")
        is_topo_only = graph_type == "Topographic Map"

        # Topomap mode selector only for Topographic Map (not Joint Maps)
        self.topo_mode_container.setVisible(is_topo_only)

        if is_topo_only:
            is_animated = self.topo_mode_combo.currentText() == "Animated"
            self.topo_times_container.setVisible(not is_animated)
            self.anim_controls_container.setVisible(is_animated)
            if not is_animated:
                self._stop_animation()
        else:
            self._stop_animation()
            self.anim_controls_container.setVisible(False)
            self.topo_times_container.setVisible(supports_topo)

        # Sensor dropdown for Time Series and Joint Maps
        show_sensor_selection = graph_type in ("ErrP Time Series", "Joint Maps")
        self.sensor_container.setVisible(show_sensor_selection)

        # Epoch window only for Time Series
        self.epoch_container.setVisible(not is_topo_only)

        if supports_events:
            self.events_checkbox.blockSignals(True)
            self.events_checkbox.setChecked(self.events_checkbox_checked)
            self.events_checkbox.blockSignals(False)

        if is_topo_only:
            self.epoch_start.blockSignals(True)
            self.epoch_end.blockSignals(True)
            self.epoch_start.clear()
            self.epoch_end.clear()
            self.epoch_start.setPlaceholderText("Full range")
            self.epoch_end.setPlaceholderText("Full range")
            self.epoch_start.blockSignals(False)
            self.epoch_end.blockSignals(False)
            self._set_epoch_field_style(disabled=True)
        else:
            self.epoch_start.setPlaceholderText("Start")
            self.epoch_end.setPlaceholderText("End")
            self._set_epoch_field_style(disabled=False)

        # switch to all channels if needed
        if graph_type == "Topographic Map":
            # Topomap only shows "All Channels" option
            if self._last_graph_type == "ErrP Time Series":
                self._last_time_series_selection = self.sensor_combo.get_selected()
            self.sensor_combo.set_items(["All Channels"])
        elif graph_type == "Joint Maps":
            # Joint Maps shows all sensors for time series control
            if self._last_graph_type == "ErrP Time Series":
                self._last_time_series_selection = self.sensor_combo.get_selected()
            self.sensor_combo.set_items(self._all_sensors if self._all_sensors else ["All Channels"])
            if self._last_time_series_selection and any(s in self._all_sensors for s in self._last_time_series_selection):
                # Restore previous selections for Joint Maps
                for sensor in self._last_time_series_selection:
                    if sensor in self._all_sensors:
                        item = self.sensor_combo.list_widget.findItems(sensor, Qt.MatchExactly)[0]
                        MultiSelectItemDelegate.update_item_style(item, True)
                self.sensor_combo.selected = set(self._last_time_series_selection)
                self.sensor_combo._update_button_text()
            else:
                # Default to "All Channels"
                item = self.sensor_combo.list_widget.findItems("All Channels", Qt.MatchExactly)[0]
                MultiSelectItemDelegate.update_item_style(item, True)
                self.sensor_combo.selected = {"All Channels"}
                self.sensor_combo._update_button_text()
        else:
            # Time Series mode
            self.sensor_combo.set_items(self._all_sensors if self._all_sensors else ["All Channels"])
            if self._last_time_series_selection and any(s in self._all_sensors for s in self._last_time_series_selection):
                # Restore previous selections for Time Series
                for sensor in self._last_time_series_selection:
                    if sensor in self._all_sensors:
                        item = self.sensor_combo.list_widget.findItems(sensor, Qt.MatchExactly)[0]
                        MultiSelectItemDelegate.update_item_style(item, True)
                self.sensor_combo.selected = set(self._last_time_series_selection)
                self.sensor_combo._update_button_text()
            else:
                # Default to "All Channels"
                item = self.sensor_combo.list_widget.findItems("All Channels", Qt.MatchExactly)[0]
                MultiSelectItemDelegate.update_item_style(item, True)
                self.sensor_combo.selected = {"All Channels"}
                self.sensor_combo._update_button_text()

        self._last_graph_type = graph_type

    def _set_epoch_field_style(self, disabled: bool):
        if disabled:
            if self._is_dark_mode:
                s = "QLineEdit { background: #2d2d2d; color: #5f6368; border: 1px solid #3c4043; border-radius: 4px; }"
                lc = "color: #5f6368; font-size: 12px;"
            else:
                s = "QLineEdit { background: #f1f3f4; color: #9aa0a6; border: 1px solid #dadce0; border-radius: 4px; }"
                lc = "color: #9aa0a6; font-size: 12px;"
        else:
            if self._is_dark_mode:
                s = "QLineEdit { background: #202124; color: #e8eaed; border: 1px solid #5f6368; border-radius: 4px; }"
                lc = "color: #e8eaed; font-size: 12px;"
            else:
                s = "QLineEdit { background: #ffffff; color: #202124; border: 1px solid #dadce0; border-radius: 4px; }"
                lc = "color: #202124; font-size: 12px;"
        self.epoch_start.setStyleSheet(s)
        self.epoch_end.setStyleSheet(s)
        self.epoch_label.setStyleSheet(lc)

    def _on_events_checkbox_changed(self, _state):
        self.events_checkbox_checked = self.events_checkbox.isChecked()

    # ---- Animated topomap helpers ----

    def _on_topo_mode_changed(self, mode: str):
        is_animated = mode == "Animated"
        self.topo_times_container.setVisible(not is_animated)
        self.anim_controls_container.setVisible(is_animated)
        if not is_animated:
            self._stop_animation()
        self.mark_needs_update()

    def _setup_animated_topomap(self, evoked, theme):
        self._stop_animation()
        self._anim_evoked = evoked
        self._anim_theme = theme
        self._anim_global_vmax = float(np.abs(evoked.data).max() * 1e6)

        fig = plot_topomap_frame(
            evoked, time=evoked.times[0],
            theme=theme, global_vmax=self._anim_global_vmax,
        )
        self._replace_canvas(fig)

        n_times = len(evoked.times)
        self.anim_slider.blockSignals(True)
        self.anim_slider.setRange(0, n_times - 1)
        self.anim_slider.setValue(0)
        self.anim_slider.blockSignals(False)
        self._update_anim_time_label(0)

    def _on_anim_slider_changed(self, value):
        if self._anim_evoked is None:
            return
        time = self._anim_evoked.times[value]
        self._update_anim_time_label(value)
        plot_topomap_frame(
            self._anim_evoked, time=time,
            fig=self.figure, theme=self._anim_theme,
            global_vmax=self._anim_global_vmax,
        )
        self.canvas.draw_idle()

    def _update_anim_time_label(self, slider_value):
        if self._anim_evoked is not None and slider_value < len(self._anim_evoked.times):
            time_ms = self._anim_evoked.times[slider_value] * 1000
            self.anim_time_label.setText(f"{time_ms:.1f} ms")

    def _toggle_animation(self):
        if self._anim_evoked is None:
            return
        if self._anim_playing:
            self._pause_animation()
        else:
            self._start_animation()

    def _start_animation(self):
        if self._anim_evoked is None:
            return
        self._anim_playing = True
        self.anim_play_btn.setText("⏸  Pause")
        self._anim_timer.start(50)

    def _pause_animation(self):
        self._anim_playing = False
        self.anim_play_btn.setText("▶  Play")
        self._anim_timer.stop()

    def _stop_animation(self):
        self._pause_animation()
        self._anim_evoked = None

    def _on_anim_tick(self):
        if self._anim_evoked is None:
            self._pause_animation()
            return

        speed_text = self.anim_speed_combo.currentText()
        speed = float(speed_text.replace("x", ""))

        sfreq = self._anim_evoked.sfreq
        step = max(1, round(sfreq * 0.025 * speed))

        current = self.anim_slider.value()
        new_val = current + step
        if new_val > self.anim_slider.maximum():
            new_val = 0
        self.anim_slider.setValue(new_val)

    # time parser for topo maps
    def _parse_topomap_times(self) -> List[float]:
        defaults = [0.1, 0.2, 0.3]
        result = []
        for i, w in enumerate([self.topo_time_1, self.topo_time_2, self.topo_time_3]):
            t = w.text().strip()
            if not t:
                result.append(defaults[i])
                continue
            try:
                result.append(float(t))
            except ValueError:
                return defaults
        return result if result else defaults

    # handle visualize btn state
    def mark_needs_update(self):
        if self._is_dark_mode:
            self.visualize_btn.setStyleSheet(
                "QPushButton { background: #8ab4f8; border: 1px solid #8ab4f8; border-radius: 4px;"
                " font-size: 14px; color: #000000; }"
                "QPushButton:hover { background: #669df6; }"
                "QPushButton:pressed { background: #4a8af5; }"
            )
        else:
            self.visualize_btn.setStyleSheet(
                "QPushButton { background: #1a73e8; border: 1px solid #1a73e8; border-radius: 4px;"
                " font-size: 14px; color: white; }"
                "QPushButton:hover { background: #1666c1; }"
                "QPushButton:pressed { background: #1450b1; }"
            )

    def reset_visualize_button(self):
        if self._is_dark_mode:
            self.visualize_btn.setStyleSheet(
                "QPushButton { background: #202124; border: 1px solid #5f6368; border-radius: 4px;"
                " font-size: 14px; color: #e8eaed; }"
                "QPushButton:hover { background: #303134; }"
                "QPushButton:pressed { background: #3c4043; }"
            )
        else:
            self.visualize_btn.setStyleSheet(
                "QPushButton { background: #ffffff; border: 1px solid #202124; border-radius: 4px;"
                " font-size: 14px; color: #202124; }"
                "QPushButton:hover { background: #f6f8fe; }"
                "QPushButton:pressed { background: #e8f0fe; }"
            )

    # Global theme
    def apply_theme(self, is_dark: bool):
        """Called by FileWindow when the global darkmode toggle changes."""
        self._is_dark_mode = is_dark

        # Graph frame border
        if is_dark:
            self.graph_frame.setStyleSheet(
                "QFrame { background: #121212; border: 1px solid #3c4043; border-radius: 4px; }"
            )
        else:
            self.graph_frame.setStyleSheet(
                "QFrame { background: #ffffff; border: 1px solid #dadce0; border-radius: 4px; }"
            )

        # Options box title
        if is_dark:
            self.options_box.setStyleSheet(
                "QGroupBox { font-size: 13px; font-weight: 600; color: #e8eaed; }"
                "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #e8eaed; }"
            )
        else:
            self.options_box.setStyleSheet(
                "QGroupBox { font-size: 13px; font-weight: 600; color: #202124; }"
                "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #202124; }"
            )

        # All QLineEdits inside this tab
        le_style = (
            "QLineEdit { background: #202124; color: #e8eaed; border: 1px solid #5f6368; border-radius: 4px; }"
            if is_dark else
            "QLineEdit { background: #ffffff; color: #202124; border: 1px solid #dadce0; border-radius: 4px; }"
        )
        for le in self.findChildren(QLineEdit):
            le.setStyleSheet(le_style)

        # All QComboBoxes inside this tab
        cb_style = (
            "QComboBox { background: #202124; color: #e8eaed; border: 1px solid #5f6368; border-radius: 4px; }"
            "QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: top right;"
            " width: 18px; border-left: 1px solid #5f6368; }"
            if is_dark else
            "QComboBox { background: #ffffff; color: #202124; border: 1px solid #dadce0; border-radius: 4px; }"
            "QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: top right;"
            " width: 18px; border-left: 1px solid #dadce0; }"
        )
        for cb in self.findChildren(QComboBox):
            cb.setStyleSheet(cb_style)

        # All QLabels inside this tab
        for lbl in self.findChildren(QLabel):
            s = lbl.styleSheet() or ""
            if is_dark:
                s = s.replace("#202124", "#e8eaed").replace("#5f6368", "#9aa0a6")
            else:
                s = s.replace("#e8eaed", "#202124").replace("#9aa0a6", "#5f6368")
            lbl.setStyleSheet(s)

        # Events checkbox
        if is_dark:
            self.events_checkbox.setStyleSheet(
                "QCheckBox { font-size: 12px; color: #e8eaed; }"
                "QCheckBox::indicator { width: 16px; height: 16px; }"
                "QCheckBox::indicator:unchecked { border-radius: 3px; border: 1px solid #9aa0a6; background: #202124; }"
                "QCheckBox::indicator:checked { border-radius: 3px; border: 1px solid #8ab4f8; background: #8ab4f8; }"
            )
        else:
            self.events_checkbox.setStyleSheet(
                "QCheckBox { font-size: 12px; color: #202124; }"
                "QCheckBox::indicator { width: 16px; height: 16px; }"
                "QCheckBox::indicator:unchecked { border-radius: 3px; border: 1px solid #dadce0; background: #ffffff; }"
                "QCheckBox::indicator:checked { border-radius: 3px; border: 1px solid #1a73e8; background: #1a73e8; }"
            )

        # Animation play button + slider
        if is_dark:
            self.anim_play_btn.setStyleSheet(
                "QPushButton { background: #303134; color: #e8eaed; border: 1px solid #5f6368;"
                " border-radius: 4px; padding: 4px 10px; font-size: 12px; }"
                "QPushButton:hover { background: #3c4043; }"
                "QPushButton:pressed { background: #4a4e51; }"
            )
            self.anim_slider.setStyleSheet(
                "QSlider::groove:horizontal { background: #3c4043; height: 6px; border-radius: 3px; }"
                "QSlider::handle:horizontal { background: #8ab4f8; width: 14px; margin: -4px 0;"
                " border-radius: 7px; }"
                "QSlider::sub-page:horizontal { background: #8ab4f8; border-radius: 3px; }"
            )
        else:
            self.anim_play_btn.setStyleSheet(
                "QPushButton { background: #ffffff; color: #202124; border: 1px solid #dadce0;"
                " border-radius: 4px; padding: 4px 10px; font-size: 12px; }"
                "QPushButton:hover { background: #f1f3f4; }"
                "QPushButton:pressed { background: #e8eaed; }"
            )
            self.anim_slider.setStyleSheet(
                "QSlider::groove:horizontal { background: #dadce0; height: 6px; border-radius: 3px; }"
                "QSlider::handle:horizontal { background: #1a73e8; width: 14px; margin: -4px 0;"
                " border-radius: 7px; }"
                "QSlider::sub-page:horizontal { background: #1a73e8; border-radius: 3px; }"
            )

        # Keep animation theme in sync so slider redraws use the right colours
        self._anim_theme = "dark" if is_dark else "light"

        # Re-apply epoch field disabled style if currently on Topographic Map
        self._on_graph_type_changed(self.graph_type_combo.currentText())

        # Retheme the live matplotlib figure
        if self._anim_evoked is not None:
            self._on_anim_slider_changed(self.anim_slider.value())
        else:
            self._apply_mpl_theme(self.figure)
            self.canvas.draw_idle()

        # Update button appearance
        self.reset_visualize_button()

    def _apply_mpl_theme(self, fig: Figure):
        if fig is None:
            return
        if self._is_dark_mode:
            bg, axis_bg, text, grid = "#121212", "#121212", "#e8eaed", "#3c4043"
        else:
            bg, axis_bg, text, grid = "#ffffff", "#ffffff", "#202124", "#dadce0"

        fig.patch.set_facecolor(bg)
        if hasattr(fig, "_suptitle") and fig._suptitle is not None:
            fig._suptitle.set_color(text)
        for ax in fig.get_axes():
            ax.set_facecolor(axis_bg)
            ax.tick_params(colors=text)
            ax.xaxis.label.set_color(text)
            ax.yaxis.label.set_color(text)
            ax.title.set_color(text)
            ax.grid(color=grid, alpha=0.3)
            for spine in ax.spines.values():
                spine.set_color(grid)

# Owns FileTabs and bottom bar
class FileWindow(QMainWindow):
    def __init__(self, file_path: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("ErrP Visualizer")
        self.resize(1280, 760)

        self.is_dark_mode = False
        # Track absolute paths already open so we dont duplicate tabs
        self._open_paths: List[str] = []

        central = QWidget()
        self.setCentralWidget(central)

        self.outer = QVBoxLayout(central)
        self.outer.setContentsMargins(18, 18, 18, 18)
        self.outer.setSpacing(14)

        self._build_top_bar()   # darkmode toggle lives here
        self._build_tab_area()  # QTabWidget
        self._build_bottom_bar()  # drag/drop, browse, clear all

        self._apply_window_light_styles()

        if file_path:
            self.add_files([file_path])

    def _build_top_bar(self):
        """Top bar: title | [Record EEG] [? Help] | Dark mode"""
        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        bar.setSpacing(14)

        # Title
        title_lbl = QLabel("ErrP Visualizer")
        title_lbl.setStyleSheet("font-size: 15px; font-weight: 700; color: #202124;")
        self.title_lbl = title_lbl
        bar.addWidget(title_lbl)
        bar.addStretch(1)

        # Record EEG button — opens live BCI web app in browser
        self.record_eeg_btn = QPushButton("⬤  Record EEG")
        self.record_eeg_btn.setCursor(Qt.PointingHandCursor)
        self.record_eeg_btn.setToolTip("Open the live EEG recording session in your browser")
        self.record_eeg_btn.clicked.connect(self._open_live_recording)
        self._style_record_eeg_btn(dark=False)
        bar.addWidget(self.record_eeg_btn)

        # Help button
        self.help_btn = QPushButton("?  Help")
        self.help_btn.setCursor(Qt.PointingHandCursor)
        self.help_btn.setToolTip("How to use ErrP Visualizer")
        self.help_btn.clicked.connect(self._open_help)
        self._style_help_btn(dark=False)
        bar.addWidget(self.help_btn)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setFrameShadow(QFrame.Sunken)
        self.top_sep = sep
        bar.addWidget(sep)

        # Dark mode toggle
        dark_row = QHBoxLayout()
        dark_row.setSpacing(6)
        dark_lbl = QLabel("Dark mode")
        dark_lbl.setStyleSheet("font-size: 13px; color: #202124;")
        self.dark_lbl = dark_lbl
        self.dark_mode_toggle = ToggleSwitch("")
        self.dark_mode_toggle.set_dark_mode(False)
        self.dark_mode_toggle.stateChanged.connect(self._on_dark_mode_toggled)
        dark_row.addWidget(dark_lbl)
        dark_row.addWidget(self.dark_mode_toggle)
        bar.addLayout(dark_row)

        self.outer.addLayout(bar)

    def _open_help(self):
        dlg = HelpDialog(is_dark=self.is_dark_mode, parent=self)
        dlg.exec_()

    def _open_live_recording(self):
        QDesktopServices.openUrl(QUrl(LIVE_RECORDING_URL))

    def _style_help_btn(self, dark: bool):
        if dark:
            self.help_btn.setStyleSheet(
                "QPushButton { background: #303134; color: #e8eaed; border: 1px solid #5f6368;"
                " border-radius: 4px; padding: 4px 12px; font-size: 13px; }"
                "QPushButton:hover { background: #3c4043; }"
                "QPushButton:pressed { background: #4a4e51; }"
            )
        else:
            self.help_btn.setStyleSheet(
                "QPushButton { background: #ffffff; color: #202124; border: 1px solid #dadce0;"
                " border-radius: 4px; padding: 4px 12px; font-size: 13px; }"
                "QPushButton:hover { background: #f1f3f4; }"
                "QPushButton:pressed { background: #e8eaed; }"
            )

    def _style_record_eeg_btn(self, dark: bool):
        if dark:
            self.record_eeg_btn.setStyleSheet(
                "QPushButton { background: #2d2d2d; color: #f28b82; border: 1px solid #f28b82;"
                " border-radius: 4px; padding: 4px 12px; font-size: 13px; }"
                "QPushButton:hover { background: #3c4043; }"
                "QPushButton:pressed { background: #4a4e51; }"
            )
        else:
            self.record_eeg_btn.setStyleSheet(
                "QPushButton { background: #ffffff; color: #c5221f; border: 1px solid #f28b82;"
                " border-radius: 4px; padding: 4px 12px; font-size: 13px; }"
                "QPushButton:hover { background: #fce8e6; }"
                "QPushButton:pressed { background: #fad2cf; }"
            )

    def _build_tab_area(self):
        """QTabWidget that holds one FileTab per loaded file."""
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.setMovable(True)
        self.tab_widget.tabCloseRequested.connect(self._close_tab)
        self.tab_widget.setStyleSheet(self._tab_widget_style(dark=False))

        # Show a placeholder when no tabs are open
        self._empty_label = QLabel("Drop files below or use Browse to get started")
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._empty_label.setStyleSheet("color: #9aa0a6; font-size: 14px;")

        # Stack: either the tab widget or the empty label
        self._tab_stack = QVBoxLayout()
        self._tab_stack.addWidget(self._empty_label)
        self._tab_stack.addWidget(self.tab_widget)
        self.tab_widget.setVisible(False)

        self.outer.addLayout(self._tab_stack, stretch=1)

    def _build_bottom_bar(self):
        """Shared drag/drop zone + browse + clear all."""
        bar = QHBoxLayout()
        bar.setSpacing(18)

        # Left spacer so the drop/browse area can be centered independently
        bar.addStretch(1)

        # Drop + browse frame
        drop_frame = QFrame()
        drop_frame.setFrameShape(QFrame.StyledPanel)
        self.drop_browse_frame = drop_frame
        df_layout = QGridLayout(drop_frame)
        df_layout.setContentsMargins(14, 14, 14, 14)
        df_layout.setHorizontalSpacing(14)
        df_layout.setVerticalSpacing(8)

        # Download Graph button

        BTN_W, BTN_H = 110, 30

        # Download Graph column
        download_col = QVBoxLayout()
        download_col.setSpacing(4)
        download_lbl = QLabel("Download")
        download_lbl.setStyleSheet("font-size: 13px; color: #202124;")
        download_lbl.setAlignment(Qt.AlignHCenter)
        download_lbl.setFixedWidth(BTN_W)
        self.download_lbl = download_lbl
        self.download_btn = QPushButton("↓ Save")
        self.download_btn.setCursor(Qt.PointingHandCursor)
        self.download_btn.setFixedSize(BTN_W, BTN_H)
        self.download_btn.clicked.connect(self._download_graph)
        download_col.addStretch(1)
        download_col.addWidget(download_lbl, alignment=Qt.AlignHCenter)
        download_col.addWidget(self.download_btn, alignment=Qt.AlignHCenter)
        download_col.addStretch(1)
        df_layout.addLayout(download_col, 0, 0, 1, 1)

        # Main drop zone occupies the center columns
        self.drop_zone = FileDropFrame()
        self.drop_zone.filesDropped.connect(self.add_files)
        df_layout.addWidget(self.drop_zone, 0, 1, 1, 3)

        # Browse / Clear All column
        side_col = QVBoxLayout()
        side_col.setSpacing(4)

        browse_lbl = QLabel("Browse")
        browse_lbl.setStyleSheet("font-size: 13px; color: #202124;")
        browse_lbl.setAlignment(Qt.AlignHCenter)
        browse_lbl.setFixedWidth(BTN_W)
        self.browse_lbl = browse_lbl
        self.browse_btn = QPushButton("…")
        self.browse_btn.setCursor(Qt.PointingHandCursor)
        self.browse_btn.setFixedSize(BTN_W, BTN_H)
        self.browse_btn.clicked.connect(self._browse_files)

        clear_lbl = QLabel("Clear All")
        clear_lbl.setStyleSheet("font-size: 13px; color: #202124;")
        clear_lbl.setAlignment(Qt.AlignHCenter)
        clear_lbl.setFixedWidth(BTN_W)
        self.clear_lbl = clear_lbl

        self.clear_btn = QPushButton("✕")
        self.clear_btn.setCursor(Qt.PointingHandCursor)
        self.clear_btn.setFixedSize(BTN_W, BTN_H)
        self.clear_btn.clicked.connect(self._clear_all_tabs)
        self._style_clear_btn(dark=False)

        side_col.addStretch(1)
        side_col.addWidget(browse_lbl, alignment=Qt.AlignHCenter)
        side_col.addWidget(self.browse_btn, alignment=Qt.AlignHCenter)
        side_col.addSpacing(10)
        side_col.addWidget(clear_lbl, alignment=Qt.AlignHCenter)
        side_col.addWidget(self.clear_btn, alignment=Qt.AlignHCenter)
        side_col.addStretch(1)

        df_layout.addLayout(side_col, 0, 4, 1, 1)

        self.files_label = QLabel("No files loaded")
        self.files_label.setStyleSheet("color: #5f6368; font-size: 11px;")
        self.files_label.setWordWrap(True)
        df_layout.addWidget(self.files_label, 1, 0, 1, 5)

        bar.addWidget(drop_frame, stretch=2)
        bar.addStretch(1)

        self.outer.addLayout(bar)


    def _browse_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select .set or .csv file(s)",
            "",
            "EEG Files (*.set *.csv);;All Files (*.*)"
        )
        if paths:
            self.add_files(paths)

    def _download_graph(self):
        """Save the currently displayed graph as PNG."""
        current_tab = self.tab_widget.currentWidget()
        if not current_tab or not hasattr(current_tab, 'figure'):
            QMessageBox.warning(self, "No Graph", "No graph is currently displayed.")
            return

        # Get save location from user
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Graph As",
            "",
            "PNG Files (*.png);;All Files (*.*)"
        )

        if filename:
            try:
                # Ensure the filename has .png extension
                if not filename.lower().endswith('.png'):
                    filename += '.png'

                # Save the figure
                current_tab.figure.savefig(filename, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "Success", f"Graph saved as {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save graph:\n{str(e)}")


    def add_files(self, paths: List[str]):
        """
        Add one tab per new file, loading is lazy.
        Validates files before loading, then automatically converts .csv files to .set format.
        """
        added = []
        for p in paths:
            ap = os.path.abspath(p)

            # Validate file first
            try:
                file_type, validation_info = FileValidator.validate_file(ap)
                logger.debug(f"File validation passed for: {os.path.basename(ap)}")
            except FileValidationError as e:
                QMessageBox.critical(
                    self, "File Validation Error",
                    f"Invalid file:\n{os.path.basename(p)}\n\n{str(e)}"
                )
                continue  # Skip this file

            # AUTO-CONVERT CSV TO .SET
            if ap.lower().endswith('.csv'):
                try:
                    logger.debug(f"Detected CSV file, converting to .set format...")
                    ap = self.convert_ganglion_csv_to_set(ap)
                    logger.debug(f"Converted to: {ap}")
                except Exception as e:
                    QMessageBox.critical(
                        self, "CSV Conversion Error",
                        f"Could not convert CSV file:\n{os.path.basename(p)}\n\n{str(e)}"
                    )
                    continue  # Skip this file

            # Check if already open
            if ap in self._open_paths:
                continue

            self._open_paths.append(ap)

            tab = FileTab(filepath=ap, is_dark_mode=self.is_dark_mode, parent=self)
            label = os.path.basename(ap)
            self.tab_widget.addTab(tab, label)
            added.append(label)

        if added:
            first_new_idx = self.tab_widget.count() - len(added)
            self.tab_widget.setCurrentIndex(first_new_idx)
            self._update_empty_state()
            self._update_files_label()

    def convert_ganglion_csv_to_set(self, csv_path: str) -> str:
        """
        Convert Ganglion CSV to EEGLAB .set format.
        Creates continuous data format (2D: channels × timepoints).
        Silently handles the conversion - user doesn't need to know.
        Returns: Path to the converted .set file
        """
        import pandas as pd
        from scipy.io import savemat
        import numpy as np

        # Read CSV, skipping header comments
        df = pd.read_csv(csv_path, comment='%', header=None, skipinitialspace=True)

        # Extract 4 EEG channels (columns 1-4 in OpenBCI format)
        # Shape: (4, n_samples)
        data = df.iloc[:, 1:5].values.T

        data = data / 1e6 # uV -> V

        # Ganglion specs
        n_channels = 4
        sfreq = 200
        n_samples = data.shape[1]

        # KEEP AS 2D for continuous data (channels, timepoints)
        # DO NOT add epoch dimension - let data_loader handle it
        data_continuous = data.astype(np.float32)

        # Channel locations
        ch_locs = [
            {'labels': 'TP9',  'X': -0.87, 'Y': -0.31, 'Z': 0.0, 'theta': -110.0, 'radius': 0.9},
            {'labels': 'AF7',  'X': -0.6,  'Y': 0.87,  'Z': 0.0, 'theta': -55.0,  'radius': 0.9},
            {'labels': 'AF8',  'X': 0.6,   'Y': 0.87,  'Z': 0.0, 'theta': 55.0,   'radius': 0.9},
            {'labels': 'TP10', 'X': 0.87,  'Y': -0.31, 'Z': 0.0, 'theta': 110.0,  'radius': 0.9},
        ]

        # Create EEGLAB structure for CONTINUOUS data
        EEG = {
            'data': data_continuous,  # 2D: (channels, timepoints)
            'setname': 'Ganglion_Recording',
            'nbchan': n_channels,
            'pnts': n_samples,      # Total number of timepoints
            'trials': 1,            # Indicates continuous data
            'srate': sfreq,
            'xmin': 0.0,
            'xmax': n_samples / sfreq,
            'times': (np.arange(n_samples) / sfreq).tolist(),  # In seconds
            'chanlocs': ch_locs,
            'ref': 'common',
        }

        # Save next to original CSV
        output_path = csv_path.replace('.csv', '_converted.set')
        savemat(output_path, {'EEG': EEG}, appendmat=False)

        logger.debug(f"Converted Ganglion CSV to continuous .set format")
        logger.debug(f"  Duration: {n_samples/sfreq:.1f} seconds ({n_samples} samples)")

        return output_path

    def _close_tab(self, index: int):
        tab: FileTab = self.tab_widget.widget(index)
        if tab and tab.filepath in self._open_paths:
            self._open_paths.remove(tab.filepath)
        self.tab_widget.removeTab(index)
        self._update_empty_state()
        self._update_files_label()

    def _clear_all_tabs(self):
        self.tab_widget.clear()
        self._open_paths.clear()
        self._update_empty_state()
        self._update_files_label()

    def _update_empty_state(self):
        has_tabs = self.tab_widget.count() > 0
        self.tab_widget.setVisible(has_tabs)
        self._empty_label.setVisible(not has_tabs)

    def _update_files_label(self):
        n = self.tab_widget.count()
        if n == 0:
            self.files_label.setText("No files loaded")
        elif n == 1:
            self.files_label.setText(f"1 file open: {self.tab_widget.tabText(0)}")
        else:
            names = [self.tab_widget.tabText(i) for i in range(n)]
            preview = ", ".join(names[:6])
            if n > 6:
                preview += f" … (+{n - 6} more)"
            self.files_label.setText(f"{n} files open: {preview}")

    def _on_dark_mode_toggled(self, state: int):
        app = QApplication.instance()
        if app is None:
            return

        self.is_dark_mode = bool(state)

        if self.is_dark_mode:
            apply_dark_theme(app)
            self._apply_window_dark_styles()
        else:
            apply_light_theme(app)
            self._apply_window_light_styles()

        # Propagate to every open tab
        for i in range(self.tab_widget.count()):
            tab: FileTab = self.tab_widget.widget(i)
            tab.apply_theme(self.is_dark_mode)

    def _apply_window_light_styles(self):
        self.dark_mode_toggle.set_dark_mode(False)
        self.drop_zone.set_dark_mode(False)
        self.tab_widget.setStyleSheet(self._tab_widget_style(dark=False))

        self.drop_browse_frame.setStyleSheet(
            "QFrame { background: #ffffff; border: 1px solid #dadce0; border-radius: 4px; }"
        )
        self._style_clear_btn(dark=False)

        for lbl in [self.title_lbl, self.dark_lbl,
                    self.browse_lbl, self.clear_lbl, self.download_lbl]:
            s = lbl.styleSheet()
            s = s.replace("#e8eaed", "#202124").replace("#9aa0a6", "#5f6368")
            lbl.setStyleSheet(s)
        self.files_label.setStyleSheet("color: #5f6368; font-size: 11px;")
        self._empty_label.setStyleSheet("color: #9aa0a6; font-size: 14px;")

        light_btn_style = (
            "QPushButton { background: #ffffff; color: #202124; border: 1px solid #dadce0;"
            " border-radius: 4px; padding: 4px 8px; }"
            " QPushButton:hover { background: #f1f3f4; }"
        )
        self.browse_btn.setStyleSheet(light_btn_style)
        self.download_btn.setStyleSheet(light_btn_style)

    def _apply_window_dark_styles(self):
        self.dark_mode_toggle.set_dark_mode(True)
        self.drop_zone.set_dark_mode(True)
        self.tab_widget.setStyleSheet(self._tab_widget_style(dark=True))

        self.drop_browse_frame.setStyleSheet(
            "QFrame { background: #121212; border: 1px solid #3c4043; border-radius: 4px; }"
        )
        self._style_clear_btn(dark=True)

        for lbl in [self.title_lbl, self.dark_lbl,
                    self.browse_lbl, self.clear_lbl, self.download_lbl]:
            s = lbl.styleSheet()
            s = s.replace("#202124", "#e8eaed").replace("#5f6368", "#9aa0a6")
            lbl.setStyleSheet(s)
        self.files_label.setStyleSheet("color: #9aa0a6; font-size: 11px;")
        self._empty_label.setStyleSheet("color: #5f6368; font-size: 14px;")

        dark_btn_style = (
            "QPushButton { background: #303134; color: #e8eaed; border: 1px solid #5f6368;"
            " border-radius: 4px; padding: 4px 8px; }"
            " QPushButton:hover { background: #3c4043; }"
        )
        self.browse_btn.setStyleSheet(dark_btn_style)
        self.download_btn.setStyleSheet(dark_btn_style)

    def _style_clear_btn(self, dark: bool):
        if dark:
            self.clear_btn.setStyleSheet(
                "QPushButton { background: #202124; border: 1px solid #f28b82; border-radius: 4px;"
                " color: #f28b82; font-size: 16px; }"
                "QPushButton:hover { background: #3c4043; }"
            )
        else:
            self.clear_btn.setStyleSheet(
                "QPushButton { background: #ffffff; border: 1px solid #d93025; border-radius: 4px;"
                " color: #d93025; font-size: 16px; }"
                "QPushButton:hover { background: #fce8e6; }"
            )

    @staticmethod
    def _tab_widget_style(dark: bool) -> str:
        if dark:
            return (
                "QTabWidget::pane { border: 1px solid #3c4043; background: #1e1e1e; border-radius: 4px; }"
                "QTabBar::tab { background: #2d2d2d; color: #9aa0a6; padding: 6px 16px;"
                " border: 1px solid #3c4043; border-bottom: none; border-radius: 4px 4px 0 0; margin-right: 2px; }"
                "QTabBar::tab:selected { background: #1e1e1e; color: #e8eaed; border-bottom: 1px solid #1e1e1e; }"
                "QTabBar::tab:hover { background: #3c4043; color: #e8eaed; }"
                "QTabBar::close-button { subcontrol-position: right; }"
            )
        else:
            return (
                "QTabWidget::pane { border: 1px solid #dadce0; background: #ffffff; border-radius: 4px; }"
                "QTabBar::tab { background: #f1f3f4; color: #5f6368; padding: 6px 16px;"
                " border: 1px solid #dadce0; border-bottom: none; border-radius: 4px 4px 0 0; margin-right: 2px; }"
                "QTabBar::tab:selected { background: #ffffff; color: #202124; border-bottom: 1px solid #ffffff; }"
                "QTabBar::tab:hover { background: #e8eaed; color: #202124; }"
                "QTabBar::close-button { subcontrol-position: right; }"
            )
