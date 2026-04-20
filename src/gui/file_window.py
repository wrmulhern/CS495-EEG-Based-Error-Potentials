"""
Main application window and per-file tab widget for the ErrP Visualizer.

Classes
-------
* :class:`FileTab` — self-contained widget for a single loaded EEG
  file.  Owns the Matplotlib canvas, all "Graph Options" controls, the
  animated-topomap playback logic, and lazy data loading.
* :class:`FileWindow` — top-level ``QMainWindow`` that manages the tab
  bar, bottom drag-and-drop zone, file browsing, CSV-to-.set
  conversion, dark-mode toggling, and the "Record EEG" / "Help"
  buttons.

Design notes
~~~~~~~~~~~~
* **Lazy loading** — ``FileTab`` does not read the ``.set`` file until
  the user clicks *Visualize* for the first time.  A modal
  ``QProgressDialog`` provides feedback during loading.
* **Independent tabs** — each tab keeps its own ``EpochsData``, figure,
  and control state.  Switching tabs is instant.
* **Theming** — light and dark palettes are applied to both Qt widgets
  *and* embedded Matplotlib figures via ``_apply_mpl_theme`` and per-
  widget stylesheet swapping.
"""

import os
import logging
from typing import List, Optional

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QKeySequence, QIcon
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
    QVBoxLayout,
    QWidget,
    QTabWidget,
    QMessageBox,
    QShortcut,
    QSlider,
    QProgressDialog,
)

from src.data_processing.data_loader import read_epochs_eeglab_minimal
from src.data_processing.file_validator import FileValidator, FileValidationError
from src.data_processing.data_processor import average_epochs, select_time_window
from src.data_processing.csv_converter import convert_ganglion_csv_to_set
from src.data_visualization.visualizer import (
    plot_evoked, plot_topomap, plot_joint, plot_topomap_frame,
    _apply_mpl_theme,
)
from .utils.drag_and_drop import FileDropFrame
from .utils.checkbox import ToggleSwitch
from .utils.multi_select import MultiSelectDropdown, MultiSelectItemDelegate
from .help_dialog import HelpDialog
from .themes.theme import apply_theme
from .themes.colors import get_palette
from .flanker_window import FlankerWindow
from src.config import VALIDATION, EXPORT, PLOT

logger = logging.getLogger(__name__)
class FileTab(QWidget):
    """Self-contained tab widget for a single loaded ``.set`` file.

    Each ``FileTab`` owns:

    * A Matplotlib ``Figure`` / ``FigureCanvas`` embedded in the left
      pane, showing either a placeholder or the latest visualisation.
    * A "Graph Options" panel on the right with epoch-range inputs,
      channel selection (:class:`MultiSelectDropdown`), graph-type
      combo, topomap-mode selector, animation controls, and the
      *Visualize* button.
    * Cached :class:`~src.data_processing.data_loader.EpochsData` —
      populated lazily the first time the user clicks *Visualize*.
    * Animation state for the animated-topomap mode (``QTimer``,
      ``QSlider``, play/pause logic).

    The parent :class:`FileWindow` owns the dark-mode toggle and
    drag-and-drop zone; it calls :meth:`apply_theme` when the global
    theme changes.

    Parameters:
        filepath (str): Absolute path to the ``.set`` file.
        is_dark_mode (bool): Initial theme state.
        parent (QWidget | None): Parent widget (usually the
            ``QTabWidget``).
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
        self._apply_mpl_theme_to_fig(self.figure)


    def _build_graph_frame(self) -> QFrame:
        """Create the left-hand pane containing the Matplotlib canvas."""
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
        """Build the right-hand "Graph Options" panel and all sub-controls."""
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
        p = get_palette(self._is_dark_mode)
        self.epoch_label.setStyleSheet(f"color: {p.text}; font-size: 12px;")

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
        sensor_label.setStyleSheet(f"color: {p.text}; font-size: 12px;")
        self.sensor_combo = MultiSelectDropdown(["All Channels"])
        self.sensor_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.sensor_combo.confirmed.connect(self.mark_needs_update)
        sensor_layout.addWidget(sensor_label)
        sensor_layout.addWidget(self.sensor_combo)
        layout.addWidget(self.sensor_container)

        # Graph type dropdown
        graph_type_label = QLabel("Graph Type")
        graph_type_label.setStyleSheet(f"color: {p.text}; font-size: 12px;")
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
        topo_label = QLabel("Topomap times (ms)")
        topo_label.setStyleSheet(f"color: {p.text}; font-size: 12px;")
        topo_layout.addWidget(topo_label)

        topo_row = QHBoxLayout()
        topo_row.setSpacing(8)
        self.topo_time_1 = QLineEdit()
        self.topo_time_1.setPlaceholderText("100")
        self.topo_time_1.setFixedWidth(70)
        self.topo_time_1.textChanged.connect(self.mark_needs_update)
        self.topo_time_2 = QLineEdit()
        self.topo_time_2.setPlaceholderText("200")
        self.topo_time_2.setFixedWidth(70)
        self.topo_time_2.textChanged.connect(self.mark_needs_update)
        self.topo_time_3 = QLineEdit()
        self.topo_time_3.setPlaceholderText("300")
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
        self.topo_mode_label.setStyleSheet(f"color: {p.text}; font-size: 12px;")
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
        anim_speed_label.setStyleSheet(f"color: {p.text}; font-size: 12px;")
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
        self.anim_time_label.setStyleSheet(f"color: {p.text}; font-size: 12px;")
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
        self.events_checkbox.setStyleSheet(f"font-size: 12px; color: {p.text};")
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

    def ensure_loaded(self) -> bool:
        """Lazily load the ``.set`` file, showing a progress dialog.

        On first call, reads the file via
        :func:`~src.data_processing.data_loader.read_epochs_eeglab_minimal`,
        populates :attr:`current_epochs`, and refreshes the channel
        dropdown.  Subsequent calls are no-ops.

        Returns:
            bool: ``True`` if data is available, ``False`` on failure.
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

    def visualize(self):
        """Run the full visualisation pipeline for the current options.

        Steps: ensure data is loaded → apply epoch window → select
        channels → average epochs → dispatch to the appropriate plot
        function → replace the canvas.  For the animated topomap mode,
        delegates to :meth:`_setup_animated_topomap` instead.
        """
        if not self.ensure_loaded():
            return

        graph_type = self.graph_type_combo.currentText()

        if graph_type in ("Topographic Map", "Joint Maps"):
                n_channels = len(self.current_epochs.ch_names)
                if n_channels < VALIDATION.min_topo_channels:
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
        """Swap out the current Matplotlib canvas for one backed by *new_fig*."""
        layout = self.graph_frame.layout()
        layout.removeWidget(self.canvas)
        self.canvas.setParent(None)
        self.canvas.deleteLater()

        self.figure = new_fig
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, stretch=1)
        self.canvas.draw_idle()

    def _draw_placeholder(self):
        """Draw a centred "Load data and click Visualize" message."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, "Load data and click Visualize",
                ha="center", va="center", fontsize=16,
                color="#9aa0a6" if self._is_dark_mode else "#5f6368")
        ax.axis("off")

    def _on_graph_type_changed(self, graph_type: str):
        """Show / hide sub-controls and manage animation state on type switch.

        Controls visibility of: epoch inputs, sensor dropdown, topomap
        time fields, topomap mode selector, animation controls, and
        the events checkbox.  Also restores the previous channel
        selection when returning to Time Series or Joint Maps.
        """
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
                self.sensor_combo.update_button_text()
            else:
                self._restore_default_sensor_selection()
        else:
            # Time Series mode
            self.sensor_combo.set_items(self._all_sensors if self._all_sensors else ["All Channels"])
            if self._last_time_series_selection and any(s in self._all_sensors for s in self._last_time_series_selection):
                for sensor in self._last_time_series_selection:
                    if sensor in self._all_sensors:
                        item = self.sensor_combo.list_widget.findItems(sensor, Qt.MatchExactly)[0]
                        MultiSelectItemDelegate.update_item_style(item, True)
                self.sensor_combo.selected = set(self._last_time_series_selection)
                self.sensor_combo.update_button_text()
            else:
                self._restore_default_sensor_selection()

        self._last_graph_type = graph_type

    def _restore_default_sensor_selection(self):
        """Set sensor dropdown to "All Channels" and update the button text."""
        item = self.sensor_combo.list_widget.findItems("All Channels", Qt.MatchExactly)[0]
        MultiSelectItemDelegate.update_item_style(item, True)
        self.sensor_combo.selected = {"All Channels"}
        self.sensor_combo.update_button_text()

    def _set_epoch_field_style(self, disabled: bool):
        """Apply greyed-out or active styling to the epoch start/end fields."""
        p = get_palette(self._is_dark_mode)
        if disabled:
            bg, fg = p.surface_dim, p.text_disabled
        else:
            bg, fg = (p.surface_alt if self._is_dark_mode else p.surface), p.text
        s = f"QLineEdit {{ background: {bg}; color: {fg}; border: 1px solid {p.border}; border-radius: 4px; }}"
        lc = f"color: {fg}; font-size: 12px;"
        self.epoch_start.setStyleSheet(s)
        self.epoch_end.setStyleSheet(s)
        self.epoch_label.setStyleSheet(lc)

    def _on_events_checkbox_changed(self, _state):
        self.events_checkbox_checked = self.events_checkbox.isChecked()

    # ---- Animated topomap helpers ----

    def _on_topo_mode_changed(self, mode: str):
        """Toggle between Static and Animated sub-controls."""
        is_animated = mode == "Animated"
        self.topo_times_container.setVisible(not is_animated)
        self.anim_controls_container.setVisible(is_animated)
        if not is_animated:
            self._stop_animation()
        self.mark_needs_update()

    def _setup_animated_topomap(self, evoked, theme):
        """Initialise the animated topomap: render frame 0, configure slider range."""
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
        """Redraw the topomap at the sample index given by *value*."""
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
        """Format the current time in ms and display it beside the slider."""
        if self._anim_evoked is not None and slider_value < len(self._anim_evoked.times):
            time_ms = self._anim_evoked.times[slider_value] * 1000
            self.anim_time_label.setText(f"{time_ms:.1f} ms")

    def _toggle_animation(self):
        """Play ↔ Pause toggle for the animated topomap."""
        if self._anim_evoked is None:
            return
        if self._anim_playing:
            self._pause_animation()
        else:
            self._start_animation()

    def _start_animation(self):
        """Start the QTimer that advances the slider every 50 ms."""
        if self._anim_evoked is None:
            return
        self._anim_playing = True
        self.anim_play_btn.setText("⏸  Pause")
        self._anim_timer.start(PLOT.anim_timer_ms)

    def _pause_animation(self):
        """Stop the timer and restore the Play button label."""
        self._anim_playing = False
        self.anim_play_btn.setText("▶  Play")
        self._anim_timer.stop()

    def _stop_animation(self):
        """Pause playback and discard the cached evoked data."""
        self._pause_animation()
        self._anim_evoked = None

    def _on_anim_tick(self):
        """Timer callback: advance the slider by a speed-dependent step, looping at the end."""
        if self._anim_evoked is None:
            self._pause_animation()
            return

        speed_text = self.anim_speed_combo.currentText()
        speed = float(speed_text.replace("x", ""))

        sfreq = self._anim_evoked.sfreq
        step = max(1, round(sfreq * PLOT.anim_tick_duration_s * speed))

        current = self.anim_slider.value()
        new_val = current + step
        if new_val > self.anim_slider.maximum():
            new_val = 0
        self.anim_slider.setValue(new_val)

    def _parse_topomap_times(self) -> List[float]:
        """Read the three topomap-time text fields (ms), falling back to 100 / 200 / 300 ms.

        Returns times converted to seconds for downstream use.
        """
        defaults_ms = [100, 200, 300]
        result_ms = []
        for i, w in enumerate([self.topo_time_1, self.topo_time_2, self.topo_time_3]):
            t = w.text().strip()
            if not t:
                result_ms.append(defaults_ms[i])
                continue
            try:
                result_ms.append(float(t))
            except ValueError:
                return [ms / 1000.0 for ms in defaults_ms]
        result_ms = result_ms if result_ms else defaults_ms
        return [ms / 1000.0 for ms in result_ms]

    def mark_needs_update(self):
        """Highlight the Visualize button to signal that options have changed."""
        p = get_palette(self._is_dark_mode)
        fg = "#000000" if self._is_dark_mode else "#ffffff"
        self.visualize_btn.setStyleSheet(
            f"QPushButton {{ background: {p.accent}; border: 1px solid {p.accent}; border-radius: 4px;"
            f" font-size: 14px; color: {fg}; }}"
            f"QPushButton:hover {{ background: {p.accent_hover}; }}"
            f"QPushButton:pressed {{ background: {p.accent_pressed}; }}"
        )

    def reset_visualize_button(self):
        """Return the Visualize button to its neutral (non-highlighted) style."""
        p = get_palette(self._is_dark_mode)
        bg = p.surface_alt if self._is_dark_mode else p.surface
        self.visualize_btn.setStyleSheet(
            f"QPushButton {{ background: {bg}; border: 1px solid {p.border}; border-radius: 4px;"
            f" font-size: 14px; color: {p.text}; }}"
            f"QPushButton:hover {{ background: {p.surface_hover}; }}"
            f"QPushButton:pressed {{ background: {p.accent_tint}; }}"
        )

    def apply_theme(self, is_dark: bool):
        """Re-style every widget in this tab for light or dark mode.

        Called by :meth:`FileWindow._on_dark_mode_toggled` whenever the
        global toggle changes.  Updates the graph frame, options box,
        all ``QLineEdit``/``QComboBox``/``QLabel``/``QCheckBox``
        children, the animation controls, and the embedded Matplotlib
        figure (either by redrawing the current animation frame or by
        applying ``_apply_mpl_theme`` to a static figure).
        """
        self._is_dark_mode = is_dark
        p = get_palette(is_dark)
        input_bg = p.surface_alt if is_dark else p.surface

        self.graph_frame.setStyleSheet(
            f"QFrame {{ background: {p.window}; border: 1px solid {p.border_strong}; border-radius: 4px; }}"
        )

        self.options_box.setStyleSheet(
            f"QGroupBox {{ font-size: 13px; font-weight: 600; color: {p.text}; }}"
            f"QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {p.text}; }}"
        )

        le_style = (
            f"QLineEdit {{ background: {input_bg}; color: {p.text}; border: 1px solid {p.border}; border-radius: 4px; }}"
        )
        for le in self.findChildren(QLineEdit):
            le.setStyleSheet(le_style)

        cb_style = (
            f"QComboBox {{ background: {input_bg}; color: {p.text}; border: 1px solid {p.border}; border-radius: 4px; }}"
            f"QComboBox::drop-down {{ subcontrol-origin: padding; subcontrol-position: top right;"
            f" width: 18px; border-left: 1px solid {p.border}; }}"
        )
        for cb in self.findChildren(QComboBox):
            cb.setStyleSheet(cb_style)

        for lbl in self.findChildren(QLabel):
            s = lbl.styleSheet() or ""
            old_txt = "#e8eaed" if not is_dark else "#202124"
            old_sec = "#9aa0a6" if not is_dark else "#5f6368"
            s = s.replace(old_txt, p.text).replace(old_sec, p.text_secondary)
            lbl.setStyleSheet(s)

        self.events_checkbox.setStyleSheet(
            f"QCheckBox {{ font-size: 12px; color: {p.text}; }}"
            f"QCheckBox::indicator {{ width: 16px; height: 16px; }}"
            f"QCheckBox::indicator:unchecked {{ border-radius: 3px; border: 1px solid {p.border}; background: {input_bg}; }}"
            f"QCheckBox::indicator:checked {{ border-radius: 3px; border: 1px solid {p.accent}; background: {p.accent}; }}"
        )

        self.anim_play_btn.setStyleSheet(
            f"QPushButton {{ background: {p.surface_elevated}; color: {p.text}; border: 1px solid {p.border};"
            f" border-radius: 4px; padding: 4px 10px; font-size: 12px; }}"
            f"QPushButton:hover {{ background: {p.surface_hover}; }}"
            f"QPushButton:pressed {{ background: {p.surface_pressed}; }}"
        )
        self.anim_slider.setStyleSheet(
            f"QSlider::groove:horizontal {{ background: {p.border_strong}; height: 6px; border-radius: 3px; }}"
            f"QSlider::handle:horizontal {{ background: {p.accent}; width: 14px; margin: -4px 0;"
            f" border-radius: 7px; }}"
            f"QSlider::sub-page:horizontal {{ background: {p.accent}; border-radius: 3px; }}"
        )

        # Keep animation theme in sync so slider redraws use the right colours
        self._anim_theme = "dark" if is_dark else "light"

        # Re-apply epoch field disabled style if currently on Topographic Map
        self._on_graph_type_changed(self.graph_type_combo.currentText())

        # Retheme the live matplotlib figure
        if self._anim_evoked is not None:
            self._on_anim_slider_changed(self.anim_slider.value())
        else:
            self._apply_mpl_theme_to_fig(self.figure)
            self.canvas.draw_idle()

        # Update button appearance
        self.reset_visualize_button()

    def _apply_mpl_theme_to_fig(self, fig: Figure):
        """Apply the current light/dark palette to a static Matplotlib figure."""
        if fig is None:
            return
        theme = "dark" if self._is_dark_mode else "light"
        _apply_mpl_theme(fig, fig.get_axes(), theme=theme)

class FileWindow(QMainWindow):
    """Top-level application window for the ErrP Visualizer.

    Manages the global layout:

    * **Top bar** — app title, "Record EEG" button, "Help" button,
      dark-mode toggle.
    * **Tab area** — a ``QTabWidget`` holding one :class:`FileTab` per
      loaded file.  Shows a placeholder label when empty.
    * **Bottom bar** — drag-and-drop zone (:class:`FileDropFrame`),
      "Browse" / "Clear All" buttons, "Download Graph" button, and a
      status label with the count/names of open files.

    File lifecycle: paths are validated via :class:`FileValidator`,
    ``.csv`` files are auto-converted to ``.set`` via
    :meth:`convert_ganglion_csv_to_set`, and duplicate paths are
    silently skipped.

    Parameters:
        file_path (str | None): Optional path to open on launch.
    """

    def __init__(self, file_path: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("ErrP Visualizer")
        self.resize(*PLOT.main_window_size)

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

        self._apply_window_theme(dark=False)

        if file_path:
            self.add_files([file_path])

    def _build_top_bar(self):
        """Top bar: title | [Record EEG] [? Help] | Dark mode"""
        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        bar.setSpacing(14)

        # Title
        title_lbl = QLabel("ErrP Visualizer")
        p = get_palette(False)
        title_lbl.setStyleSheet(f"font-size: 15px; font-weight: 700; color: {p.text};")
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
        dark_lbl.setStyleSheet(f"font-size: 13px; color: {p.text};")
        self.dark_lbl = dark_lbl
        self.dark_mode_toggle = ToggleSwitch("")
        self.dark_mode_toggle.set_dark_mode(False)
        self.dark_mode_toggle.stateChanged.connect(self._on_dark_mode_toggled)
        dark_row.addWidget(dark_lbl)
        dark_row.addWidget(self.dark_mode_toggle)
        bar.addLayout(dark_row)

        self.outer.addLayout(bar)

    def _open_help(self):
        """Show the modal :class:`HelpDialog`."""
        dlg = HelpDialog(is_dark=self.is_dark_mode, parent=self)
        dlg.exec_()

    def _open_live_recording(self):
        """Launch the Flanker-task recording dialog and auto-import the result."""
        dlg = FlankerWindow(
            is_dark=self.is_dark_mode,
            output_dir=os.path.expanduser("~"),
            parent=self,
        )
        dlg.recording_finished.connect(lambda path: self.add_files([path]))
        dlg.exec_()

    def _style_help_btn(self, dark: bool):
        """Apply light or dark stylesheet to the Help button."""
        p = get_palette(dark)
        self.help_btn.setStyleSheet(
            f"QPushButton {{ background: {p.surface_elevated}; color: {p.text}; border: 1px solid {p.border};"
            f" border-radius: 4px; padding: 4px 12px; font-size: 13px; }}"
            f"QPushButton:hover {{ background: {p.surface_hover}; }}"
            f"QPushButton:pressed {{ background: {p.surface_pressed}; }}"
        )

    def _style_record_eeg_btn(self, dark: bool):
        """Apply light or dark stylesheet to the Record EEG button."""
        p = get_palette(dark)
        self.record_eeg_btn.setStyleSheet(
            f"QPushButton {{ background: {p.surface_elevated}; color: {p.danger}; border: 1px solid {p.danger_border};"
            f" border-radius: 4px; padding: 4px 12px; font-size: 13px; }}"
            f"QPushButton:hover {{ background: {p.danger_hover}; }}"
            f"QPushButton:pressed {{ background: {p.surface_pressed}; }}"
        )

    def _build_tab_area(self):
        """QTabWidget that holds one FileTab per loaded file."""
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.setMovable(True)
        self.tab_widget.tabCloseRequested.connect(self._close_tab)
        self.tab_widget.setStyleSheet(self._tab_widget_style(dark=False))

        # Show a placeholder when no tabs are open
        p = get_palette(False)
        self._empty_label = QLabel("Drop files below or use Browse to get started")
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._empty_label.setStyleSheet(f"color: {p.text_disabled}; font-size: 14px;")

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
        p = get_palette(False)
        download_lbl = QLabel("Download")
        download_lbl.setStyleSheet(f"font-size: 13px; color: {p.text};")
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
        browse_lbl.setStyleSheet(f"font-size: 13px; color: {p.text};")
        browse_lbl.setAlignment(Qt.AlignHCenter)
        browse_lbl.setFixedWidth(BTN_W)
        self.browse_lbl = browse_lbl
        self.browse_btn = QPushButton("…")
        self.browse_btn.setCursor(Qt.PointingHandCursor)
        self.browse_btn.setFixedSize(BTN_W, BTN_H)
        self.browse_btn.clicked.connect(self._browse_files)

        clear_lbl = QLabel("Clear All")
        clear_lbl.setStyleSheet(f"font-size: 13px; color: {p.text};")
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
        self.files_label.setStyleSheet(f"color: {p.text_secondary}; font-size: 11px;")
        self.files_label.setWordWrap(True)
        df_layout.addWidget(self.files_label, 1, 0, 1, 5)

        bar.addWidget(drop_frame, stretch=2)
        bar.addStretch(1)

        self.outer.addLayout(bar)


    def _browse_files(self):
        """Open a native file dialog filtered to ``.set`` / ``.csv``."""
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select .set or .csv file(s)",
            "",
            EXPORT.file_dialog_filter
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
            EXPORT.save_dialog_filter
        )

        if filename:
            try:
                # Ensure the filename has .png extension
                if not filename.lower().endswith(EXPORT.save_format):
                    filename += EXPORT.save_format

                current_tab.figure.savefig(filename, dpi=EXPORT.save_dpi, bbox_inches='tight')
                QMessageBox.information(self, "Success", f"Graph saved as {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save graph:\n{str(e)}")


    def add_files(self, paths: List[str]):
        """Validate, optionally convert, and open one tab per new file.

        Each path goes through :class:`FileValidator`.  ``.csv`` files
        are auto-converted to ``.set`` via
        :meth:`convert_ganglion_csv_to_set`.  Duplicate paths (already
        open) are silently skipped.  The first newly added tab is
        activated.
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
            idx = self.tab_widget.addTab(tab, label)
            self._make_close_btn(idx)
            added.append(label)

        if added:
            first_new_idx = self.tab_widget.count() - len(added)
            self.tab_widget.setCurrentIndex(first_new_idx)
            self._update_empty_state()
            self._update_files_label()

    @staticmethod
    def convert_ganglion_csv_to_set(csv_path: str) -> str:
        """Delegate to :func:`src.data_processing.csv_converter.convert_ganglion_csv_to_set`."""
        return convert_ganglion_csv_to_set(csv_path)

    def _close_tab(self, index: int):
        """Remove a single tab and untrack its file path."""
        tab: FileTab = self.tab_widget.widget(index)
        if tab and tab.filepath in self._open_paths:
            self._open_paths.remove(tab.filepath)
        self.tab_widget.removeTab(index)
        self._update_empty_state()
        self._update_files_label()

    def _clear_all_tabs(self):
        """Close every tab and reset the tracked-paths list."""
        self.tab_widget.clear()
        self._open_paths.clear()
        self._update_empty_state()
        self._update_files_label()

    def _update_empty_state(self):
        """Show the tab widget or the "no files" placeholder."""
        has_tabs = self.tab_widget.count() > 0
        self.tab_widget.setVisible(has_tabs)
        self._empty_label.setVisible(not has_tabs)

    def _update_files_label(self):
        """Refresh the bottom-bar status text with the count and names of open files."""
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
        """Handle the dark-mode toggle: re-theme the window and every open tab."""
        app = QApplication.instance()
        if app is None:
            return

        self.is_dark_mode = bool(state)

        apply_theme(app, is_dark=self.is_dark_mode)
        self._apply_window_theme(dark=self.is_dark_mode)

        # Propagate to every open tab
        for i in range(self.tab_widget.count()):
            tab: FileTab = self.tab_widget.widget(i)
            tab.apply_theme(self.is_dark_mode)

    def _apply_window_theme(self, dark: bool):
        """Apply window-level styles for the given theme."""
        p = get_palette(dark)

        self.dark_mode_toggle.set_dark_mode(dark)
        self.drop_zone.set_dark_mode(dark)
        self.tab_widget.setStyleSheet(self._tab_widget_style(dark=dark))

        self.drop_browse_frame.setStyleSheet(
            f"QFrame {{ background: {p.window}; border: 1px solid {p.border_strong}; border-radius: 4px; }}"
        )
        self._style_clear_btn(dark=dark)

        for lbl in [self.title_lbl, self.dark_lbl,
                    self.browse_lbl, self.clear_lbl, self.download_lbl]:
            s = lbl.styleSheet() or ""
            old_txt = "#e8eaed" if not dark else "#202124"
            old_sec = "#9aa0a6" if not dark else "#5f6368"
            s = s.replace(old_txt, p.text).replace(old_sec, p.text_secondary)
            lbl.setStyleSheet(s)
        self.files_label.setStyleSheet(f"color: {p.text_secondary}; font-size: 11px;")
        self._empty_label.setStyleSheet(f"color: {p.text_disabled}; font-size: 14px;")

        btn_style = (
            f"QPushButton {{ background: {p.surface_elevated}; color: {p.text}; border: 1px solid {p.border};"
            f" border-radius: 4px; padding: 4px 8px; }}"
            f" QPushButton:hover {{ background: {p.surface_hover}; }}"
        )
        self.browse_btn.setStyleSheet(btn_style)
        self.download_btn.setStyleSheet(btn_style)

        self._style_help_btn(dark=dark)
        self._style_record_eeg_btn(dark=dark)
        self._restyle_all_close_btns(dark=dark)

    def _style_clear_btn(self, dark: bool):
        """Apply light or dark stylesheet to the "Clear All" button."""
        p = get_palette(dark)
        bg = p.surface_alt if dark else p.surface
        self.clear_btn.setStyleSheet(
            f"QPushButton {{ background: {bg}; border: 1px solid {p.danger_border}; border-radius: 4px;"
            f" color: {p.danger}; font-size: 16px; }}"
            f"QPushButton:hover {{ background: {p.danger_hover}; }}"
        )

    @staticmethod
    def _tab_widget_style(dark: bool) -> str:
        """Return the ``QTabWidget`` / ``QTabBar`` stylesheet for the given theme."""
        p = get_palette(dark)
        return (
            f"QTabWidget::pane {{ border: 1px solid {p.border_strong}; background: {p.surface}; border-radius: 4px; }}"
            f"QTabBar::tab {{ background: {p.surface_dim}; color: {p.text_secondary}; padding: 6px 16px;"
            f" border: 1px solid {p.border_strong}; border-bottom: none; border-radius: 4px 4px 0 0; margin-right: 2px; }}"
            f"QTabBar::tab:selected {{ background: {p.surface}; color: {p.text}; border-bottom: 1px solid {p.surface}; }}"
            f"QTabBar::tab:hover {{ background: {p.surface_hover}; color: {p.text}; }}"
            f"QTabBar::close-button {{ subcontrol-position: right; }}"
        )

    def _make_close_btn(self, index: int) -> QPushButton:
        """Create a themed 'x' close button and attach it to the tab at *index*."""
        btn = QPushButton("×")
        btn.setFixedSize(18, 18)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setFlat(True)
        btn.clicked.connect(self._close_tab_by_btn)
        self._style_close_btn(btn, self.is_dark_mode)
        self.tab_widget.tabBar().setTabButton(index, self.tab_widget.tabBar().RightSide, btn)
        return btn

    @staticmethod
    def _style_close_btn(btn: QPushButton, dark: bool):
        """Apply theme-aware stylesheet to a tab close button."""
        p = get_palette(dark)
        btn.setStyleSheet(
            f"QPushButton {{ color: {p.text_secondary}; border: none; border-radius: 9px;"
            f" font-size: 14px; font-weight: bold; padding: 0; margin: 0; background: transparent; }}"
            f"QPushButton:hover {{ background: {p.icon_hover_bg}; color: {p.text}; }}"
        )

    def _close_tab_by_btn(self):
        """Handle close-button clicks by finding which tab owns the sender."""
        bar = self.tab_widget.tabBar()
        sender_btn = self.sender()
        for i in range(self.tab_widget.count()):
            if bar.tabButton(i, bar.RightSide) is sender_btn:
                self._close_tab(i)
                return

    def _restyle_all_close_btns(self, dark: bool):
        """Update every existing tab close button for the new theme."""
        bar = self.tab_widget.tabBar()
        for i in range(self.tab_widget.count()):
            btn = bar.tabButton(i, bar.RightSide)
            if isinstance(btn, QPushButton):
                self._style_close_btn(btn, dark)
