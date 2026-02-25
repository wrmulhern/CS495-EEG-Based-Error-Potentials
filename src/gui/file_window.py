import os
from typing import List

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from PyQt5.QtCore import Qt
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

from src.data_processing.data_loader import read_epochs_eeglab_minimal
from src.data_processing.data_processor import average_epochs, select_time_window
from src.data_visualization.visualizer import plot_evoked, plot_topomap, plot_joint
from .utils.drag_and_drop import FileDropFrame
from .utils.checkbox import ToggleSwitch
from .themes.light_theme import apply_light_theme
from .themes.dark_theme import apply_dark_theme


class FileWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ErrP Visualizer")
        self.resize(1200, 720)

        self.selected_files: List[str] = []
        self.current_epochs = None
        # Shared state for events checkbox across Time Series and Joint Maps
        self.events_checkbox_checked = False
        # Track current theme for both Qt widgets and Matplotlib figures
        self.is_dark_mode = False
        # Backup list of sensor combo items so we can restore after forcing All Channels
        self._sensor_items_backup = None
        # Backup the previously selected sensor text for restore when switching graph types
        self._sensor_selected_backup = None
        # Canonical list of all sensors (includes 'All Channels') populated on file load
        self._all_sensors = ["All Channels"]
        # Last selected sensor when viewing ErrP Time Series
        self._last_time_series_selection = "All Channels"
        # Track last graph type to detect transitions
        self._last_graph_type = "ErrP Time Series"

        central = QWidget()
        self.setCentralWidget(central)

        self.outer = QVBoxLayout(central)
        self.outer.setContentsMargins(18, 18, 18, 18)
        self.outer.setSpacing(14)

        # graph and graph options
        self.topInit()
        # live mode, drag and drop, browse
        self.middleInit()
        # visualize
        self.bottomInit()

        # Ensure initial widget styling matches the global light palette
        self.apply_light_styles()


    def topInit(self):
        # --- Top: Graph area (left) + Graph Options (right)
        top_row = QHBoxLayout()
        top_row.setSpacing(18)
        self.outer.addLayout(top_row, stretch=1)

        self.graph_frame = self.graphInit()

        top_row.addWidget(self.graph_frame, stretch=3)

        # Graph Options panel
        options_box = self.graphOptionsInit()

        top_row.addWidget(options_box, stretch=1)

    def graphInit(self):
        # Graph area (placeholder)
        graph_frame = QFrame()
        graph_frame.setFrameShape(QFrame.StyledPanel)
        graph_layout = QVBoxLayout(graph_frame)
        graph_layout.setContentsMargins(10, 10, 10, 10)

        # Create matplotlib figure and canvas
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)

        # Initial placeholder
        ax = self.figure.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            "Load data and click Visualize",
            ha="center",
            va="center",
            fontsize=16,
            color="#5f6368",
        )
        ax.axis("off")

        # Match the initial Matplotlib background to the current theme
        self.apply_current_mpl_theme_to_figure(self.figure)

        graph_layout.addWidget(self.canvas, stretch=1)
        return graph_frame

    def graphOptionsInit(self):
        options_box = QGroupBox("Graph Options")
        self.options_box = options_box
        options_box.setStyleSheet(
            """
            QGroupBox {
                font-size: 13px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px 0 4px;
            }
            """
        )
        options_layout = QVBoxLayout(options_box)
        options_layout.setContentsMargins(14, 16, 14, 14)
        options_layout.setSpacing(14)

        # Dark mode toggle row (top-right, above all other graph options)
        mode_row = QHBoxLayout()
        mode_row.addStretch(1)
        self.dark_mode_toggle = ToggleSwitch("Dark mode")
        self.dark_mode_toggle.set_dark_mode(self.is_dark_mode)
        self.dark_mode_toggle.stateChanged.connect(self.on_dark_mode_toggled)
        mode_row.addWidget(self.dark_mode_toggle, alignment=Qt.AlignRight)
        options_layout.addLayout(mode_row)

        # Epoch inputs row
        self.epoch_label = QLabel("Epoch (in ms)")
        self.epoch_label.setStyleSheet("color: #202124; font-size: 12px;")

        epoch_row = QHBoxLayout()
        epoch_row.setSpacing(10)

        self.epoch_start = QLineEdit()
        self.epoch_start.setPlaceholderText("Start")
        self.epoch_start.setFixedWidth(110)
        self.epoch_start.textChanged.connect(self.mark_visualize_button_needs_update)

        self.epoch_end = QLineEdit()
        self.epoch_end.setPlaceholderText("End")
        self.epoch_end.setFixedWidth(110)
        self.epoch_end.textChanged.connect(self.mark_visualize_button_needs_update)

        dash = QLabel("—")
        dash.setAlignment(Qt.AlignCenter)
        dash.setFixedWidth(16)

        epoch_row.addWidget(self.epoch_start)
        epoch_row.addWidget(dash)
        epoch_row.addWidget(self.epoch_end)
        epoch_row.addStretch(1)

        options_layout.addWidget(self.epoch_label)
        options_layout.addLayout(epoch_row)

        # Sensor dropdown
        sensor_label = QLabel("Sensor")
        sensor_label.setStyleSheet("color: #202124; font-size: 12px;")

        self.sensor_combo = QComboBox()
        self.sensor_combo.addItems(["All Channels"])
        self.sensor_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.sensor_combo.currentTextChanged.connect(self.mark_visualize_button_needs_update)

        options_layout.addWidget(sensor_label)
        options_layout.addWidget(self.sensor_combo)

        # Graph Type dropdown
        graph_type_label = QLabel("Graph Type")
        graph_type_label.setStyleSheet("color: #202124; font-size: 12px;")

        self.graph_type_combo = QComboBox()
        self.graph_type_combo.addItems(["ErrP Time Series", "Topographic Map", "Joint Maps"])  # placeholder
        self.graph_type_combo.currentTextChanged.connect(self.on_graph_type_changed)
        self.graph_type_combo.currentTextChanged.connect(self.mark_visualize_button_needs_update)

        options_layout.addWidget(graph_type_label)
        options_layout.addWidget(self.graph_type_combo)

        # Container for topomap times (visible only for Topographic Map / Joint Maps)
        self.topo_times_container = QWidget()
        topo_times_layout = QVBoxLayout(self.topo_times_container)
        topo_times_layout.setContentsMargins(0, 0, 0, 0)
        topo_times_layout.setSpacing(6)

        topo_times_label = QLabel("Topomap times (s)")
        topo_times_label.setStyleSheet("color: #202124; font-size: 12px;")
        topo_times_layout.addWidget(topo_times_label)

        topo_times_row = QHBoxLayout()
        topo_times_row.setSpacing(8)
        self.topo_time_1 = QLineEdit()
        self.topo_time_1.setPlaceholderText("0.1")
        self.topo_time_1.setFixedWidth(70)
        self.topo_time_1.textChanged.connect(self.mark_visualize_button_needs_update)
        self.topo_time_2 = QLineEdit()
        self.topo_time_2.setPlaceholderText("0.2")
        self.topo_time_2.setFixedWidth(70)
        self.topo_time_2.textChanged.connect(self.mark_visualize_button_needs_update)
        self.topo_time_3 = QLineEdit()
        self.topo_time_3.setPlaceholderText("0.3")
        self.topo_time_3.setFixedWidth(70)
        self.topo_time_3.textChanged.connect(self.mark_visualize_button_needs_update)
        topo_times_row.addWidget(self.topo_time_1)
        topo_times_row.addWidget(self.topo_time_2)
        topo_times_row.addWidget(self.topo_time_3)
        topo_times_row.addStretch(1)
        topo_times_layout.addLayout(topo_times_row)

        options_layout.addWidget(self.topo_times_container)
        self.topo_times_container.setVisible(False)

        # Container for Events checkbox (can be hidden/shown)
        self.events_checkbox_container = QWidget()
        events_container_layout = QVBoxLayout(self.events_checkbox_container)
        events_container_layout.setContentsMargins(0, 0, 0, 0)
        events_container_layout.setSpacing(0)

        # Checkbox
        self.events_checkbox = QCheckBox("Display Events and Responses")
        self.events_checkbox.setStyleSheet("font-size: 12px; color: #202124;")
        self.events_checkbox.stateChanged.connect(self.on_events_checkbox_state_changed)
        self.events_checkbox.stateChanged.connect(self.mark_visualize_button_needs_update)
        events_container_layout.addWidget(self.events_checkbox)

        options_layout.addWidget(self.events_checkbox_container)

        options_layout.addStretch(1)
        return options_box


    def middleInit(self):
        mid_row = QHBoxLayout()
        mid_row.setSpacing(18)
        self.outer.addLayout(mid_row)

        # Live mode toggle area
        live_col = self.liveModeToggleInit()

        mid_row.addLayout(live_col, stretch=1)

        # Drag/drop + Browse group (mimics the wide box)
        drop_browse_frame = self.dropBrowseFileInit()

        mid_row.addWidget(drop_browse_frame, stretch=2)
        mid_row.addStretch(1)

    def liveModeToggleInit(self):
        live_col = QVBoxLayout()
        live_label = QLabel("Live mode")
        live_label.setStyleSheet("font-size: 13px; color: #202124;")
        self.live_toggle = ToggleSwitch("")
        live_col.addWidget(live_label)
        live_col.addWidget(self.live_toggle)
        live_col.addStretch(1)
        return live_col

    def dropBrowseFileInit(self):
        drop_browse_frame = QFrame()
        drop_browse_frame.setFrameShape(QFrame.StyledPanel)
        self.drop_browse_frame = drop_browse_frame
        drop_browse_layout = QGridLayout(drop_browse_frame)
        drop_browse_layout.setContentsMargins(14, 14, 14, 14)
        drop_browse_layout.setHorizontalSpacing(14)
        drop_browse_layout.setVerticalSpacing(8)

        self.drop_zone = FileDropFrame()
        self.drop_zone.filesDropped.connect(self.add_files)
        drop_browse_layout.addWidget(self.drop_zone, 0, 0, 1, 3)

        browse_col = QVBoxLayout()
        browse_label = QLabel("Browse")
        browse_label.setStyleSheet("font-size: 13px; color: #202124;")

        # browse button
        self.browse_btn = QPushButton("…")
        self.browse_btn.setFixedWidth(70)
        self.browse_btn.clicked.connect(self.browse_files)

        # clear button
        clear_label = QLabel("Clear")
        clear_label.setStyleSheet("font-size: 13px; color: #202124;")
        clear_label.setAlignment(Qt.AlignHCenter)

        self.clear_btn = QPushButton("✕")
        self.clear_btn.setFixedWidth(70)
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background: #ffffff;
                border: 1px solid #d93025;
                border-radius: 4px;
                color: #d93025;
                font-size: 16px;
            }
            QPushButton:hover { background: #fce8e6; }
        """)
        self.clear_btn.clicked.connect(self.clear_files)

        browse_col.addWidget(browse_label, alignment=Qt.AlignHCenter)
        browse_col.addWidget(self.browse_btn, alignment=Qt.AlignHCenter)
        browse_col.addWidget(clear_label, alignment=Qt.AlignHCenter)
        browse_col.addWidget(self.clear_btn, alignment=Qt.AlignHCenter)
        browse_col.addItem(QSpacerItem(10, 10, QSizePolicy.Minimum, QSizePolicy.Expanding))

        drop_browse_layout.addLayout(browse_col, 0, 3, 1, 1)

        self.files_label = QLabel("No files selected")
        self.files_label.setStyleSheet("color: #5f6368; font-size: 11px;")
        self.files_label.setWordWrap(True)
        drop_browse_layout.addWidget(self.files_label, 1, 0, 1, 4)
        return drop_browse_frame

    def bottomInit(self):
        bottom_row = QHBoxLayout()
        self.outer.addLayout(bottom_row)

        bottom_row.addStretch(1)

        self.visualize_btn = self.visualizeButtonInit()

        bottom_row.addWidget(self.visualize_btn)

        bottom_row.addStretch(1)

    def visualizeButtonInit(self):
        visualize_btn = QPushButton("Visualize")
        visualize_btn.setCursor(Qt.PointingHandCursor)
        visualize_btn.setFixedSize(260, 48)
        visualize_btn.clicked.connect(self.visualize)
        return visualize_btn

    # ---------- File selection ----------
    def browse_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select file(s)",
            "",
            "All Files (*.*)",
        )
        if paths:
            self.add_files(paths)

    def add_files(self, paths: List[str]):
        self.selected_files.clear()
        self.current_epochs = None
        
        # de-dup + keep stable order
        for p in paths:
            ap = os.path.abspath(p)
            if ap not in self.selected_files:
                self.selected_files.append(ap)

        if self.selected_files:
            # show a compact summary, with the first few file names
            names = [os.path.basename(p) for p in self.selected_files]
            preview = ", ".join(names[:6])
            if len(names) > 6:
                preview += f" … (+{len(names) - 6} more)"
            self.files_label.setText(f"{len(self.selected_files)} file(s): {preview}")

            if self.current_epochs is None:
                try:
                    print("Loading first file to get channel names...")
                    self.current_epochs = read_epochs_eeglab_minimal(self.selected_files[0], verbose=False)

                    # Update sensor dropdown: "All Channels" first, then actual channel names
                    # Populate sensor list and update combo
                    self._all_sensors = ["All Channels"] + list(self.current_epochs.ch_names)
                    self.sensor_combo.clear()
                    self.sensor_combo.addItems(self._all_sensors)
                    self.sensor_combo.setCurrentText("All Channels")
                    print(f"Loaded {len(self.current_epochs.ch_names)} channels")
                except Exception as e:
                    print(f"Could not auto-load file: {e}")
            
            # Mark visualize button as needing update
            self.mark_visualize_button_needs_update()

        else:
            self.files_label.setText("No files selected")

    # ---------- Visualize stub ----------
    def mark_visualize_button_needs_update(self):
        """Mark the Visualize button as needing an update by turning it blue."""
        if not hasattr(self, 'visualize_btn'):
            return
        if self.is_dark_mode:
            self.visualize_btn.setStyleSheet(
                """
                QPushButton {
                    background: #8ab4f8;
                    border: 1px solid #8ab4f8;
                    border-radius: 4px;
                    font-size: 14px;
                    color: #000000;
                }
                QPushButton:hover { background: #669df6; }
                QPushButton:pressed { background: #4a8af5; }
                """
            )
        else:
            self.visualize_btn.setStyleSheet(
                """
                QPushButton {
                    background: #1a73e8;
                    border: 1px solid #1a73e8;
                    border-radius: 4px;
                    font-size: 14px;
                    color: white;
                }
                QPushButton:hover { background: #1666c1; }
                QPushButton:pressed { background: #1450b1; }
                """
            )

    def reset_visualize_button(self):
        """Reset the Visualize button to its default white state."""
        if not hasattr(self, 'visualize_btn'):
            return
        if self.is_dark_mode:
            self.visualize_btn.setStyleSheet(
                """
                QPushButton {
                    background: #202124;
                    border: 1px solid #5f6368;
                    border-radius: 4px;
                    font-size: 14px;
                    color: #e8eaed;
                }
                QPushButton:hover { background: #303134; }
                QPushButton:pressed { background: #3c4043; }
                """
            )
        else:
            self.visualize_btn.setStyleSheet(
                """
                QPushButton {
                    background: #ffffff;
                    border: 1px solid #202124;
                    border-radius: 4px;
                    font-size: 14px;
                    color: #202124;
                }
                QPushButton:hover { background: #f6f8fe; }
                QPushButton:pressed { background: #e8f0fe; }
                """
            )

    def on_events_checkbox_state_changed(self, state):
        """Track checkbox state (shared across Time Series and Joint Maps)."""
        self.events_checkbox_checked = self.events_checkbox.isChecked()

    def visualize(self):
        if not self.selected_files:
            QMessageBox.warning(self, "No Files", "Please select at least one .set file")
            return

        graph_type = self.graph_type_combo.currentText()

        opts = {
            "epoch_start": self.epoch_start.text().strip(),
            "epoch_end": self.epoch_end.text().strip(),
            "sensor": self.sensor_combo.currentText(),
            "graph_type": graph_type,
            "display_events_responses": self.events_checkbox.isChecked(),
            "live_mode": self.live_toggle.isChecked(),
            "files": list(self.selected_files),
        }

        try:
            # STEP 1: LOAD DATA (if not already loaded)
            if self.current_epochs is None:
                print(f"Loading {self.selected_files[0]}...")
                self.current_epochs = read_epochs_eeglab_minimal(self.selected_files[0], verbose=True)
                print(f"Loaded: {self.current_epochs}")

            epochs = self.current_epochs

            # STEP 2: PROCESS DATA

            # If topo, do not use the use specificed epoch times
            if graph_type != "Topographic Map" and (opts['epoch_start'] or opts['epoch_end']):
                try:
                    # Use the loaded epochs' actual limits as defaults
                    tmin = self.current_epochs.tmin
                    tmax = self.current_epochs.tmax

                    # Override with user input if provided
                    if opts['epoch_start']:
                        tmin = float(opts['epoch_start']) / 1000  # Convert ms to seconds
                    if opts['epoch_end']:
                        tmax = float(opts['epoch_end']) / 1000  # Convert ms to seconds

                    print(f"Selecting time window: {tmin} to {tmax} s")
                    epochs = select_time_window(epochs, tmin, tmax)
                except ValueError:
                    print("Invalid epoch times, using full range")

            # Select specific channel if needed (topomap/joint require all channels)
            channel_picks = None
            sensor_name = opts['sensor']
            needs_all_channels = graph_type in ("Topographic Map", "Joint Maps")
            if not needs_all_channels and sensor_name != "All Channels" and sensor_name in epochs.ch_names:
                channel_idx = epochs.ch_names.index(sensor_name)
                channel_picks = [channel_idx]
                print(f"Selected channel: {sensor_name}")

            # Average epochs to get evoked response
            print("Averaging epochs...")
            evoked = average_epochs(epochs, picks=channel_picks)
            print(f"Result: {evoked}")

            # STEP 3: VISUALIZE

            theme = "dark" if self.is_dark_mode else "light"

            if graph_type == "ErrP Time Series":
                fig = plot_evoked(
                    evoked,
                    window_title="ErrP Time Series",
                    display_events_responses=opts["display_events_responses"],
                    show=False,
                    theme=theme,
                )
            elif graph_type == "Topographic Map":
                times = self._parse_topomap_times()
                fig = plot_topomap(evoked, times=times, show=False, theme=theme)

            elif graph_type == "Joint Maps":
                times = self._parse_topomap_times()
                # out of range times handled in plot_joint
                fig = plot_joint(
                    evoked,
                    times=times,
                    title="ErrP Analysis",
                    display_events_responses=opts["display_events_responses"],
                    show=False,
                    theme=theme,
                )
            else:
                fig = plot_evoked(evoked, show=False, theme=theme)

            # STEP 4: EMBED IN GUI

            # Remove old canvas widget from layout
            graph_layout = self.graph_frame.layout()
            graph_layout.removeWidget(self.canvas)
            self.canvas.setParent(None)
            self.canvas.deleteLater()

            # Install new canvas bound correctly to the new figure
            self.figure = fig
            self.canvas = FigureCanvas(self.figure)
            graph_layout.addWidget(self.canvas, stretch=1)
            self.canvas.draw_idle()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Visualization failed:\n{str(e)}")
            print(f"Full error: {e}")
            import traceback
            traceback.print_exc()
        
        # Reset the button after successful visualization
        self.reset_visualize_button()

    def _parse_topomap_times(self) -> List[float]:
        """
        Parse the three topomap time fields (seconds). Empty means use default.
        Returns a list of 1-3 times in seconds; invalid input falls back to [0.1, 0.2, 0.3].
        """
        defaults = [0.1, 0.2, 0.3]
        widgets = [self.topo_time_1, self.topo_time_2, self.topo_time_3]
        times = []
        for i, w in enumerate(widgets):
            t = w.text().strip()
            if not t:
                times.append(defaults[i])
                continue
            try:
                times.append(float(t))
            except ValueError:
                return defaults
        return times if times else defaults

    def on_graph_type_changed(self, graph_type: str):
        """
        When graph type changes:
        - Show/hide the events checkbox based on graph type
        - Show/hide the topomap times inputs for Topographic Map / Joint Maps
        - Checkbox state is shared between Time Series and Joint Maps
        - Epoch window is DISABLED for Topographic Map (full range required)
        """
        # Determine if this graph type supports event highlighting
        supports_events = graph_type in ("ErrP Time Series", "Joint Maps")

        # Show/hide the checkbox container
        self.events_checkbox_container.setVisible(supports_events)

        # Show/hide topomap times (used for Topographic Map and Joint Maps)
        supports_topo_times = graph_type in ("Topographic Map", "Joint Maps")
        self.topo_times_container.setVisible(supports_topo_times)

        # Restore the shared checkbox state when showing
        if supports_events:
            self.events_checkbox.blockSignals(True)
            self.events_checkbox.setChecked(self.events_checkbox_checked)
            self.events_checkbox.blockSignals(False)

        # Disable epoch input for Topo Map
        is_topo_only = graph_type == "Topographic Map"
        self.epoch_start.setEnabled(not is_topo_only)
        self.epoch_end.setEnabled(not is_topo_only)
        if is_topo_only:
            # Clear any values the user had entered and show hint placeholders
            self.epoch_start.blockSignals(True)
            self.epoch_end.blockSignals(True)
            self.epoch_start.clear()
            self.epoch_end.clear()
            self.epoch_start.setPlaceholderText("Full range")
            self.epoch_end.setPlaceholderText("Full range")
            self.epoch_start.blockSignals(False)
            self.epoch_end.blockSignals(False)
            # Style as visually disabled so the user knows
            disabled_style = (
                "QLineEdit { background: #f1f3f4; color: #9aa0a6; "
                "border: 1px solid #dadce0; border-radius: 4px; }"
            )
            if self.is_dark_mode:
                disabled_style = (
                    "QLineEdit { background: #2d2d2d; color: #5f6368; "
                    "border: 1px solid #3c4043; border-radius: 4px; }"
                )
            self.epoch_start.setStyleSheet(disabled_style)
            self.epoch_end.setStyleSheet(disabled_style)
            # Also dim the label
            if hasattr(self, 'epoch_label'):
                self.epoch_label.setStyleSheet(
                    "color: #9aa0a6; font-size: 12px;"
                    if not self.is_dark_mode else
                    "color: #5f6368; font-size: 12px;"
                )
        else:
            # Restore normal epoch placeholders and styling
            self.epoch_start.setPlaceholderText("Start")
            self.epoch_end.setPlaceholderText("End")
            normal_style = (
                "QLineEdit { background: #ffffff; color: #202124; "
                "border: 1px solid #dadce0; border-radius: 4px; }"
            )
            if self.is_dark_mode:
                normal_style = (
                    "QLineEdit { background: #202124; color: #e8eaed; "
                    "border: 1px solid #5f6368; border-radius: 4px; }"
                )
            self.epoch_start.setStyleSheet(normal_style)
            self.epoch_end.setStyleSheet(normal_style)
            if hasattr(self, 'epoch_label'):
                self.epoch_label.setStyleSheet(
                    "color: #202124; font-size: 12px;"
                    if not self.is_dark_mode else
                    "color: #e8eaed; font-size: 12px;"
                )

        # Adjust sensor dropdown: force to All Channels for topo/joint and restore full list when returning
        if graph_type in ("Topographic Map", "Joint Maps"):
            # If coming from ErrP Time Series, remember its selected sensor
            if self._last_graph_type == "ErrP Time Series":
                self._last_time_series_selection = self.sensor_combo.currentText()

            # Show only All Channels (keep combo enabled so dropdown opens)
            self.sensor_combo.blockSignals(True)
            self.sensor_combo.clear()
            self.sensor_combo.addItem("All Channels")
            self.sensor_combo.setCurrentIndex(0)
            self.sensor_combo.setEnabled(True)
            self.sensor_combo.blockSignals(False)
        else:
            # Restore the full sensor list and the previously selected time-series sensor
            self.sensor_combo.blockSignals(True)
            self.sensor_combo.setEnabled(True)
            self.sensor_combo.clear()
            if self._all_sensors:
                self.sensor_combo.addItems(self._all_sensors)
                if self._last_time_series_selection in self._all_sensors:
                    try:
                        self.sensor_combo.setCurrentText(self._last_time_series_selection)
                    except Exception:
                        self.sensor_combo.setCurrentIndex(0)
                else:
                    self.sensor_combo.setCurrentIndex(0)
            else:
                self.sensor_combo.addItem("All Channels")
            self.sensor_combo.blockSignals(False)

        # Update last graph type
        self._last_graph_type = graph_type

    def clear_files(self):
        """Clear all selected files"""
        self.selected_files.clear()
        self.current_epochs = None
        self.files_label.setText("No files selected")
        self.sensor_combo.clear()
        self.sensor_combo.addItems(["All Channels"])
    
        # Clear the graph and restore the placeholder under the current theme
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            "Load data and click Visualize",
            ha="center",
            va="center",
            fontsize=16,
            color="#9aa0a6" if self.is_dark_mode else "#5f6368",
        )
        ax.axis("off")
        self.apply_current_mpl_theme_to_figure(self.figure)
        self.canvas.draw()
        
        # Reset the Visualize button to white (no files to visualize)
        self.reset_visualize_button()
    
        print("Files cleared")

    # ---------- Theme handling ----------

    def on_dark_mode_toggled(self, state: int) -> None:
        """Switch between light and dark themes for the entire GUI."""
        app = QApplication.instance()
        if app is None:
            return

        self.is_dark_mode = bool(state)

        if self.is_dark_mode:
            apply_dark_theme(app)
            self.apply_dark_styles()
        else:
            apply_light_theme(app)
            self.apply_light_styles()

        # Restyle existing Matplotlib figure to keep background/text consistent
        if self.figure is not None:
            self.apply_current_mpl_theme_to_figure(self.figure)
            self.canvas.draw_idle()

        # Re-apply epoch field state in case we're on Topographic Map
        self.on_graph_type_changed(self.graph_type_combo.currentText())

    def apply_light_styles(self) -> None:
        """Apply light-mode styles to widgets that use explicit stylesheets."""
        # Graph frame
        self.graph_frame.setStyleSheet(
            """
            QFrame {
                background: #ffffff;
                border: 1px solid #dadce0;
                border-radius: 4px;
            }
            """
        )

        # Drag/drop + browse frame
        if hasattr(self, "drop_browse_frame"):
            self.drop_browse_frame.setStyleSheet(
                """
                QFrame {
                    background: #ffffff;
                    border: 1px solid #dadce0;
                    border-radius: 4px;
                }
                """
            )
        # Drop zone interior
        if hasattr(self, "drop_zone"):
            self.drop_zone.set_dark_mode(False)

        # Group box title / label colors
        if hasattr(self, "options_box"):
            self.options_box.setStyleSheet(
                """
                QGroupBox {
                    font-size: 13px;
                    font-weight: 600;
                    color: #202124;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 4px 0 4px;
                    color: #202124;
                }
                """
            )

        # Labels
        for label in self.findChildren(QLabel):
            # Keep semantic differences (primary vs secondary text) roughly intact
            style = label.styleSheet() or ""
            style = style.replace("#e8eaed", "#202124").replace("#9aa0a6", "#5f6368")
            label.setStyleSheet(style)

        # Files label (secondary text)
        self.files_label.setStyleSheet("color: #5f6368; font-size: 11px;")

        # Text fields and combos
        for line_edit in self.findChildren(QLineEdit):
            line_edit.setStyleSheet(
                """
                QLineEdit {
                    background: #ffffff;
                    color: #202124;
                    border: 1px solid #dadce0;
                    border-radius: 4px;
                }
                """
            )
        for combo in self.findChildren(QComboBox):
            combo.setStyleSheet(
                """
                QComboBox {
                    background: #ffffff;
                    color: #202124;
                    border: 1px solid #dadce0;
                    border-radius: 4px;
                }
                QComboBox::drop-down {
                    subcontrol-origin: padding;
                    subcontrol-position: top right;
                    width: 18px;
                    border-left: 1px solid #dadce0;
                }
                """
            )

        # Events checkbox (indicator + text)
        self.events_checkbox.setStyleSheet(
            """
            QCheckBox {
                font-size: 12px;
                color: #202124;
            }
            """
        )

        # Clear button (destructive accent)
        self.clear_btn.setStyleSheet(
            """
            QPushButton {
                background: #ffffff;
                border: 1px solid #d93025;
                border-radius: 4px;
                color: #d93025;
                font-size: 16px;
            }
            QPushButton:hover { background: #fce8e6; }
            """
        )

        # Toggles
        self.live_toggle.set_dark_mode(False)
        self.dark_mode_toggle.set_dark_mode(False)

        # Visualize button baseline
        self.reset_visualize_button()

        # Re-apply epoch field disabled state if on Topographic Map
        if hasattr(self, 'graph_type_combo'):
            self.on_graph_type_changed(self.graph_type_combo.currentText())

    def apply_dark_styles(self) -> None:
        """Apply dark-mode styles to widgets that use explicit stylesheets."""
        # Graph frame
        self.graph_frame.setStyleSheet(
            """
            QFrame {
                background: #121212;
                border: 1px solid #3c4043;
                border-radius: 4px;
            }
            """
        )

        # Drag/drop + browse frame
        if hasattr(self, "drop_browse_frame"):
            self.drop_browse_frame.setStyleSheet(
                """
                QFrame {
                    background: #121212;
                    border: 1px solid #3c4043;
                    border-radius: 4px;
                }
                """
            )
        # Drop zone interior
        if hasattr(self, "drop_zone"):
            self.drop_zone.set_dark_mode(True)

        # Group box title / label colors
        if hasattr(self, "options_box"):
            self.options_box.setStyleSheet(
                """
                QGroupBox {
                    font-size: 13px;
                    font-weight: 600;
                    color: #e8eaed;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 4px 0 4px;
                    color: #e8eaed;
                }
                """
            )

        # Labels
        for label in self.findChildren(QLabel):
            style = label.styleSheet() or ""
            # Primary text becomes near-white, secondary text slightly dimmer
            style = style.replace("#202124", "#e8eaed").replace("#5f6368", "#9aa0a6")
            label.setStyleSheet(style)

        # Files label (secondary text)
        self.files_label.setStyleSheet("color: #9aa0a6; font-size: 11px;")

        # Text fields and combos
        for line_edit in self.findChildren(QLineEdit):
            line_edit.setStyleSheet(
                """
                QLineEdit {
                    background: #202124;
                    color: #e8eaed;
                    border: 1px solid #5f6368;
                    border-radius: 4px;
                }
                """
            )
        for combo in self.findChildren(QComboBox):
            combo.setStyleSheet(
                """
                QComboBox {
                    background: #202124;
                    color: #e8eaed;
                    border: 1px solid #5f6368;
                    border-radius: 4px;
                }
                QComboBox::drop-down {
                    subcontrol-origin: padding;
                    subcontrol-position: top right;
                    width: 18px;
                    border-left: 1px solid #5f6368;
                }
                """
            )

        # Events checkbox (indicator + text) for dark background
        self.events_checkbox.setStyleSheet(
            """
            QCheckBox {
                font-size: 12px;
                color: #e8eaed;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
            QCheckBox::indicator:unchecked {
                border-radius: 3px;
                border: 1px solid #9aa0a6;
                background: #202124;
            }
            QCheckBox::indicator:checked {
                border-radius: 3px;
                border: 1px solid #8ab4f8;
                background: #8ab4f8;
            }
            """
        )

        # Clear button (destructive accent on dark background)
        self.clear_btn.setStyleSheet(
            """
            QPushButton {
                background: #202124;
                border: 1px solid #f28b82;
                border-radius: 4px;
                color: #f28b82;
                font-size: 16px;
            }
            QPushButton:hover { background: #3c4043; }
            """
        )

        # Toggles
        self.live_toggle.set_dark_mode(True)
        self.dark_mode_toggle.set_dark_mode(True)

        # Visualize button baseline
        self.reset_visualize_button()

        # Reapply epoch field disabled state if on Topographic Map
        if hasattr(self, 'graph_type_combo'):
            self.on_graph_type_changed(self.graph_type_combo.currentText())

    def apply_current_mpl_theme_to_figure(self, fig: Figure) -> None:
        """
        Harmonize a Matplotlib figure with the current light/dark theme.
        """
        if fig is None:
            return

        if self.is_dark_mode:
            bg_color = "#121212"
            axis_bg = "#121212"
            text_color = "#e8eaed"
            grid_color = "#3c4043"
        else:
            bg_color = "#ffffff"
            axis_bg = "#ffffff"
            text_color = "#202124"
            grid_color = "#dadce0"

        fig.patch.set_facecolor(bg_color)

        if hasattr(fig, "_suptitle") and fig._suptitle is not None:
            fig._suptitle.set_color(text_color)

        for ax in fig.get_axes():
            ax.set_facecolor(axis_bg)
            ax.tick_params(colors=text_color)
            ax.xaxis.label.set_color(text_color)
            ax.yaxis.label.set_color(text_color)
            ax.title.set_color(text_color)
            ax.grid(color=grid_color, alpha=0.3)
            for spine in ax.spines.values():
                spine.set_color(grid_color)