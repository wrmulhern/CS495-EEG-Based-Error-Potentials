import os
from typing import List

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

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from data_loader import read_epochs_eeglab_minimal
from data_processor import average_epochs, select_time_window
from visualizer import plot_evoked, plot_topomap, plot_joint


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

    def dropEvent(self, event):
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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ErrP Visualizer")
        self.resize(1200, 720)

        self.selected_files: List[str] = []
        self.current_epochs = None

        central = QWidget()
        self.setCentralWidget(central)

        outer = QVBoxLayout(central)
        outer.setContentsMargins(18, 18, 18, 18)
        outer.setSpacing(14)

        # --- Top: Graph area (left) + Graph Options (right)
        top_row = QHBoxLayout()
        top_row.setSpacing(18)
        outer.addLayout(top_row, stretch=1)

        # Graph area (placeholder)
        self.graph_frame = QFrame()
        self.graph_frame.setFrameShape(QFrame.StyledPanel)
        self.graph_frame.setStyleSheet(
            """
            QFrame {
                background: #ffffff;
                border: 1px solid #dadce0;
                border-radius: 4px;
            }
            """
        )
        graph_layout = QVBoxLayout(self.graph_frame)
        graph_layout.setContentsMargins(10, 10, 10, 10)

        # Create matplotlib figure and canvas
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)

        # Initial placeholder
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, 'Load data and click Visualize',
                ha='center', va='center', fontsize=16, color='#5f6368')
        ax.axis('off')

        graph_layout.addWidget(self.canvas, stretch=1)

        top_row.addWidget(self.graph_frame, stretch=3)

        # Graph Options panel
        options_box = QGroupBox("Graph Options")
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

        # Epoch inputs row
        epoch_label = QLabel("Epoch in ms")
        epoch_label.setStyleSheet("color: #202124; font-size: 12px;")

        epoch_row = QHBoxLayout()
        epoch_row.setSpacing(10)

        self.epoch_start = QLineEdit()
        self.epoch_start.setPlaceholderText("Start")
        self.epoch_start.setFixedWidth(110)

        self.epoch_end = QLineEdit()
        self.epoch_end.setPlaceholderText("End")
        self.epoch_end.setFixedWidth(110)

        dash = QLabel("—")
        dash.setAlignment(Qt.AlignCenter)
        dash.setFixedWidth(16)

        epoch_row.addWidget(self.epoch_start)
        epoch_row.addWidget(dash)
        epoch_row.addWidget(self.epoch_end)
        epoch_row.addStretch(1)

        options_layout.addWidget(epoch_label)
        options_layout.addLayout(epoch_row)

        # Sensor dropdown
        sensor_label = QLabel("Sensor")
        sensor_label.setStyleSheet("color: #202124; font-size: 12px;")

        self.sensor_combo = QComboBox()
        self.sensor_combo.addItems(["Sensor A", "Sensor B", "Sensor C"])  # placeholder
        self.sensor_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        options_layout.addWidget(sensor_label)
        options_layout.addWidget(self.sensor_combo)

        # Graph Type dropdown
        graph_type_label = QLabel("Graph Type")
        graph_type_label.setStyleSheet("color: #202124; font-size: 12px;")

        self.graph_type_combo = QComboBox()
        self.graph_type_combo.addItems(["Line", "Scatter", "Bar"])  # placeholder

        options_layout.addWidget(graph_type_label)
        options_layout.addWidget(self.graph_type_combo)

        # Checkbox
        self.events_checkbox = QCheckBox("Display Events and Responses")
        self.events_checkbox.setStyleSheet("font-size: 12px; color: #202124;")
        options_layout.addWidget(self.events_checkbox)

        options_layout.addStretch(1)

        top_row.addWidget(options_box, stretch=1)

        # --- Middle: Live mode + Drop zone + Browse
        mid_row = QHBoxLayout()
        mid_row.setSpacing(18)
        outer.addLayout(mid_row)

        # Live mode toggle area
        live_col = QVBoxLayout()
        live_label = QLabel("Live mode")
        live_label.setStyleSheet("font-size: 13px; color: #202124;")
        self.live_toggle = ToggleSwitch("")
        live_col.addWidget(live_label)
        live_col.addWidget(self.live_toggle)
        live_col.addStretch(1)

        mid_row.addLayout(live_col, stretch=1)

        # Drag/drop + Browse group (mimics the wide box)
        drop_browse_frame = QFrame()
        drop_browse_frame.setFrameShape(QFrame.StyledPanel)
        drop_browse_frame.setStyleSheet(
            """
            QFrame {
                background: #ffffff;
                border: 1px solid #dadce0;
                border-radius: 4px;
            }
            """
        )
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
        self.browse_btn = QPushButton("…")
        self.browse_btn.setFixedWidth(70)
        self.browse_btn.clicked.connect(self.browse_files)

        browse_col.addWidget(browse_label, alignment=Qt.AlignHCenter)
        browse_col.addWidget(self.browse_btn, alignment=Qt.AlignHCenter)
        browse_col.addItem(QSpacerItem(10, 10, QSizePolicy.Minimum, QSizePolicy.Expanding))

        drop_browse_layout.addLayout(browse_col, 0, 3, 1, 1)

        self.files_label = QLabel("No files selected")
        self.files_label.setStyleSheet("color: #5f6368; font-size: 11px;")
        self.files_label.setWordWrap(True)
        drop_browse_layout.addWidget(self.files_label, 1, 0, 1, 4)

        mid_row.addWidget(drop_browse_frame, stretch=2)
        mid_row.addStretch(1)

        # --- Bottom: Visualize button
        bottom_row = QHBoxLayout()
        outer.addLayout(bottom_row)

        bottom_row.addStretch(1)

        self.visualize_btn = QPushButton("Visualize")
        self.visualize_btn.setCursor(Qt.PointingHandCursor)
        self.visualize_btn.setFixedSize(260, 48)
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
        self.visualize_btn.clicked.connect(self.visualize)
        bottom_row.addWidget(self.visualize_btn)

        bottom_row.addStretch(1)

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

                # Update sensor dropdown with actual channel names
                self.sensor_combo.clear()
                self.sensor_combo.addItems(self.current_epochs.ch_names)
                print(f"Loaded {len(self.current_epochs.ch_names)} channels")
            except Exception as e:
                print(f"Could not auto-load file: {e}")

        else:
            self.files_label.setText("No files selected")

    # ---------- Visualize stub ----------
    def visualize(self):
        if not self.selected_files:
            QMessageBox.warning(self, "No Files", "Please select at least one .set file")
            return

        opts = {
            "epoch_start": self.epoch_start.text().strip(),
            "epoch_end": self.epoch_end.text().strip(),
            "sensor": self.sensor_combo.currentText(),
            "graph_type": self.graph_type_combo.currentText(),
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

            # Apply time window filter if specified
            if opts['epoch_start'] and opts['epoch_end']:
                try:
                    tmin = float(opts['epoch_start']) / 1000  # Convert ms to seconds
                    tmax = float(opts['epoch_end']) / 1000
                    print(f"Selecting time window: {tmin} to {tmax} s")
                    epochs = select_time_window(epochs, tmin, tmax)
                except ValueError:
                    print("Invalid epoch times, using full range")

            # Select specific channel if needed
            channel_picks = None
            sensor_name = opts['sensor']
            if sensor_name != "Sensor A" and sensor_name in epochs.ch_names:  # Update dropdown values later
                channel_idx = epochs.ch_names.index(sensor_name)
                channel_picks = [channel_idx]
                print(f"Selected channel: {sensor_name}")

            # Average epochs to get evoked response
            print("Averaging epochs...")
            evoked = average_epochs(epochs, picks=channel_picks)
            print(f"Result: {evoked}")

            # STEP 3: VISUALIZE

            graph_type = opts['graph_type']

            if graph_type == "Line":
                fig = plot_evoked(evoked, window_title="ErrP Time Series", show=False)
            elif graph_type == "Scatter":  # Using "Scatter" for topomaps
                times = [0.1, 0.2, 0.3]  # Default times in seconds
                fig = plot_topomap(evoked, times=times, show=False)
            elif graph_type == "Bar":  # Using "Bar" for joint plot
                fig = plot_joint(evoked, title="ErrP Analysis", show=False)
            else:
                fig = plot_evoked(evoked, show=False)

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


def apply_light_theme(app: QApplication) -> None:
    app.setStyle("Fusion")

    pal = QPalette()

    pal.setColor(QPalette.Window, QColor(245, 245, 245))
    pal.setColor(QPalette.WindowText, QColor(32, 33, 36))

    pal.setColor(QPalette.Base, QColor(255, 255, 255))
    pal.setColor(QPalette.AlternateBase, QColor(240, 240, 240))

    pal.setColor(QPalette.Text, QColor(32, 33, 36))
    pal.setColor(QPalette.Button, QColor(255, 255, 255))
    pal.setColor(QPalette.ButtonText, QColor(32, 33, 36))

    pal.setColor(QPalette.Highlight, QColor(26, 115, 232))
    pal.setColor(QPalette.HighlightedText, QColor(255, 255, 255))

    app.setPalette(pal)

def main():
    app = QApplication([])
    apply_light_theme(app)
    w = MainWindow()
    w.show()
    app.exec_()


if __name__ == "__main__":
    main()
