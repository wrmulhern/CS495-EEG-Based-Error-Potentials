"""
flanker_window.py
Native Flanker Task with simultaneous Ganglion EEG recording.

Opens from FileWindow when the user clicks "Record EEG".
On completion it saves a CSV,
auto converts it to .set (averaging trials with the event occuring at 0ms), and signals the parent to open it as a new tab. 
From there, user presses 'visualize' to see the epoched ERPs.
"""

import os
import time
import random
import logging
import threading

from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QKeyEvent
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QFrame, QSpinBox, QFormLayout, QMessageBox,
    QProgressBar, QWidget, QStackedWidget,
)

from src.data_processing.eeg_recorder import (
    EEGRecorder,
    MARKER_CONGRUENT, MARKER_INCONGRUENT,
    MARKER_CORRECT, MARKER_ERROR, MARKER_NO_RESPONSE,
)

logger = logging.getLogger(__name__)

# The flanker stimuli, with their congruency and correct response direction
# User will be asked to respond to the CENTER arrow only, ignoring the flankers. If user is incorrect, 
# should trigger a perceived error and thus an ERN in the EEG.
STIMULI = [
    # (display string,   congruent, correct_direction)
    ("< < < < <",  True,  "left"),
    ("> > > > >",  True,  "right"),
    ("< < > < <",  False, "right"),
    ("> > < > >",  False, "left"),
]

# Timing (milliseconds)
FIXATION_MS   = 500
STIMULUS_MS   = 200
RESPONSE_MS   = 1000    # window after stimulus disappears
FEEDBACK_MS   = 400
ITI_MS        = 800     # inter-trial interval (blank)


#
# Worker signal bridge (so background threads can talk to Qt)
#
class _Signals(QObject):
    finished = pyqtSignal(str)   # emits CSV path when done
    error    = pyqtSignal(str)   # emits error message


#
# FlankerWindow - holds all the UI and logic for running the task and recording EEG.
#
class FlankerWindow(QDialog):
    """
    Fullscreen dialog that runs the Flanker task and records EEG.

    Signals
    -------
    recording_finished(str)
        Emitted with the path to the saved CSV when the task completes.
        The parent (FileWindow) connects this to add_files().
    """

    recording_finished = pyqtSignal(str)

    # Pages in the stacked widget
    _PAGE_SETUP   = 0 # choose which port to look for Ganglion, and how many trials to run
    _PAGE_INTRO   = 1 # directions for completing the flanker task
    _PAGE_TASK    = 2 # flanker task
    _PAGE_DONE    = 3 # shows stats and file path, with option to open in visualizer

    def __init__(self, is_dark: bool = False, output_dir: str = None, parent=None):
        super().__init__(parent)
        self.is_dark    = is_dark
        self.output_dir = output_dir or os.path.expanduser("~")

        self.setWindowTitle("Record EEG — Flanker Task")
        self.setMinimumSize(800, 600)
        self.resize(900, 650)
        self.setModal(True)

        # State 
        self._recorder: EEGRecorder | None = None
        self._signals   = _Signals()
        self._signals.finished.connect(self._on_recording_finished)
        self._signals.error.connect(self._on_recording_error)

        self._trial_index   = 0
        self._trials: list  = []
        self._response_key  = None   # "left",  "right",  None
        self._response_time = None
        self._awaiting_response = False
        self._results: list = []     # per trial dicts

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._on_timer)
        self._phase = None   # "fixation", "stimulus", "response", "feedback", "iti"

        # Build UI
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        self._stack = QStackedWidget()
        root.addWidget(self._stack)

        self._stack.addWidget(self._build_setup_page())    # 0
        self._stack.addWidget(self._build_intro_page())    # 1
        self._stack.addWidget(self._build_task_page())     # 2
        self._stack.addWidget(self._build_done_page())     # 3

        self._apply_theme(is_dark)
        self._stack.setCurrentIndex(self._PAGE_SETUP)

    #
    #
    # Build each page
    #
    #

    def _build_setup_page(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(60, 50, 60, 50)
        outer.setSpacing(20)

        title = QLabel("EEG Recording Setup")
        title.setStyleSheet("font-size: 22px; font-weight: 700;")
        outer.addWidget(title)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        outer.addWidget(sep)

        form = QFormLayout()
        form.setSpacing(14)
        form.setLabelAlignment(Qt.AlignRight)

        # Port selector
        self._port_combo = QComboBox()
        self._port_combo.setEditable(True)
        self._port_combo.setMinimumWidth(200)
        self._refresh_ports()

        refresh_btn = QPushButton("↺")
        refresh_btn.setFixedWidth(45)
        refresh_btn.setToolTip("Refresh port list")
        refresh_btn.clicked.connect(self._refresh_ports)

        port_row = QHBoxLayout()
        port_row.addWidget(self._port_combo)
        port_row.addWidget(refresh_btn)
        form.addRow("Serial port:", port_row)

        # Trial count
        self._trial_spin = QSpinBox()
        self._trial_spin.setRange(20, 400)
        self._trial_spin.setValue(100)
        self._trial_spin.setSingleStep(20)
        self._trial_spin.setSuffix("  trials")
        form.addRow("Number of trials:", self._trial_spin)

        # Output dir label
        self._output_lbl = QLabel(self.output_dir)
        self._output_lbl.setWordWrap(True)
        self._output_lbl.setStyleSheet("color: #5f6368; font-size: 11px;")
        form.addRow("Save to:", self._output_lbl)

        outer.addLayout(form)
        outer.addStretch(1)

        # Brainflow warning if not installed
        if not EEGRecorder.is_brainflow_available():
            warn = QLabel(
                "⚠  brainflow is not installed.\n"
                "Run:  uv add brainflow\n"
                "The task will run without EEG recording until then."
            )
            warn.setStyleSheet(
                "background: #fce8e6; color: #c5221f; border-radius: 6px;"
                " padding: 10px; font-size: 12px;"
            )
            warn.setWordWrap(True)
            outer.addWidget(warn)
            self._no_brainflow = True
        else:
            self._no_brainflow = False

        # Buttons
        btn_row = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        self._start_btn = QPushButton("Connect & Start →")
        self._start_btn.setFixedHeight(40)
        self._start_btn.clicked.connect(self._on_setup_start)
        btn_row.addWidget(cancel_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self._start_btn)
        outer.addLayout(btn_row)

        return page

    def _build_intro_page(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(80, 60, 80, 60)
        outer.setSpacing(24)
        outer.addStretch(1)

        heading = QLabel("Flanker Task")
        heading.setAlignment(Qt.AlignCenter)
        heading.setStyleSheet("font-size: 28px; font-weight: 700;")
        outer.addWidget(heading)

        instructions = QLabel(
            "You will see a row of five arrows.\n\n"
            "Press  ←  if the CENTER arrow points LEFT\n"
            "Press  →  if the CENTER arrow points RIGHT\n\n"
            "Respond as quickly and accurately as possible.\n"
            "Ignore the surrounding arrows.\n"
            "Remain as still as possible to get a good EEG recording."
        )
        instructions.setAlignment(Qt.AlignCenter)
        instructions.setStyleSheet("font-size: 16px; line-height: 1.6;")
        instructions.setWordWrap(True)
        outer.addWidget(instructions)

        outer.addSpacing(20)

        examples = QLabel("Examples:\n< < < < <   →  press ←\n> > < > >   →  press →")
        examples.setAlignment(Qt.AlignCenter)
        examples.setStyleSheet(
            "font-size: 18px; font-family: monospace; letter-spacing: 2px;"
            " background: rgba(128,128,128,0.1); border-radius: 8px; padding: 14px;"
        )
        outer.addWidget(examples)

        outer.addStretch(1)

        begin_btn = QPushButton("Begin Task  ▶")
        begin_btn.setFixedHeight(48)
        begin_btn.setStyleSheet(
            "QPushButton { background: #1a73e8; color: white; border-radius: 6px;"
            " font-size: 16px; font-weight: 600; }"
            "QPushButton:hover { background: #1557b0; }"
        )
        begin_btn.clicked.connect(self._begin_task)
        outer.addWidget(begin_btn, alignment=Qt.AlignHCenter)

        return page

    def _build_task_page(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)

        # Progress bar
        self._progress = QProgressBar()
        self._progress.setTextVisible(False)
        self._progress.setFixedHeight(6)
        outer.addWidget(self._progress)

        # Central display
        center = QVBoxLayout()
        center.addStretch(1)

        self._stimulus_lbl = QLabel("+")
        self._stimulus_lbl.setAlignment(Qt.AlignCenter)
        font = QFont("Courier", 52, QFont.Bold)
        self._stimulus_lbl.setFont(font)
        center.addWidget(self._stimulus_lbl)

        self._feedback_lbl = QLabel("")
        self._feedback_lbl.setAlignment(Qt.AlignCenter)
        self._feedback_lbl.setStyleSheet("font-size: 20px;")
        center.addWidget(self._feedback_lbl)

        center.addStretch(1)
        outer.addLayout(center)

        # Trial counter
        self._trial_counter_lbl = QLabel("")
        self._trial_counter_lbl.setAlignment(Qt.AlignCenter)
        self._trial_counter_lbl.setStyleSheet("font-size: 12px; color: #9aa0a6; padding: 10px;")
        outer.addWidget(self._trial_counter_lbl)

        # Abort button (small, bottom right)
        abort_row = QHBoxLayout()
        abort_row.addStretch(1)
        abort_btn = QPushButton("Abort")
        abort_btn.setStyleSheet("color: #c5221f; border: none; font-size: 11px;")
        abort_btn.clicked.connect(self._abort_task)
        abort_row.addWidget(abort_btn)
        outer.addLayout(abort_row)

        return page

    def _build_done_page(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(60, 60, 60, 60)
        outer.setSpacing(20)
        outer.addStretch(1)

        self._done_icon = QLabel("✓")
        self._done_icon.setAlignment(Qt.AlignCenter)
        self._done_icon.setStyleSheet("font-size: 64px; color: #34a853;")
        outer.addWidget(self._done_icon)

        self._done_title = QLabel("Recording Complete")
        self._done_title.setAlignment(Qt.AlignCenter)
        self._done_title.setStyleSheet("font-size: 24px; font-weight: 700;")
        outer.addWidget(self._done_title)

        self._done_stats = QLabel("")
        self._done_stats.setAlignment(Qt.AlignCenter)
        self._done_stats.setStyleSheet("font-size: 14px; color: #5f6368; line-height: 1.8;")
        self._done_stats.setWordWrap(True)
        outer.addWidget(self._done_stats)

        self._done_path = QLabel("")
        self._done_path.setAlignment(Qt.AlignCenter)
        self._done_path.setStyleSheet(
            "font-size: 11px; color: #5f6368; font-family: monospace;"
            " background: rgba(128,128,128,0.08); border-radius: 4px; padding: 8px;"
        )
        self._done_path.setWordWrap(True)
        outer.addWidget(self._done_path)

        outer.addStretch(1)

        btn_row = QHBoxLayout()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        self._open_btn = QPushButton("Open in Visualizer →")
        self._open_btn.setFixedHeight(40)
        self._open_btn.setStyleSheet(
            "QPushButton { background: #1a73e8; color: white; border-radius: 6px;"
            " font-size: 14px; font-weight: 600; }"
            "QPushButton:hover { background: #1557b0; }"
        )
        self._open_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self._open_btn)
        outer.addLayout(btn_row)

        return page

    #
    #
    # Helper functions for page actions
    #
    #

    def _refresh_ports(self):
        self._port_combo.clear()
        ports = EEGRecorder.list_ports()
        if ports:
            self._port_combo.addItems(ports)
        else:
            self._port_combo.addItem("COM3")   # sensible default, can be changed for windows users

    def _on_setup_start(self):
        port = self._port_combo.currentText().strip()
        if not port:
            QMessageBox.warning(self, "No port", "Please select or enter a serial port.")
            return

        n_trials = self._trial_spin.value()
        self._trials = self._build_trial_list(n_trials)
        self._trial_index = 0
        self._results = []

        if self._no_brainflow:
            # Run task without EEG
            self._recorder = None
            self._stack.setCurrentIndex(self._PAGE_INTRO)
            return

        # Connect in background so UI stays responsive
        self._start_btn.setEnabled(False)
        self._start_btn.setText("Connecting…")

        def _connect():
            try:
                rec = EEGRecorder(port)
                rec.start()
                self._recorder = rec
                self._signals.finished.emit("__connected__")
            except Exception as exc:
                self._signals.error.emit(str(exc))

        threading.Thread(target=_connect, daemon=True).start()

    #
    #
    # Trials
    #
    #

    @staticmethod
    def _build_trial_list(n: int) -> list:
        """
        Build a balanced, pseudo- andomised trial list.
        ~50% congruent, ~50% incongruent.
        Returns list of dicts with keys: stimulus, congruent, correct_direction.
        """
        base = STIMULI * (n // len(STIMULI) + 1)
        random.shuffle(base)
        trials = []
        for stim, cong, direction in base[:n]:
            trials.append({
                "stimulus":   stim,
                "congruent":  cong,
                "direction":  direction,
            })
        return trials

    #
    #
    # Task flow helper functions
    #
    #

    def _begin_task(self):
        self._stack.setCurrentIndex(self._PAGE_TASK)
        self._progress.setMaximum(len(self._trials))
        self._progress.setValue(0)
        self.setFocus()
        QTimer.singleShot(500, self._next_trial)

    def _next_trial(self):
        if self._trial_index >= len(self._trials):
            self._finish_task()
            return

        trial = self._trials[self._trial_index]
        self._response_key  = None
        self._response_time = None
        self._awaiting_response = False
        self._feedback_lbl.setText("")

        n = len(self._trials)
        self._trial_counter_lbl.setText(f"Trial {self._trial_index + 1} of {n}")
        self._progress.setValue(self._trial_index)

        self._phase = "fixation"
        self._stimulus_lbl.setText("+")
        self._timer.start(FIXATION_MS)

    def _on_timer(self):
        trial = self._trials[self._trial_index]

        if self._phase == "fixation":
            # Show stimulus + send marker
            self._phase = "stimulus"
            self._stimulus_lbl.setText(trial["stimulus"])
            marker = MARKER_CONGRUENT if trial["congruent"] else MARKER_INCONGRUENT
            if self._recorder:
                self._recorder.insert_marker(marker)
            self._stimulus_onset = time.time()
            self._awaiting_response = True
            self._timer.start(STIMULUS_MS)

        elif self._phase == "stimulus":
            # Stimulus disappears — open response window
            self._phase = "response"
            self._stimulus_lbl.setText("")
            self._timer.start(RESPONSE_MS)

        elif self._phase == "response":
            # Response window closed
            self._awaiting_response = False
            self._phase = "feedback"

            if self._response_key is None:
                # No response
                if self._recorder:
                    self._recorder.insert_marker(MARKER_NO_RESPONSE)
                self._record_result(trial, "no_response", None)
                self._feedback_lbl.setText("Too slow!")
                self._feedback_lbl.setStyleSheet("font-size: 20px; color: #f4a400;")
            # (correct/error feedback already shown in keypress handler)

            self._timer.start(FEEDBACK_MS)

        elif self._phase == "feedback":
            # ITI
            self._phase = "iti"
            self._feedback_lbl.setText("")
            self._stimulus_lbl.setText("")
            self._timer.start(ITI_MS)

        elif self._phase == "iti":
            self._trial_index += 1
            self._next_trial()

    def keyPressEvent(self, event: QKeyEvent):
        if not self._awaiting_response:
            return

        key = event.key()
        if key == Qt.Key_Left:
            self._handle_response("left")
        elif key == Qt.Key_Right:
            self._handle_response("right")

    def _handle_response(self, direction: str):
        if not self._awaiting_response:
            return
        self._awaiting_response = False
        self._response_key  = direction
        self._response_time = time.time() - self._stimulus_onset

        trial   = self._trials[self._trial_index]
        correct = (direction == trial["direction"])

        if correct:
            if self._recorder:
                self._recorder.insert_marker(MARKER_CORRECT)
            self._feedback_lbl.setText("✓")
            self._feedback_lbl.setStyleSheet("font-size: 28px; color: #34a853;")
            self._record_result(trial, "correct", self._response_time)
        else:
            if self._recorder:
                self._recorder.insert_marker(MARKER_ERROR)
            self._feedback_lbl.setText("✗")
            self._feedback_lbl.setStyleSheet("font-size: 28px; color: #ea4335;")
            self._record_result(trial, "error", self._response_time)

    def _record_result(self, trial: dict, outcome: str, rt):
        self._results.append({
            "trial":     self._trial_index + 1,
            "congruent": trial["congruent"],
            "direction": trial["direction"],
            "outcome":   outcome,
            "rt_ms":     round(rt * 1000) if rt is not None else None,
        })

    def _abort_task(self):
        reply = QMessageBox.question(
            self, "Abort?",
            "Stop the task and discard this recording?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._timer.stop()
            self._awaiting_response = False
            if self._recorder:
                try:
                    self._recorder.stop(self.output_dir)
                except Exception:
                    pass
                self._recorder = None
            self.reject()
 
    #
    #
    # Finish & saving helper functions
    #
    #

    def _finish_task(self):
        self._timer.stop()
        self._awaiting_response = False
        self._progress.setValue(len(self._trials))
        self._stimulus_lbl.setText("")
        self._feedback_lbl.setText("")

        if self._recorder:
            # Save in background thread
            def _save():
                try:
                    path = self._recorder.stop(self.output_dir)
                    self._signals.finished.emit(path)
                except Exception as exc:
                    self._signals.error.emit(str(exc))

            threading.Thread(target=_save, daemon=True).start()
        else:
            # No recorder — go straight to done page with no file
            self._show_done_page(csv_path=None)

    def _on_recording_finished(self, csv_path: str):
        if csv_path == "__connected__":
            self._start_btn.setEnabled(True)
            self._start_btn.setText("Connect & Start →")
            self._stack.setCurrentIndex(self._PAGE_INTRO)
            return
        self._csv_path = csv_path
        self._show_done_page(csv_path)
        self.recording_finished.emit(csv_path)

    def _on_recording_error(self, msg: str):
        QMessageBox.critical(
            self, "Save Failed",
            f"Could not save EEG recording:\n\n{msg}"
        )
        self.reject()

    def _show_done_page(self, csv_path):
        # Compute stats
        total   = len(self._results)
        correct = sum(1 for r in self._results if r["outcome"] == "correct")
        errors  = sum(1 for r in self._results if r["outcome"] == "error")
        no_resp = sum(1 for r in self._results if r["outcome"] == "no_response")
        rts     = [r["rt_ms"] for r in self._results if r["rt_ms"] is not None]
        avg_rt  = f"{sum(rts)/len(rts):.0f} ms" if rts else "—"
        acc     = f"{correct/total*100:.1f}%" if total else "—"

        self._done_stats.setText(
            f"Trials: {total}    Accuracy: {acc}    Avg RT: {avg_rt}\n"
            f"Correct: {correct}    Errors: {errors}    No response: {no_resp}"
        )

        if csv_path:
            self._done_path.setText(f"Saved to:\n{csv_path}")
            self._open_btn.setVisible(True)
        else:
            self._done_path.setText("(No EEG file — brainflow not available)")
            self._open_btn.setVisible(False)

        self._stack.setCurrentIndex(self._PAGE_DONE)

    #
    #
    # Themes
    #
    #

    def _apply_theme(self, dark: bool):
        if dark:
            self.setStyleSheet(
                "QDialog, QWidget { background: #1e1e1e; color: #e8eaed; }"
                "QLabel { color: #e8eaed; }"
                "QPushButton { background: #303134; color: #e8eaed;"
                " border: 1px solid #5f6368; border-radius: 4px; padding: 6px 16px; }"
                "QPushButton:hover { background: #3c4043; }"
                "QComboBox { background: #303134; color: #e8eaed; border: 1px solid #5f6368;"
                " border-radius: 4px; padding: 4px 8px; }"
                "QSpinBox  { background: #303134; color: #e8eaed; border: 1px solid #5f6368;"
                " border-radius: 4px; padding: 4px 8px; }"
                "QProgressBar { background: #303134; border: none; }"
                "QProgressBar::chunk { background: #1a73e8; }"
                "QFrame[frameShape='4'] { color: #3c4043; }"  # HLine
            )
        else:
            self.setStyleSheet(
                "QDialog, QWidget { background: #ffffff; color: #202124; }"
                "QLabel { color: #202124; }"
                "QPushButton { background: #ffffff; color: #202124;"
                " border: 1px solid #dadce0; border-radius: 4px; padding: 6px 16px; }"
                "QPushButton:hover { background: #f1f3f4; }"
                "QComboBox { background: #ffffff; color: #202124; border: 1px solid #dadce0;"
                " border-radius: 4px; padding: 4px 8px; }"
                "QSpinBox  { background: #ffffff; color: #202124; border: 1px solid #dadce0;"
                " border-radius: 4px; padding: 4px 8px; }"
                "QProgressBar { background: #f1f3f4; border: none; }"
                "QProgressBar::chunk { background: #1a73e8; }"
                "QFrame[frameShape='4'] { color: #dadce0; }"  # HLine
            )