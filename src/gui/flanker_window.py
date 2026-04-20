"""
Native Flanker Task dialog with simultaneous OpenBCI Ganglion EEG recording.

Opened from :class:`~src.gui.file_window.FileWindow` when the user
clicks **Record EEG**.  The dialog has four pages managed by a
``QStackedWidget``:

1. **Setup** — serial-port selector, trial-count spinner, output
   directory display, and a "Connect & Start" button.
2. **Intro** — task instructions explaining the Flanker paradigm.
3. **Task** — full-screen stimulus presentation driven by a
   single-shot ``QTimer`` state machine that cycles through the
   phases *fixation → stimulus → response → feedback → ITI* for
   each trial.  Arrow-key responses are captured via
   ``keyPressEvent``.  Event markers (congruent / incongruent /
   correct / error / no-response) are injected into the Brainflow
   stream in real time.
4. **Done** — accuracy / RT summary and a button to open the saved
   file in the visualiser.

On completion the recorder saves a ``.csv`` and emits
:pyqt:`recording_finished(str)` with the file path.  The parent
:class:`~src.gui.file_window.FileWindow` auto-converts the CSV to
``.set`` and opens it as a new tab.

If ``brainflow`` is not installed the task still runs (useful for
UI testing) but no EEG is recorded.
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
from src.gui.themes.colors import get_palette
from src.config import FLANKER, GANGLION

logger = logging.getLogger(__name__)

#: The four Flanker stimulus variants.  Each tuple is
#: ``(display_string, is_congruent, correct_direction)``.
#: The participant must respond to the **centre** arrow only;
#: incorrect responses are expected to elicit an ERN in the EEG.
STIMULI = [
    ("< < < < <",  True,  "left"),
    ("> > > > >",  True,  "right"),
    ("< < > < <",  False, "right"),
    ("> > < > >",  False, "left"),
]

FIXATION_MS   = FLANKER.fixation_ms
STIMULUS_MS   = FLANKER.stimulus_ms
RESPONSE_MS   = FLANKER.response_ms
FEEDBACK_MS   = FLANKER.feedback_ms
ITI_MS        = FLANKER.iti_ms


class _TrialSpinBox(QSpinBox):
    """QSpinBox that snaps arrow-button stepping onto {5, 20, 40, 60, …}.

    The minimum (5) is a low-end quick-test value; everything above
    steps by the configured ``singleStep``.  Arrowing up from 5 lands
    on 20 instead of ``5 + singleStep`` (e.g. 25); arrowing down from
    20 lands on 5 instead of clamping at 0.
    """

    def stepBy(self, steps: int) -> None:
        current = self.value()
        step    = self.singleStep()
        low     = self.minimum()

        if current == low and steps > 0:
            # Jump from the low-end value onto the regular grid at `step`,
            # then consume any remaining steps at the normal interval.
            self.setValue(step)
            if steps > 1:
                super().stepBy(steps - 1)
            return

        if current == step and steps < 0:
            # Step down from the lowest grid value back to the low-end value.
            self.setValue(low)
            return

        super().stepBy(steps)


class _Signals(QObject):
    """Thread-safe signal bridge so background workers can update the Qt UI.

    Attributes:
        finished(str): Emitted with the CSV path on success, or the
            sentinel ``"__connected__"`` after initial Ganglion handshake.
        error(str): Emitted with an error message on failure.
    """
    finished = pyqtSignal(str)
    error    = pyqtSignal(str)


class FlankerWindow(QDialog):
    """Modal dialog that runs the Flanker task with optional EEG recording.

    The dialog is a four-page ``QStackedWidget`` (see module docstring
    for page descriptions).  A single-shot ``QTimer`` drives the trial
    state machine; arrow-key presses are captured in
    :meth:`keyPressEvent`.

    Signals:
        recording_finished(str): Emitted with the path to the saved
            CSV when the task completes.  The parent
            :class:`~src.gui.file_window.FileWindow` connects this to
            :meth:`~src.gui.file_window.FileWindow.add_files`.

    Parameters:
        is_dark (bool): Current theme state.
        output_dir (str | None): Directory for saved CSVs (defaults to
            ``~/``).
        parent (QWidget | None): Parent widget.
    """

    recording_finished = pyqtSignal(str)

    _PAGE_SETUP   = 0  #: Port / trial-count configuration.
    _PAGE_INTRO   = 1  #: Task instructions.
    _PAGE_TASK    = 2  #: Live stimulus presentation.
    _PAGE_DONE    = 3  #: Results summary.

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

    def _build_setup_page(self) -> QWidget:
        """Page 0: serial port, trial count, output directory, and start button."""
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
        self._trial_spin = _TrialSpinBox()
        self._trial_spin.setRange(5, 400)
        self._trial_spin.setValue(FLANKER.default_n_trials)
        self._trial_spin.setSingleStep(20)
        self._trial_spin.setSuffix("  trials")
        form.addRow("Number of trials:", self._trial_spin)

        # Output dir label
        self._output_lbl = QLabel(self.output_dir)
        self._output_lbl.setWordWrap(True)
        _p = get_palette(self.is_dark)
        self._output_lbl.setStyleSheet(f"color: {_p.text_secondary}; font-size: 11px;")
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
                f"background: {_p.danger_bg}; color: {_p.danger}; border-radius: 6px;"
                f" padding: 10px; font-size: 12px;"
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
        """Page 1: Flanker task instructions and a "Begin Task" button."""
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
        _p = get_palette(self.is_dark)
        begin_btn.setStyleSheet(
            f"QPushButton {{ background: {_p.accent}; color: white; border-radius: 6px;"
            f" font-size: 16px; font-weight: 600; }}"
            f"QPushButton:hover {{ background: {_p.accent_hover}; }}"
        )
        begin_btn.clicked.connect(self._begin_task)
        outer.addWidget(begin_btn, alignment=Qt.AlignHCenter)

        return page

    def _build_task_page(self) -> QWidget:
        """Page 2: progress bar, central stimulus/feedback labels, and abort button."""
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
        _p = get_palette(self.is_dark)
        self._trial_counter_lbl.setStyleSheet(f"font-size: 12px; color: {_p.text_disabled}; padding: 10px;")
        outer.addWidget(self._trial_counter_lbl)

        # Abort button (small, bottom right)
        abort_row = QHBoxLayout()
        abort_row.addStretch(1)
        abort_btn = QPushButton("Abort")
        abort_btn.setStyleSheet(f"color: {_p.danger}; border: none; font-size: 11px;")
        abort_btn.clicked.connect(self._abort_task)
        abort_row.addWidget(abort_btn)
        outer.addLayout(abort_row)

        return page

    def _build_done_page(self) -> QWidget:
        """Page 3: completion icon, accuracy/RT stats, file path, and open button."""
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(60, 60, 60, 60)
        outer.setSpacing(20)
        outer.addStretch(1)

        _p = get_palette(self.is_dark)
        self._done_icon = QLabel("✓")
        self._done_icon.setAlignment(Qt.AlignCenter)
        self._done_icon.setStyleSheet(f"font-size: 64px; color: {_p.success};")
        outer.addWidget(self._done_icon)

        self._done_title = QLabel("Recording Complete")
        self._done_title.setAlignment(Qt.AlignCenter)
        self._done_title.setStyleSheet("font-size: 24px; font-weight: 700;")
        outer.addWidget(self._done_title)

        self._done_stats = QLabel("")
        self._done_stats.setAlignment(Qt.AlignCenter)
        self._done_stats.setStyleSheet(f"font-size: 14px; color: {_p.text_secondary}; line-height: 1.8;")
        self._done_stats.setWordWrap(True)
        outer.addWidget(self._done_stats)

        self._done_path = QLabel("")
        self._done_path.setAlignment(Qt.AlignCenter)
        self._done_path.setStyleSheet(
            f"font-size: 11px; color: {_p.text_secondary}; font-family: monospace;"
            f" background: rgba(128,128,128,0.08); border-radius: 4px; padding: 8px;"
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
            f"QPushButton {{ background: {_p.accent}; color: white; border-radius: 6px;"
            f" font-size: 14px; font-weight: 600; }}"
            f"QPushButton:hover {{ background: {_p.accent_hover}; }}"
        )
        self._open_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self._open_btn)
        outer.addLayout(btn_row)

        return page

    def _refresh_ports(self):
        """Re-populate the serial-port combo from ``EEGRecorder.list_ports()``."""
        self._port_combo.clear()
    
        try:
            import serial.tools.list_ports
            ports = list(serial.tools.list_ports.comports())
            if ports:
                for port in ports:
                    # Shows e.g. "COM4 — OpenBCI Ganglion" or "/dev/cu.usbmodem1 — USB Serial"
                    description = port.description or "Unknown device"
                    display = f"{port.device} — {description}"
                    self._port_combo.addItem(display, userData=port.device)
            else:
                self._port_combo.addItem(GANGLION.default_port, userData=GANGLION.default_port)
        except ImportError:
            # Fall back to EEGRecorder.list_ports() if pyserial not directly available
            ports = EEGRecorder.list_ports()
            if ports:
                self._port_combo.addItems(ports)
            else:
                self._port_combo.addItem(GANGLION.default_port)

    def _on_setup_start(self):
        # Get actual port device from userData, fall back to text if not set
        port = self._port_combo.currentData() or self._port_combo.currentText().strip()
        if " — " in port:  # user typed a full display string manually
            port = port.split(" — ")[0].strip()
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

    @staticmethod
    def _build_trial_list(n: int) -> list:
        """Build a balanced, pseudo-randomised trial list.

        Repeats the four :data:`STIMULI` variants to reach *n* trials,
        then shuffles.  Approximately 50 % congruent / 50 % incongruent.

        Returns:
            list[dict]: Each dict has keys ``stimulus`` (str),
            ``congruent`` (bool), and ``direction`` (``"left"`` or
            ``"right"``).
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

    def _begin_task(self):
        """Switch to the task page and kick off the first trial after a short delay."""
        self._stack.setCurrentIndex(self._PAGE_TASK)
        self._progress.setMaximum(len(self._trials))
        self._progress.setValue(0)
        self.setFocus()
        QTimer.singleShot(FLANKER.initial_delay_ms, self._next_trial)

    def _next_trial(self):
        """Reset per-trial state, show the fixation cross, and start the timer."""
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
        """Single-shot timer callback: advance through the phase state machine."""
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
                _p = get_palette(self.is_dark)
                self._feedback_lbl.setStyleSheet(f"font-size: 20px; color: {_p.warning};")
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
        """Capture left / right arrow keys during the response window."""
        if not self._awaiting_response:
            return

        key = event.key()
        if key == Qt.Key_Left:
            self._handle_response("left")
        elif key == Qt.Key_Right:
            self._handle_response("right")

    def _handle_response(self, direction: str):
        """Score the response, insert an EEG marker, and show feedback."""
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
            _p = get_palette(self.is_dark)
            self._feedback_lbl.setText("✓")
            self._feedback_lbl.setStyleSheet(f"font-size: 28px; color: {_p.success};")
            self._record_result(trial, "correct", self._response_time)
        else:
            if self._recorder:
                self._recorder.insert_marker(MARKER_ERROR)
            _p = get_palette(self.is_dark)
            self._feedback_lbl.setText("✗")
            self._feedback_lbl.setStyleSheet(f"font-size: 28px; color: {_p.error};")
            self._record_result(trial, "error", self._response_time)

    def _record_result(self, trial: dict, outcome: str, rt):
        """Append a per-trial result dict for the end-of-task summary."""
        self._results.append({
            "trial":     self._trial_index + 1,
            "congruent": trial["congruent"],
            "direction": trial["direction"],
            "outcome":   outcome,
            "rt_ms":     round(rt * 1000) if rt is not None else None,
        })

    def _abort_task(self):
        """Confirm, then stop the timer, release the recorder, and close the dialog."""
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
 
    def _finish_task(self):
        """Stop the timer and save the EEG data in a background thread."""
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
        """Handle the ``_Signals.finished`` signal (connection ack or saved CSV)."""
        if csv_path == "__connected__":
            self._start_btn.setEnabled(True)
            self._start_btn.setText("Connect & Start →")
            self._stack.setCurrentIndex(self._PAGE_INTRO)
            return
        self._csv_path = csv_path
        self._show_done_page(csv_path)
        self.recording_finished.emit(csv_path)

    def _on_recording_error(self, msg: str):
        """Show a critical message box and close the dialog on save failure."""
        QMessageBox.critical(
            self, "Save Failed",
            f"Could not save EEG recording:\n\n{msg}"
        )
        self.reject()

    def _show_done_page(self, csv_path):
        """Compute accuracy / RT stats and switch to the Done page."""
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

    def _apply_theme(self, dark: bool):
        """Set global light or dark stylesheet on the dialog and all child widgets."""
        p = get_palette(dark)
        accent_raw = "#1a73e8"  # progress bar accent is always the light-mode blue
        self.setStyleSheet(
            f"QDialog, QWidget {{ background: {p.surface}; color: {p.text}; }}"
            f"QLabel {{ color: {p.text}; }}"
            f"QPushButton {{ background: {p.surface_elevated}; color: {p.text};"
            f" border: 1px solid {p.border}; border-radius: 4px; padding: 6px 16px; }}"
            f"QPushButton:hover {{ background: {p.surface_hover}; }}"
            f"QComboBox {{ background: {p.surface_elevated}; color: {p.text}; border: 1px solid {p.border};"
            f" border-radius: 4px; padding: 4px 8px; }}"
            f"QSpinBox  {{ background: {p.surface_elevated}; color: {p.text}; border: 1px solid {p.border};"
            f" border-radius: 4px; padding: 4px 8px; }}"
            f"QProgressBar {{ background: {p.surface_elevated}; border: none; }}"
            f"QProgressBar::chunk {{ background: {accent_raw}; }}"
            f"QFrame[frameShape='4'] {{ color: {p.border_strong}; }}"
        )