"""
Thread-safe EEG streaming and recording via an OpenBCI Ganglion dongle.

This module wraps the `Brainflow <https://brainflow.org>`_ library to
provide a simple start / insert_marker / stop lifecycle for recording
EEG during the Flanker task (see
:class:`~src.gui.flanker_window.FlankerWindow`).

Hardware assumptions
~~~~~~~~~~~~~~~~~~~~
* **Board**: OpenBCI Ganglion (``BoardIds.GANGLION_NATIVE_BOARD``).
* **Channels**: 4 EEG channels in BrainFlow columns 1–4 (µV).
* **Sample rate**: 200 Hz native.
* **Markers**: Injected as floats into BrainFlow column 14.

The public entry point is :class:`EEGRecorder`.  All marker constants
are module-level integers that match the event codes used by
:mod:`~src.gui.flanker_window` and the CSV→.set converter in
:meth:`~src.gui.file_window.FileWindow.convert_ganglion_csv_to_set`.

``brainflow`` and ``pyserial`` are optional dependencies — the rest of
the application functions without them (the Flanker task runs in
"no-EEG" mode and :meth:`EEGRecorder.list_ports` returns an empty
list).
"""

import threading
import time
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

MARKER_CONGRUENT     = 1   #: Congruent stimulus onset  (``< < < < <`` or ``> > > > >``).
MARKER_INCONGRUENT   = 2   #: Incongruent stimulus onset (``< < > < <`` or ``> > < > >``).
MARKER_CORRECT       = 3   #: Participant responded correctly.
MARKER_ERROR         = 4   #: Participant responded incorrectly (expected to elicit ERN).
MARKER_NO_RESPONSE   = 5   #: Response window expired with no keypress.


class EEGRecorder:
    """Manages a single BrainFlow Ganglion streaming session.

    Typical usage inside :class:`~src.gui.flanker_window.FlankerWindow`::

        recorder = EEGRecorder(port="/dev/cu.usbmodem1")
        recorder.start()                          # opens the stream
        recorder.insert_marker(MARKER_CONGRUENT)  # stamp trial onset
        ...
        csv_path = recorder.stop(output_dir)      # saves & returns path

    All public methods are thread-safe (guarded by an internal lock).

    Class Attributes:
        SFREQ (int): Ganglion native sample rate (200 Hz).
        N_CHANNELS (int): Number of EEG channels (4).
        EEG_COLS (list[int]): BrainFlow column indices for EEG data.
        TS_COL (int): BrainFlow column index for the Unix timestamp.
        MARKER_COL (int): BrainFlow column index for event markers.
    """

    SFREQ       = 200
    N_CHANNELS  = 4
    EEG_COLS    = [1, 2, 3, 4]
    TS_COL      = 13
    MARKER_COL  = 14

    def __init__(self, port: str):
        """
        Parameters:
            port (str): Serial port for the Ganglion dongle, e.g.
                ``"COM3"`` (Windows) or ``"/dev/cu.usbmodem1"`` (macOS).
        """
        self.port = port
        self._board = None
        self._lock  = threading.Lock()
        self._running = False

    def start(self):
        """Connect to the Ganglion and begin streaming.

        Raises:
            RuntimeError: If the board cannot be reached on the
                configured serial port.
        """
        try:
            from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
            from brainflow.data_filter import DataFilter

            BoardShim.enable_dev_board_logger()

            params = BrainFlowInputParams()
            params.serial_port = self.port

            self._board = BoardShim(BoardIds.GANGLION_NATIVE_BOARD, params)
            self._board.prepare_session()
            self._board.start_stream(45000)   # ring buffer: 45 000 samples (~225 s)
            self._running = True
            logger.info(f"EEG stream started on {self.port}")

        # cant connect
        except Exception as exc:
            self._running = False
            raise RuntimeError(f"Could not connect to Ganglion on {self.port}: {exc}") from exc

    def insert_marker(self, value: float):
        """Stamp an event marker into the live EEG stream (thread-safe).

        No-op if the recorder is not running.  Silently logs a warning
        if the underlying BrainFlow call fails.

        Parameters:
            value (float): One of the ``MARKER_*`` constants.
        """
        if not self._running or self._board is None:
            return
        with self._lock:
            try:
                self._board.insert_marker(float(value))
            except Exception as exc:
                logger.warning(f"insert_marker failed: {exc}")

    def stop(self, output_dir: str) -> str:
        """Stop streaming, pull all buffered data, and save to CSV.

        The CSV has 15 columns matching BrainFlow's Ganglion layout:
        column 0 = sample index, 1–4 = EEG (µV), 5–12 = zeros,
        13 = Unix timestamp, 14 = marker.

        The filename is timestamped:
        ``flanker_eeg_YYYYMMDD_HHMMSS.csv``.

        Parameters:
            output_dir (str): Directory for the output file (created
                if it does not exist).

        Returns:
            str: Absolute path to the saved CSV.

        Raises:
            RuntimeError: If the recorder was never started.
        """
        if not self._running or self._board is None:
            raise RuntimeError("Recorder was not started.")

        with self._lock:
            self._running = False
            data = self._board.get_board_data()   # shape: (n_cols, n_samples)
            self._board.stop_stream()
            self._board.release_session()
            self._board = None

        logger.info(f"Captured {data.shape[1]} samples ({data.shape[1]/self.SFREQ:.1f}s)")

        # Build CSV
        # col 0 = sample index, cols 1-4 = EEG (µV), cols 5-12 = zeros,
        # col 13 = timestamp, col 14 = marker (aka the events (correct, error, etc))
        n_samples = data.shape[1]
        rows = np.zeros((n_samples, 15))

        rows[:, 0]     = np.arange(n_samples)          # sample index
        rows[:, 1:5]   = data[self.EEG_COLS, :].T      # EEG channels (µV)
        rows[:, 13]    = data[self.TS_COL, :]           # unix timestamp
        rows[:, 14]    = data[self.MARKER_COL, :]       # markers

        # Build filename
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_path = Path(output_dir) / f"flanker_eeg_{ts}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Header row
        header = ",".join(str(i) for i in range(15))
        np.savetxt(
            str(output_path),
            rows,
            delimiter=",",
            header=header,
            comments="",
            fmt="%.10g",
        )

        logger.info(f"EEG saved to {output_path}")
        return str(output_path)

    @property
    def is_running(self) -> bool:
        """``True`` while the stream is active."""
        return self._running

    @staticmethod
    def list_ports() -> list:
        """Return available serial-port device names via ``pyserial``.

        Returns an empty list if ``pyserial`` is not installed.
        """
        try:
            import serial.tools.list_ports
            return [p.device for p in serial.tools.list_ports.comports()]
        except ImportError:
            return []

    @staticmethod
    def is_brainflow_available() -> bool:
        """Check whether the ``brainflow`` package can be imported."""
        try:
            import brainflow  # noqa: F401
            return True
        except ImportError:
            return False