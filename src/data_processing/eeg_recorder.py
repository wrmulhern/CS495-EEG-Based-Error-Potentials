"""
eeg_recorder.py
Streams EEG data from an OpenBCI Ganglion via Brainflow.
Runs in a background thread; the main thread calls insert_marker()
to stamp events, then stop() to finalize and save.

Hardcoded to work with ganglion openBCI and 4 channels (col 1-4 in Brainflow data). Markers are floats in col 14.
Uses brainflow for streaming and CSV export.
Uses pyserial to get available ports for user dropdown.

Saves csv locally.
"""

import threading
import time
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

#
#
# Constants for event markers (must be floats for Brainflow)
#
#
MARKER_CONGRUENT     = 1   # congruent stimulus onset   (<<<  or  >>>)
MARKER_INCONGRUENT   = 2   # incongruent stimulus onset (<<>  or  >><)

MARKER_CORRECT       = 3   # correct response
MARKER_ERROR         = 4   # error response

MARKER_NO_RESPONSE   = 5   # timeout / no keypress


class EEGRecorder:
    """
    Manages a Brainflow Ganglion session.

    Usage
    -----
        recorder = EEGRecorder(port="COM3")
        recorder.start()                     # begins streaming
        recorder.insert_marker(MARKER_CONGRUENT)
        ...
        path = recorder.stop(output_dir)     # saves CSV, returns path
    """

    SFREQ       = 200          # Ganglion native sample rate (Hz)
    N_CHANNELS  = 4            # EEG channels
    # BrainFlow column indices for GANGLION_BOARD
    EEG_COLS    = [1, 2, 3, 4]
    TS_COL      = 13
    MARKER_COL  = 14

    def __init__(self, port: str):
        """
        Parameters
        ----------
        port : str
            Serial port for the Ganglion dongle, e.g. "COM3" or "/dev/ttyUSB0".
        """
        self.port = port
        self._board = None
        self._lock  = threading.Lock()
        self._running = False

    #
    #
    # Public API - streams data from ganglion
    #
    #

    # once user presses 'connect'
    def start(self):
        """Connect to the Ganglion and begin streaming."""
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
        """Stamp an event into the EEG stream (thread safe)."""
        if not self._running or self._board is None:
            return
        with self._lock:
            try:
                self._board.insert_marker(float(value))
            except Exception as exc:
                logger.warning(f"insert_marker failed: {exc}")

    def stop(self, output_dir: str) -> str:
        """
        Stop streaming, pull all data, save to CSV, return the CSV path.

        Parameters
        ----------
        output_dir : str
            Directory where the CSV will be written.

        Returns
        -------
        str
            Absolute path to the saved CSV file.
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
        return self._running

    #
    #
    # Static helpers
    # 
    #

    @staticmethod
    def list_ports() -> list:
        """Return available serial port names."""
        try:
            import serial.tools.list_ports
            return [p.device for p in serial.tools.list_ports.comports()]
        except ImportError:
            return []

    @staticmethod
    def is_brainflow_available() -> bool:
        try:
            import brainflow  # noqa: F401
            return True
        except ImportError:
            return False