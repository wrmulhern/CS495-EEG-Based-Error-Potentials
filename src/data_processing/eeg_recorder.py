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

from src.config import GANGLION, MARKERS, RECORDER

logger = logging.getLogger(__name__)

MARKER_CONGRUENT     = MARKERS.congruent
MARKER_INCONGRUENT   = MARKERS.incongruent
MARKER_CORRECT       = MARKERS.correct
MARKER_ERROR         = MARKERS.error
MARKER_NO_RESPONSE   = MARKERS.no_response


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

    SFREQ       = GANGLION.sfreq
    N_CHANNELS  = GANGLION.n_channels
    EEG_COLS    = list(GANGLION.eeg_cols)
    TS_COL      = GANGLION.ts_col
    MARKER_COL  = GANGLION.marker_col

    def __init__(self, port: str):
        """
        Parameters:
            port (str): Serial port for the Ganglion dongle, e.g.
                ``"COM3"`` (Windows) or ``"/dev/cu.usbmodem1"`` (macOS).
        """
        # Port validation: ensure it's a valid string and in available ports
        if not isinstance(port, str) or not port.strip():
            raise ValueError("Port must be a non-empty string")
        
        available_ports = self.list_ports()
        if port not in available_ports:
            raise ValueError(f"Port '{port}' is not available. Available ports: {available_ports}")
        
        self.port = port
        self._board = None
        self._lock  = threading.Lock()
        self._running = False

    def start(self):
        """Connect to the Ganglion and begin streaming."""
        try:
            from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

            BoardShim.enable_dev_board_logger()

            params = BrainFlowInputParams()
            params.serial_port = self.port

            self._board = BoardShim(BoardIds.GANGLION_NATIVE_BOARD, params)
            self._board.prepare_session()
            self._board.start_stream(GANGLION.ring_buffer_samples)
            self._running = True
            logger.info(f"EEG stream started on {self.port}")

        except Exception as exc:
            self._running = False
            if self._board is not None:
                try:
                    self._board.release_session()
                except Exception:
                    pass
            self._board = None
            logger.error(f"Failed to start EEG stream on {self.port}: {exc}")
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
            try:
                data = self._board.get_board_data()   # shape: (n_cols, n_samples)
                self._board.stop_stream()
                self._board.release_session()
                self._board = None
            except Exception as exc:
                logger.error(f"Error stopping EEG stream on {self.port}")
                raise RuntimeError(f"Failed to stop EEG recording on {self.port}") from exc

        logger.info(f"Captured {data.shape[1]} samples ({data.shape[1]/self.SFREQ:.1f}s)")

        # Build CSV
        # col 0 = sample index, cols 1-4 = EEG (µV), cols 5-12 = zeros,
        # col 13 = timestamp, col 14 = marker (aka the events (correct, error, etc))
        n_samples = data.shape[1]
        rows = np.zeros((n_samples, GANGLION.csv_total_cols))

        rows[:, 0]     = np.arange(n_samples)          # sample index
        rows[:, 1:5]   = data[self.EEG_COLS, :].T      # EEG channels (µV)
        rows[:, 13]    = data[self.TS_COL, :]           # unix timestamp
        rows[:, 14]    = data[self.MARKER_COL, :]       # markers

        # Data protection: secure output directory
        output_path = Path(output_dir) / f"flanker_eeg_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Ensure output_dir is not in a public location
        if output_path.parent.is_relative_to(Path.home() / "Desktop") or output_path.parent.is_relative_to(Path.home() / "Documents"):
            logger.warning(f"Saving EEG data to user-accessible directory: {output_path.parent}")
        
        header = ",".join(str(i) for i in range(GANGLION.csv_total_cols))
        try:
            np.savetxt(
                str(output_path),
                rows,
                delimiter=",",
                header=header,
                comments="",
                fmt=RECORDER.numpy_fmt,
            )
        except Exception as exc:
            logger.error(f"Failed to save EEG data to {output_path}")
            raise RuntimeError(f"Could not save EEG recording to {output_path}") from exc

        # Data protection: set restrictive permissions on Windows (limited but attempt)
        try:
            import os
            if os.name == 'nt':  # Windows
                # On Windows, this sets read-only for owner, but limited effect
                output_path.chmod(0o400)
            else:
                output_path.chmod(0o600)  # Owner read/write only
        except Exception:
            logger.warning(f"Could not set restrictive permissions on {output_path}")

        logger.info(f"EEG saved securely to {output_path}")
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