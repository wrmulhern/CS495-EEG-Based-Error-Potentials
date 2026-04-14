"""Centralized configuration for all domain constants and tunable parameters.

Each frozen dataclass groups related settings.  Import the singleton
instances (``GANGLION``, ``MARKERS``, ``EPOCH``, ``ERP``, ``FLANKER``,
``VALIDATION``, ``EXPORT``, ``RECORDER``, ``PLOT``) wherever a value
is needed instead of scattering magic literals across the codebase.
"""

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Tuple


# ── EEG hardware ────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class GanglionConfig:
    """OpenBCI Ganglion board parameters."""

    # Native sampling rate of the Ganglion (Hz).
    sfreq: int = 200

    # Number of EEG channels on the Ganglion board.
    n_channels: int = 4

    # BrainFlow column indices that contain EEG data (one per channel).
    eeg_cols: Tuple[int, ...] = (1, 2, 3, 4)

    # BrainFlow column index for the Unix timestamp.
    ts_col: int = 13

    # BrainFlow column index for event markers injected via insert_marker().
    marker_col: int = 14

    # Total number of columns in the saved CSV (sample index + EEG + padding
    # + timestamp + marker).
    csv_total_cols: int = 15

    # BrainFlow ring-buffer length in samples (~225 s at 200 Hz).
    ring_buffer_samples: int = 45_000

    # Approximate 10-20 scalp positions for the four Ganglion electrodes.
    # Used to build EEGLAB chanlocs for topographic maps.
    ch_locs: Tuple[dict, ...] = (
        {"labels": "TP9",  "X": -0.87, "Y": -0.31, "Z": 0.0, "theta": -110.0, "radius": 0.9},
        {"labels": "AF7",  "X": -0.6,  "Y":  0.87, "Z": 0.0, "theta":  -55.0, "radius": 0.9},
        {"labels": "AF8",  "X":  0.6,  "Y":  0.87, "Z": 0.0, "theta":   55.0, "radius": 0.9},
        {"labels": "TP10", "X":  0.87, "Y": -0.31, "Z": 0.0, "theta":  110.0, "radius": 0.9},
    )

    # Fallback serial port shown when no ports are detected (Windows default).
    default_port: str = "COM3"

    # Multiplier to convert microvolts (µV) to volts (V) for EEGLAB storage.
    uv_to_v_scale: float = 1e-6


GANGLION = GanglionConfig()


# ── Event markers ───────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class MarkerConfig:
    """Flanker-task event marker codes injected into the EEG stream."""

    # Stimulus onset: all arrows point the same direction.
    congruent: int = 1

    # Stimulus onset: centre arrow conflicts with flankers.
    incongruent: int = 2

    # Response: participant pressed the correct arrow key.
    correct: int = 3

    # Response: participant pressed the wrong arrow key (expected to elicit ERN).
    error: int = 4

    # Response window expired with no keypress.
    no_response: int = 5

    @property
    def event_id(self) -> Dict[str, int]:
        """Name → code mapping used by the CSV-to-.set converter."""
        return {
            "congruent":   self.congruent,
            "incongruent": self.incongruent,
            "correct":     self.correct,
            "error":       self.error,
            "no_response": self.no_response,
        }

    @property
    def code_name(self) -> Dict[int, str]:
        """Code → name reverse lookup for EEGLAB event labels."""
        return {v: k for k, v in self.event_id.items()}

    @property
    def stimulus_codes(self) -> FrozenSet[int]:
        """The subset of marker codes that represent stimulus onsets
        (used to find epoch boundaries)."""
        return frozenset({self.congruent, self.incongruent})


MARKERS = MarkerConfig()


# ── Epoch extraction ────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class EpochConfig:
    """Parameters for cutting epochs around stimulus onsets."""

    # Baseline period before stimulus onset (ms). The epoch starts this
    # many milliseconds before the marker.
    pre_stimulus_ms: int = 200

    # Period after stimulus onset (ms). The epoch ends this many
    # milliseconds after the marker, capturing both ERN and Pe.
    post_stimulus_ms: int = 800

    # Minimum number of stimulus markers in the CSV before the epoched
    # path is attempted; below this the file is saved as continuous.
    min_stimulus_events: int = 2

    # Minimum number of valid (non-clipped) epochs required; if fewer
    # survive boundary checks, fall back to continuous.
    min_valid_epochs: int = 2

    # EEGLAB ``setname`` written into the .set file for epoched data.
    epoched_set_name: str = "Flanker_ErrP"

    # EEGLAB ``setname`` for continuous (non-epoched) recordings.
    continuous_set_name: str = "Ganglion_Recording"

    # EEG reference scheme string stored in the EEGLAB ``ref`` field.
    eeglab_ref: str = "common"

    # Suffix appended to the original CSV filename to form the .set path.
    converted_suffix: str = "_converted.set"


EPOCH = EpochConfig()


# ── ERP component windows ──────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class ERPConfig:
    """Time windows (ms) for error-related potential components."""

    # Error-Related Negativity window (ms post-response). A negative
    # deflection at fronto-central sites peaking around 50–150 ms.
    ern_window_ms: Tuple[int, int] = (50, 150)

    # Error Positivity window (ms post-response). A slower positive wave
    # following the ERN, typically 200–400 ms.
    pe_window_ms: Tuple[int, int] = (200, 400)


ERP = ERPConfig()


# ── Flanker task timing ─────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class FlankerTimingConfig:
    """Durations (ms) for each phase of the Flanker paradigm."""

    # How long the "+" fixation cross is shown before stimulus onset.
    fixation_ms: int = 500

    # How long the arrow string (e.g. "< < > < <") stays on screen.
    stimulus_ms: int = 200

    # Window after stimulus offset in which arrow-key responses are accepted.
    response_ms: int = 1000

    # Duration the feedback symbol (✓ / ✗ / "Too slow!") is displayed.
    feedback_ms: int = 400

    # Blank inter-trial interval between feedback offset and next fixation.
    iti_ms: int = 800

    # Delay before the very first trial begins after entering the task page.
    initial_delay_ms: int = 500

    # Default number of trials pre-filled in the setup QSpinBox.
    default_n_trials: int = 100


FLANKER = FlankerTimingConfig()


# ── File validation ─────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class ValidationConfig:
    """Thresholds for file and EEG data validation."""

    # File extensions the application will accept for loading.
    allowed_extensions: FrozenSet[str] = frozenset({".set", ".csv"})

    # Maximum file size allowed (megabytes); rejects excessively large uploads.
    max_file_size_mb: int = 500

    # Plausible EEG amplitude floor (µV); values below trigger a warning.
    eeg_voltage_min_uv: int = -1000

    # Plausible EEG amplitude ceiling (µV); values above trigger a warning.
    eeg_voltage_max_uv: int = 1000

    # Lowest acceptable sampling rate (Hz) in a .set file.
    eeg_sfreq_min_hz: int = 8

    # Highest acceptable sampling rate (Hz) in a .set file.
    eeg_sfreq_max_hz: int = 10_000

    # Minimum number of EEG channels a .set file must report.
    eeg_channels_min: int = 1

    # Maximum number of EEG channels a .set file may report.
    eeg_channels_max: int = 256

    # A CSV must have at least this many columns to be considered valid.
    csv_min_columns: int = 2

    # Stop scanning CSV structure after this many rows (performance cap).
    csv_max_scan_rows: int = 100_000

    # Maximum rows with inconsistent column counts before rejecting the CSV.
    csv_max_bad_rows: int = 5

    # Allowed column-count deviation (±) from the header when checking rows.
    csv_col_slack: int = 2

    # Number of leading data rows spot-checked for NaN / Inf in CSV integrity.
    csv_sample_rows: int = 100

    # Random sample size drawn from .set data for NaN / Inf checks.
    set_sample_size: int = 1000

    # Absolute value threshold (µV) that triggers a "unusually large" warning
    # during CSV data-integrity validation.
    csv_large_magnitude_threshold: int = 10_000

    # Number of bytes read from the file header for magic-byte signature checks.
    magic_header_bytes: int = 8

    # Minimum channel count required for topographic map interpolation.
    min_topo_channels: int = 19

    @property
    def max_file_size_bytes(self) -> int:
        """Derived byte-level cap (max_file_size_mb × 1 MiB)."""
        return self.max_file_size_mb * 1024 * 1024


VALIDATION = ValidationConfig()


# ── Export / save defaults ──────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class ExportConfig:
    """Defaults for saving figures and data."""

    # Resolution (dots per inch) used when saving figures to disk.
    save_dpi: int = 300

    # Default image format extension appended if the user omits one.
    save_format: str = ".png"

    # Qt file-dialog filter string for opening EEG data files.
    file_dialog_filter: str = "EEG Files (*.set *.csv);;All Files (*.*)"

    # Qt file-dialog filter string for the "Save Graph As" dialog.
    save_dialog_filter: str = "PNG Files (*.png);;All Files (*.*)"


EXPORT = ExportConfig()


# ── Recorder CSV output ─────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class RecorderConfig:
    """Settings for the raw CSV output from the EEG recorder."""

    # Template for the timestamped output filename (strftime-formatted at runtime).
    filename_pattern: str = "flanker_eeg_{timestamp}.csv"

    # numpy.savetxt format specifier for writing numeric values to CSV.
    numpy_fmt: str = "%.10g"


RECORDER = RecorderConfig()


# ── Plot / visualization defaults ───────────────────────────────────

@dataclass(frozen=True, slots=True)
class PlotConfig:
    """Default sizes and styling parameters for Matplotlib figures."""

    # Figure size (width, height in inches) for the epoch butterfly plot.
    epochs_figsize: Tuple[int, int] = (12, 6)

    # Figure size for the averaged evoked-response (ERP) plot.
    evoked_figsize: Tuple[int, int] = (12, 6)

    # Width and height (inches) of each topomap cell in a multi-time grid.
    topo_cell_size: int = 4

    # Figure size for the joint (time-series + topomap row) plot.
    joint_figsize: Tuple[int, int] = (14, 8)

    # Figure size for a single animated-topomap frame.
    topo_frame_figsize: Tuple[int, int] = (6, 5)

    # Opacity of the background grid lines on all plots.
    grid_alpha: float = 0.3

    # Fraction of axes width consumed by the topomap colorbar.
    colorbar_fraction: float = 0.05

    # Padding between the axes and the colorbar (axes-fraction units).
    colorbar_pad: float = 0.04

    # Font size (pt) for figure-level suptitles and topomap frame titles.
    suptitle_fontsize: int = 14

    # Font size (pt) for channel-name entries in the legend.
    legend_fontsize: int = 8

    # Background opacity of the legend box (0 = transparent, 1 = opaque).
    legend_framealpha: float = 0.95

    # If the number of plotted channels exceeds this, individual legend
    # labels are suppressed to avoid clutter.
    max_labeled_channels: int = 20

    # Minimum peak-to-peak distance (seconds) used by scipy.signal.find_peaks
    # when auto-selecting topomap times from the GFP waveform.
    gfp_peak_distance_s: float = 0.05

    # Fallback topomap time points (seconds) when GFP has no detectable peaks.
    default_topo_times: Tuple[float, ...] = (0.1, 0.2, 0.3)

    # Number of filled contour levels in topomap plots.
    contours: int = 6

    # Grid resolution (points per axis) for topomap cubic interpolation.
    topo_grid_resolution: int = 100

    # Extra padding (data-coordinate units) around electrode positions
    # when building the interpolation grid.
    topo_margin: float = 0.1

    # Radius of the schematic head circle drawn on topomap plots.
    head_radius: float = 1.0

    # Length of the nose wedge extending beyond the head circle.
    nose_length: float = 0.2

    # QTimer interval (ms) between animated-topomap frame redraws.
    anim_timer_ms: int = 50

    # Real-time duration (seconds) that each animation tick represents;
    # multiplied by sfreq and playback speed to compute the slider step.
    anim_tick_duration_s: float = 0.025

    # Default (width, height) in pixels for the main application window.
    main_window_size: Tuple[int, int] = (1280, 760)


PLOT = PlotConfig()
