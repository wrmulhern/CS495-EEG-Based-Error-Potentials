"""Tests for configuration module."""

import pytest
from src.config import GANGLION, MARKERS, EPOCH, ERP, FLANKER, VALIDATION, EXPORT, RECORDER, PLOT


def test_ganglion_config():
    """Test GanglionConfig defaults."""
    assert GANGLION.sfreq == 200
    assert GANGLION.n_channels == 4
    assert GANGLION.eeg_cols == (1, 2, 3, 4)
    assert GANGLION.ts_col == 13
    assert GANGLION.marker_col == 14
    assert GANGLION.csv_total_cols == 15
    assert GANGLION.ring_buffer_samples == 45_000
    assert len(GANGLION.ch_locs) == 4


def test_markers_config():
    """Test Markers config."""
    assert MARKERS.congruent == 1
    assert MARKERS.incongruent == 2
    assert MARKERS.correct == 3
    assert MARKERS.error == 4
    assert MARKERS.no_response == 5


def test_epoch_config():
    """Test Epoch config."""
    assert EPOCH.pre_stimulus_ms == 200
    assert EPOCH.post_stimulus_ms == 800
    assert EPOCH.min_stimulus_events == 2


def test_erp_config():
    """Test ERP config."""
    assert ERP.ern_window_ms == (50, 150)
    assert ERP.pe_window_ms == (200, 400)


def test_flanker_config():
    """Test Flanker config."""
    assert FLANKER.default_n_trials == 100
    assert FLANKER.stimulus_ms == 200


def test_validation_config():
    """Test Validation config."""
    assert VALIDATION.max_file_size_mb == 500
    assert VALIDATION.allowed_extensions == frozenset({'.set', '.csv'})


def test_export_config():
    """Test Export config."""
    assert EXPORT.save_dpi == 300
    assert EXPORT.save_format == '.png'


def test_recorder_config():
    """Test Recorder config."""
    assert RECORDER.numpy_fmt == '%.10g'


def test_plot_config():
    """Test Plot config."""
    assert PLOT.epochs_figsize == (12, 6)
    assert PLOT.evoked_figsize == (12, 6)