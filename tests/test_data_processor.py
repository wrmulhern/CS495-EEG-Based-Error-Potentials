"""Tests for data_processor module."""

import numpy as np
from src.data_processing.data_processor import average_epochs, select_channels, select_time_window, EvokedData


def test_evoked_data_init():
    """Test EvokedData initialization."""
    data = np.random.randn(4, 200)
    evoked = EvokedData(data, ['Ch1', 'Ch2', 'Ch3', 'Ch4'], 200, -0.2)
    assert evoked.data.shape == (4, 200)


def test_average_epochs():
    """Test averaging epochs."""
    data = np.random.randn(20, 4, 200)
    epochs_data = type('MockEpochs', (), {'data': data, 'ch_names': ['Ch1', 'Ch2', 'Ch3', 'Ch4'], 'ch_types': ['eeg', 'eeg', 'eeg', 'eeg'], 'sfreq': 200, 'tmin': -0.2, 'ch_locs': None})()
    avg = average_epochs(epochs_data)
    assert avg.data.shape == (4, 200)


def test_select_channels():
    """Test channel selection."""
    data = np.random.randn(10, 4, 200)
    epochs_data = type('MockEpochs', (), {'data': data, 'ch_names': ['Ch1', 'Ch2', 'Ch3', 'Ch4'], 'ch_types': ['eeg', 'eeg', 'eeg', 'eeg'], 'sfreq': 200, 'tmin': -0.2, 'events': None, 'event_id': None, 'ch_locs': None})()
    selected = select_channels(epochs_data, [0, 2])
    assert selected.data.shape == (10, 2, 200)
    assert selected.ch_names == ['Ch1', 'Ch3']


def test_select_time_window():
    """Test time window selection."""
    data = np.random.randn(10, 4, 200)
    times = np.linspace(-0.2, 0.8, 200)
    epochs_data = type('MockEpochs', (), {'data': data, 'ch_names': ['Ch1', 'Ch2', 'Ch3', 'Ch4'], 'ch_types': ['eeg', 'eeg', 'eeg', 'eeg'], 'sfreq': 200, 'tmin': -0.2, 'times': times, 'events': None, 'event_id': None, 'ch_locs': None})()
    selected = select_time_window(epochs_data, 0.0, 0.5)
    expected_samples = 100  # 0.5 seconds * 200 Hz = 100 samples
    assert selected.data.shape[2] == expected_samples