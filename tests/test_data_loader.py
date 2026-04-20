"""Tests for data_loader module."""

import numpy as np
from pathlib import Path
from src.data_processing.data_loader import read_epochs_eeglab_minimal, read_csv_data, EpochsData, Bunch


def test_bunch():
    """Test Bunch class."""
    b = Bunch({'a': 1, 'b': 2})
    assert b.a == 1
    assert b.b == 2
    assert b['a'] == 1
    b.c = 3
    assert b.c == 3


def test_epochs_data_init():
    """Test EpochsData initialization."""
    data = np.random.randn(10, 4, 200)
    epochs = EpochsData(data, ['Ch1', 'Ch2', 'Ch3', 'Ch4'], 200, -0.2)
    assert epochs.data.shape == (10, 4, 200)
    assert epochs.ch_names == ['Ch1', 'Ch2', 'Ch3', 'Ch4']
    assert epochs.sfreq == 200
    assert epochs.tmin == -0.2