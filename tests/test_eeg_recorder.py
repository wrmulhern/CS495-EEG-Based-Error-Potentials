"""Tests for eeg_recorder module."""

from unittest.mock import Mock, patch
from src.data_processing.eeg_recorder import EEGRecorder, MARKER_CONGRUENT


def test_eeg_recorder_init_valid():
    """Test EEGRecorder init with valid port."""
    with patch('src.data_processing.eeg_recorder.EEGRecorder.list_ports', return_value=['COM3']):
        recorder = EEGRecorder('COM3')
        assert recorder.port == 'COM3'


def test_eeg_recorder_init_invalid_port():
    """Test EEGRecorder init with invalid port."""
    with patch('src.data_processing.eeg_recorder.EEGRecorder.list_ports', return_value=['COM3']):
        try:
            EEGRecorder('COM4')
            assert False, "Should raise ValueError"
        except ValueError:
            pass


def test_list_ports():
    """Test list_ports."""
    with patch('serial.tools.list_ports.comports', return_value=[]):
        ports = EEGRecorder.list_ports()
        assert ports == []