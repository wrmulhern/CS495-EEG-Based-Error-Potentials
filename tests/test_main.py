"""Tests for main module."""

import logging
from unittest.mock import patch, Mock
from src.main import setup_logging, main


def test_setup_logging():
    """Test logging setup."""
    setup_logging(logging.INFO)
    logger = logging.getLogger("src")
    assert logger.level == logging.INFO


def test_main():
    """Test main function."""
    with patch('sys.argv', ['main.py']):
        with patch('src.main.QApplication') as mock_qapp:
            with patch('src.main.FileWindow') as mock_file_window:
                mock_app = Mock()
                mock_qapp.return_value = mock_app
                main()
                mock_qapp.assert_called_once()
                mock_file_window.assert_called_once_with(None)
                mock_file_window.return_value.show.assert_called()
                mock_app.exec_.assert_called()