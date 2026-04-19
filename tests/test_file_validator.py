"""Tests for file_validator module."""

import tempfile
import os
from src.data_processing.file_validator import FileValidator, FileValidationError


def test_validate_file_path_valid():
    """Test valid file path."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b'test')
        temp_path = f.name
    try:
        FileValidator.validate_file_path(temp_path)
    finally:
        os.unlink(temp_path)


def test_validate_file_path_invalid():
    """Test invalid file path."""
    try:
        FileValidator.validate_file_path('/nonexistent/file.set')
        assert False, "Should raise FileValidationError"
    except FileValidationError:
        pass


def test_validate_file_extension_valid():
    """Test valid extension."""
    ext = FileValidator.validate_file_extension('test.set')
    assert ext == '.set'


def test_validate_file_extension_invalid():
    """Test invalid extension."""
    try:
        FileValidator.validate_file_extension('test.txt')
        assert False, "Should raise FileValidationError"
    except FileValidationError:
        pass