"""
File validation and security module for EEG data files (.set and .csv)
Provides comprehensive validation before loading files for visualization.
"""

import os
import re
import csv
import logging
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Security and Format Constants
ALLOWED_EXTENSIONS = {'.set', '.csv'}
MAX_FILE_SIZE_MB = 500  # Maximum 500 MB
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# File signature/magic bytes for validation
SET_FILE_SIGNATURES = [
    b'MATLAB',  # MATLAB .mat files
    b'\x89HDF',  # HDF5 format
]

CSV_FILE_SIGNATURES = [
    b'\xef\xbb\xbf',  # UTF-8 BOM
]

# EEG Data validation ranges
EEG_VOLTAGE_MIN_UV = -1000
EEG_VOLTAGE_MAX_UV = 1000
EEG_SFREQ_MIN_HZ = 8
EEG_SFREQ_MAX_HZ = 10000
EEG_CHANNELS_MIN = 1
EEG_CHANNELS_MAX = 256

# Filename validation pattern - reject suspicious characters
DANGEROUS_FILENAME_PATTERN = re.compile(r'[\x00-\x1f\\]|\.\.|\0')
UNSAFE_FILENAME_CHARS = re.compile(r'[<>:"|?*\x00-\x1f]')


class FileValidationError(Exception):
    """Raised when file validation fails."""
    pass


class FileValidator:
    """
    Comprehensive file validator for EEG data files.
    Validates format, size, integrity, and data ranges.
    """

    @staticmethod
    def validate_file_path(file_path: str) -> None:
        """
        Validate file path for security issues.
        
        Args:
            file_path: Path to the file
            
        Raises:
            FileValidationError: If path validation fails
        """
        path = Path(file_path)
        filename = path.name
        
        # Check for path traversal attempts
        if '..' in str(path) or DANGEROUS_FILENAME_PATTERN.search(filename):
            raise FileValidationError(
                f"Invalid filename or path traversal detected: {filename}"
            )
        
        # Check for null bytes or other encoding attacks
        try:
            filename.encode('utf-8')
        except UnicodeEncodeError:
            raise FileValidationError(
                f"Filename contains invalid characters: {filename}"
            )
        
        # Check file exists and is readable
        if not path.exists():
            raise FileValidationError(f"File does not exist: {file_path}")
        
        if not path.is_file():
            raise FileValidationError(f"Path is not a file: {file_path}")
        
        if not os.access(file_path, os.R_OK):
            raise FileValidationError(f"File is not readable: {file_path}")

    @staticmethod
    def validate_file_extension(file_path: str) -> str:
        """
        Validate file extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File extension (lowercase)
            
        Raises:
            FileValidationError: If extension is invalid
        """
        path = Path(file_path)
        ext = path.suffix.lower()
        
        if ext not in ALLOWED_EXTENSIONS:
            raise FileValidationError(
                f"Invalid file extension '{ext}'. Allowed: {ALLOWED_EXTENSIONS}"
            )
        
        return ext

    @staticmethod
    def validate_file_size(file_path: str) -> None:
        """
        Validate file size is within acceptable limits.
        
        Args:
            file_path: Path to the file
            
        Raises:
            FileValidationError: If file is too large
        """
        path = Path(file_path)
        file_size = path.stat().st_size
        
        if file_size == 0:
            raise FileValidationError("File is empty")
        
        if file_size > MAX_FILE_SIZE_BYTES:
            raise FileValidationError(
                f"File too large: {file_size / 1024 / 1024:.2f} MB "
                f"(max: {MAX_FILE_SIZE_MB} MB)"
            )

    @staticmethod
    def validate_file_signature(file_path: str, file_type: str) -> None:
        """
        Validate file signature (magic bytes) to ensure correct format.
        
        Args:
            file_path: Path to the file
            file_type: 'set' or 'csv'
            
        Raises:
            FileValidationError: If file signature is invalid
        """
        with open(file_path, 'rb') as f:
            header = f.read(8)
        
        if file_type == 'set':
            # Check for MATLAB or HDF5 signature
            is_valid = any(header.startswith(sig) for sig in SET_FILE_SIGNATURES)
            if not is_valid:
                raise FileValidationError(
                    "Invalid .set file format. Expected MATLAB or HDF5 format."
                )
        
        elif file_type == 'csv':
            # CSV files are text, check for valid text encoding
            try:
                # Try to read as UTF-8
                header.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    # Try Latin-1 as fallback
                    header.decode('latin-1')
                except UnicodeDecodeError:
                    raise FileValidationError(
                        "CSV file has invalid text encoding. Expected UTF-8 or Latin-1."
                    )

    @staticmethod
    def validate_csv_format(file_path: str) -> Tuple[List[str], int]:
        """
        Validate CSV file format and structure.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Tuple of (headers, number_of_rows)
            
        Raises:
            FileValidationError: If CSV format is invalid
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                # Skip comment lines starting with '%'
                lines = []
                for line in f:
                    if not line.startswith('%'):
                        lines.append(line)
                
                if not lines:
                    raise FileValidationError("CSV file has no data (only comments)")
                
                # Parse with csv.reader
                reader = csv.reader(lines)
                
                try:
                    headers = next(reader)
                except StopIteration:
                    raise FileValidationError("CSV file has no headers")
                
                if not headers or all(h.strip() == '' for h in headers):
                    raise FileValidationError("CSV headers are empty")
                
                # Validate headers
                if len(headers) < 2:
                    raise FileValidationError(
                        f"CSV has too few columns ({len(headers)}). Expected at least 2."
                    )
                
                expected_cols = len(headers)
                
                # Count rows and validate consistency (allow some tolerance for missing data)
                row_count = 0
                inconsistent_rows = 0
                max_inconsistent_allowed = 5  # Allow up to 5 rows with different column counts
                
                for row_idx, row in enumerate(reader):
                    if row_idx >= 100000:  # Limit initial validation to first 100k rows
                        break
                    
                    # Skip empty rows
                    if not row or all(cell.strip() == '' for cell in row):
                        continue
                    
                    row_count += 1
                    
                    # Check column count - allow some tolerance for incomplete rows
                    # (EEG data might have trailing missing values)
                    if len(row) < expected_cols - 2 or len(row) > expected_cols + 2:
                        inconsistent_rows += 1
                        if inconsistent_rows > max_inconsistent_allowed:
                            raise FileValidationError(
                                f"Many rows with inconsistent column count. "
                                f"Expected ~{expected_cols} columns. "
                                f"Row {row_idx + 2} has {len(row)} columns."
                            )
                
                if row_count == 0:
                    raise FileValidationError("CSV file has no data rows")
                
                if inconsistent_rows > 0:
                    logger.warning(
                        f"CSV file has {inconsistent_rows} rows with inconsistent column counts. "
                        f"These may be incomplete records."
                    )
                
                return headers, row_count
        
        except csv.Error as e:
            raise FileValidationError(f"CSV parsing error: {e}")
        except Exception as e:
            raise FileValidationError(f"Error reading CSV file: {e}")

    @staticmethod
    def validate_csv_data_integrity(
        file_path: str,
        headers: List[str],
        sample_rows: int = 100
    ) -> None:
        """
        Validate CSV data integrity and ranges.
        
        Args:
            file_path: Path to CSV file
            headers: List of column headers
            sample_rows: Number of rows to sample for validation
            
        Raises:
            FileValidationError: If data integrity checks fail
        """
        numeric_columns = []
        
        # First pass: try to identify numeric columns
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f)
            next(reader)  # Skip headers
            
            # Sample first N rows
            for row_idx, row in enumerate(reader):
                if row_idx >= sample_rows:
                    break
                
                if not row or all(cell.strip() == '' for cell in row):
                    continue
                
                for col_idx, cell in enumerate(row):
                    if col_idx >= len(numeric_columns):
                        numeric_columns.append(True)
                    
                    if numeric_columns[col_idx]:
                        try:
                            val = float(cell.strip())
                            # Check for NaN/Inf
                            if not np.isfinite(val):
                                raise FileValidationError(
                                    f"Invalid numeric value in column {col_idx} "
                                    f"('{headers[col_idx]}'): {cell} contains NaN/Inf"
                                )
                            # Check EEG voltage range (if looks like EEG data)
                            if abs(val) > 10000:  # Generous upper bound for unusual units
                                logger.warning(
                                    f"Unusually large value in column {col_idx}: {val}"
                                )
                        except ValueError:
                            numeric_columns[col_idx] = False

    @staticmethod
    def validate_set_file_structure(set_file_path: str) -> None:
        """
        Validate .set file structure and required fields.
        
        Args:
            set_file_path: Path to .set file
            
        Raises:
            FileValidationError: If structure validation fails
        """
        try:
            from scipy.io import loadmat
        except ImportError:
            logger.warning("scipy not available, skipping detailed .set validation")
            return
        
        try:
            # Try to read the file
            mat = loadmat(set_file_path, squeeze_me=True, struct_as_record=False)
            
            # Check for EEG structure (nested) or flattened structure
            eeg = mat.get("EEG", None)
            if eeg is None:
                # Use mat directly (flattened structure)
                eeg = mat
            
            # Convert to dict-like if needed
            if hasattr(eeg, '__dict__'):
                eeg_dict = eeg.__dict__
            else:
                eeg_dict = dict(eeg) if isinstance(eeg, dict) else {}
            
            # Check required fields (try both the object and dict)
            required_fields = ['nbchan', 'pnts', 'srate']
            missing_fields = [
                f for f in required_fields 
                if f not in eeg_dict and not hasattr(eeg, f)
            ]
            
            if missing_fields:
                raise FileValidationError(
                    f".set file missing required fields: {missing_fields}"
                )
            
            # Validate field ranges
            nbchan = int(getattr(eeg, 'nbchan', eeg_dict.get('nbchan', 0)))
            pnts = int(getattr(eeg, 'pnts', eeg_dict.get('pnts', 0)))
            srate = float(getattr(eeg, 'srate', eeg_dict.get('srate', 0)))
            
            if not (EEG_CHANNELS_MIN <= nbchan <= EEG_CHANNELS_MAX):
                raise FileValidationError(
                    f"Invalid channel count: {nbchan}. "
                    f"Expected {EEG_CHANNELS_MIN}-{EEG_CHANNELS_MAX}"
                )
            
            if pnts <= 0:
                raise FileValidationError(f"Invalid sample count: {pnts}")
            
            if not (EEG_SFREQ_MIN_HZ <= srate <= EEG_SFREQ_MAX_HZ):
                raise FileValidationError(
                    f"Invalid sampling rate: {srate} Hz. "
                    f"Expected {EEG_SFREQ_MIN_HZ}-{EEG_SFREQ_MAX_HZ} Hz"
                )
        
        except FileValidationError:
            raise
        except Exception as e:
            raise FileValidationError(f"Error reading .set file structure: {e}")

    @staticmethod
    def validate_set_data_integrity(set_file_path: str, sample_size: int = 1000) -> None:
        """
        Validate .set file data integrity and ranges.
        
        Args:
            set_file_path: Path to .set file
            sample_size: Number of samples to check
            
        Raises:
            FileValidationError: If data validation fails
        """
        try:
            from scipy.io import loadmat
        except ImportError:
            logger.warning("scipy not available, skipping detailed .set data validation")
            return
        
        try:
            mat = loadmat(set_file_path, squeeze_me=True, struct_as_record=False)
            eeg = mat.get("EEG", mat)  # Use mat directly if no EEG key (flattened structure)
            
            # Check data field
            data = eeg.get('data', None) if isinstance(eeg, dict) else getattr(eeg, 'data', None)
            if data is None:
                raise FileValidationError(".set file does not contain 'data' field")
            
            # If data is a string, it's stored in external file
            if isinstance(data, str):
                # Validate external file reference
                data_file = Path(data)
                if not data_file.is_absolute():
                    data_file = Path(set_file_path).parent / data_file
                
                if not data_file.exists():
                    raise FileValidationError(
                        f"External data file not found: {data_file}"
                    )
                
                if data_file.stat().st_size == 0:
                    raise FileValidationError(
                        f"External data file is empty: {data_file}"
                    )
            else:
                # Data is inline, sample it
                if isinstance(data, np.ndarray) and data.size > 0:
                    # Check a random sample
                    sample_indices = np.random.choice(
                        data.size,
                        min(sample_size, data.size),
                        replace=False
                    )
                    sample_data = data.flat[sample_indices]
                    
                    # Check for invalid values
                    if np.any(~np.isfinite(sample_data)):
                        invalid_count = np.sum(~np.isfinite(sample_data))
                        logger.warning(
                            f"Found {invalid_count} NaN/Inf values in data sample"
                        )
        
        except FileValidationError:
            raise
        except Exception as e:
            logger.warning(f"Could not fully validate .set data: {e}")

    @staticmethod
    def validate_file(file_path: str) -> Tuple[str, dict]:
        """
        Comprehensive file validation.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Tuple of (file_type, validation_info dict)
            
        Raises:
            FileValidationError: If any validation fails
        """
        validation_info = {
            'file_path': file_path,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Step 1: Validate file path
            FileValidator.validate_file_path(file_path)
            
            # Step 2: Validate extension
            ext = FileValidator.validate_file_extension(file_path)
            file_type = ext.lstrip('.')
            validation_info['file_type'] = file_type
            
            # Step 3: Validate file size
            FileValidator.validate_file_size(file_path)
            validation_info['file_size_bytes'] = Path(file_path).stat().st_size
            
            # Step 4: Validate file signature
            FileValidator.validate_file_signature(file_path, file_type)
            
            # Step 5: Format-specific validation
            if file_type == 'csv':
                headers, row_count = FileValidator.validate_csv_format(file_path)
                validation_info['csv_headers'] = headers
                validation_info['csv_rows'] = row_count
                
                # Validate data integrity
                FileValidator.validate_csv_data_integrity(file_path, headers)
            
            elif file_type == 'set':
                FileValidator.validate_set_file_structure(file_path)
                FileValidator.validate_set_data_integrity(file_path)
            
            logger.info(f"File validation passed: {file_path}")
            return file_type, validation_info
        
        except FileValidationError as e:
            logger.error(f"File validation failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during validation: {e}")
            raise FileValidationError(f"Validation error: {e}")


def validate_and_check_file(file_path: str) -> Tuple[bool, Optional[str]]:
    """
    User-friendly wrapper for file validation.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Tuple of (is_valid, error_message)
        error_message is None if valid, otherwise contains error text
    """
    try:
        FileValidator.validate_file(file_path)
        return True, None
    except FileValidationError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected validation error: {str(e)}"
