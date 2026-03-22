"""
Data loader for EEGLAB .set files (MNE-free implementation)
"""

import os
import logging
from pathlib import Path
import numpy as np
from scipy.io import loadmat

from .file_validator import FileValidator, FileValidationError

logger = logging.getLogger(__name__)

class Bunch(dict):
    """Dictionary that allows attribute-style access."""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class EpochsData:
    """
    Container for epoched EEG data with basic processing and visualization.

    Attributes:
        data: numpy array of shape (n_epochs, n_channels, n_times)
        ch_names: list of channel names
        sfreq: sampling frequency in Hz
        times: time vector for each epoch
        events: event information
        event_id: mapping of event names to event codes
        info: dictionary containing metadata
    """

    def __init__(self, data, ch_names, sfreq, tmin, events=None, event_id=None,
                 ch_types=None, ch_locs=None):
        """
        Initialize EpochsData object.

        Parameters:
            data: array (n_epochs, n_channels, n_times)
            ch_names: list of channel names
            sfreq: sampling frequency
            tmin: start time of epoch relative to event
            events: event array (n_events, 3) with [sample, 0, event_code]
            event_id: dict mapping event names to codes
            ch_types: list of channel types (e.g., 'eeg', 'eog')
            ch_locs: channel location information
        """
        self.data = data
        self.ch_names = ch_names
        self.sfreq = sfreq
        self.tmin = tmin
        self.events = events
        self.event_id = event_id or {}
        self.ch_types = ch_types or ['eeg'] * len(ch_names)
        self.ch_locs = ch_locs

        # Create time vector
        n_times = data.shape[2]
        self.times = np.arange(n_times) / sfreq + tmin
        self.tmax = self.times[-1]

        # Store info dict for compatibility
        self.info = {
            'sfreq': sfreq,
            'ch_names': ch_names,
            'nchan': len(ch_names),
            'ch_types': self.ch_types
        }

    def __repr__(self):
        n_epochs, n_channels, n_times = self.data.shape
        return (f"<EpochsData | {n_epochs} epochs, {n_channels} channels, "
                f"{n_times} time points, {self.tmin:.3f} - {self.tmax:.3f} s, "
                f"sfreq={self.sfreq} Hz>")

    def get_data(self):
        """Return the data array."""
        return self.data

    def to_data_frame(self):
        """
        Convert epochs to a pandas-like structure (returns dict for now).

        Returns:
            Dictionary with shape information
        """
        n_epochs, n_channels, n_times = self.data.shape
        return {
            'shape': (n_epochs * n_channels * n_times, 4),  # epoch, channel, time, value
            'n_epochs': n_epochs,
            'n_channels': n_channels,
            'n_times': n_times
        }


def read_epochs_eeglab_minimal(set_file):
    """
    Read EEGLAB .set file containing epoched data.

    This is a minimal implementation that reads the essential fields
    from EEGLAB format without requiring MNE.

    Parameters:
        set_file: path to .set file
        verbose: print debugging information

    Returns:
        EpochsData object

    Raises:
        FileValidationError: If file validation fails
    """
    set_file = Path(set_file)

    logger.debug(f"Loading file: {set_file}")

    # Validate file before loading
    try:
        file_type, validation_info = FileValidator.validate_file(str(set_file))
        if file_type != 'set':
            raise FileValidationError(f"Expected .set file, got .{file_type}")
        logger.debug(f"File validation passed. Size: {validation_info['file_size_bytes']} bytes")
    except FileValidationError as e:
        logger.error(f"File validation failed: {e}")
        raise

    # Load MATLAB file
    mat = loadmat(set_file, squeeze_me=True, struct_as_record=False)

    logger.debug(f"Keys in .mat file: {list(mat.keys())}")

    # Get EEG structure
    eeg = mat.get("EEG", mat)
    if isinstance(eeg, dict) and "EEG" in eeg:
        eeg = eeg["EEG"]
    if isinstance(eeg, dict):
        eeg = Bunch(**eeg)

        logger.debug(f"EEG structure type: {type(eeg)}")
        if hasattr(eeg, '__dict__'):
            logger.debug(f"EEG attributes: {list(eeg.__dict__.keys())[:10]}...")

    # Check if data is epoched or continuous
    trials = getattr(eeg, 'trials', 1)
    logger.debug(f"Number of trials: {trials}")

    is_continuous = int(trials) <= 1

    if is_continuous:
        logger.debug("Detected continuous data (will be treated as single epoch)")

    # Extract data
    if not hasattr(eeg, 'data'):
        raise ValueError("EEG structure does not contain 'data' field")

    data = eeg.data

    logger.debug(f"Raw data type: {type(data)}")
    logger.debug(f"Data shape/size: {getattr(data, 'shape', getattr(data, 'size', 'unknown'))}")

    # Handle case where data is stored in separate .fdt file
    if isinstance(data, str):
        logger.debug(f"Data is stored externally in file: {data}")

        # Get data file path
        data_file = Path(data)
        if not data_file.is_absolute():
            data_file = set_file.parent / data_file

        logger.debug(f"Loading data from: {data_file}")

        if not data_file.suffix:
            data_file = data_file.with_suffix('.fdt')

        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")

        # Get dimensions from EEG structure
        n_channels = int(eeg.nbchan)
        n_times = int(eeg.pnts)
        n_epochs = int(eeg.trials)

        logger.debug(f"Expected dimensions: {n_channels} channels x {n_times} timepoints x {n_epochs} epochs")

        # Read binary data (EEGLAB uses float32)
        data = np.fromfile(str(data_file), dtype=np.float32)

        logger.debug(f"Loaded {len(data)} values from .fdt file")

        # Reshape to EEGLAB format: (channels, timepoints, epochs)
        try:
            data = data.reshape(n_channels, n_times, n_epochs, order='F')  # Fortran order
        except ValueError:
            try:
                data = data.reshape(n_channels, n_times, n_epochs, order='C')
                logger.debug("Note: Using C-order reshape instead of Fortran-order")
            except ValueError:
                raise ValueError(f"Cannot reshape data. Expected {n_channels}x{n_times}x{n_epochs} = "
                               f"{n_channels*n_times*n_epochs}, but got {len(data)} values")

    # Convert to numpy array if it isn't already
    if not isinstance(data, np.ndarray):
        data = np.array(data)
        logger.debug(f"Converted to numpy array, shape: {data.shape}")

    # Ensure we have 3D data
    if data.ndim == 2:
        # Continuous data: (channels, timepoints)
        logger.debug("Data is 2D, treating as single epoch")
        data = data[np.newaxis, :, :]
    elif data.ndim == 3:
        # Epoched data: (channels, timepoints, epochs)
        # Transpose to our format: (epochs, channels, timepoints)
        logger.debug(f"Data is 3D with shape {data.shape}, transposing to (epochs, channels, times)")
        data = np.transpose(data, (2, 0, 1))
    else:
        raise ValueError(f"Unexpected data dimensions: {data.shape}")

        logger.debug(f"Final data shape: {data.shape} (epochs, channels, times)")

    # Extract channel information
    ch_names = []
    ch_locs = None

    if hasattr(eeg, 'chanlocs') and eeg.chanlocs is not None:
        chanlocs = eeg.chanlocs

        if not isinstance(chanlocs, (list, np.ndarray)):
            chanlocs = [chanlocs]

        # Extract channel names
        for i, ch in enumerate(chanlocs):
            if hasattr(ch, 'labels'):
                ch_names.append(str(ch.labels))
            elif hasattr(ch, 'label'):
                ch_names.append(str(ch.label))
            else:
                ch_names.append(f'Ch{i+1}')

        ch_locs = chanlocs

        logger.debug(f"Found {len(ch_names)} channels: {ch_names[:5]}..." if len(ch_names) > 5 else f"Channels: {ch_names}")
    else:
        # No channel info, create default names
        n_channels = data.shape[1]
        ch_names = [f'Ch{i+1}' for i in range(n_channels)]
        logger.debug(f"No channel info found, created {len(ch_names)} default channel names")

    # Extract timing information
    if hasattr(eeg, 'srate'):
        sfreq = float(eeg.srate)
    elif hasattr(eeg, 'sfreq'):
        sfreq = float(eeg.sfreq)
    else:
        raise ValueError("Cannot find sampling rate (srate or sfreq) in EEG structure")

    if hasattr(eeg, 'xmin'):
        xmin = float(eeg.xmin)
    elif hasattr(eeg, 'tmin'):
        xmin = float(eeg.tmin)
    else:
        xmin = 0.0
        logger.debug("Warning: Could not find xmin/tmin, assuming 0.0")

        logger.debug(f"Sampling rate: {sfreq} Hz")
        logger.debug(f"Epoch start time: {xmin} seconds")

    # Extract events
    events = None
    event_id = {}

    if hasattr(eeg, 'event') and eeg.event is not None:
        if hasattr(eeg, 'epoch'):
            event_id = {'epoch': 1}  # Simplified

    # Create EpochsData object
    epochs = EpochsData(
        data=data,
        ch_names=ch_names,
        sfreq=sfreq,
        tmin=xmin,
        events=events,
        event_id=event_id,
        ch_locs=ch_locs
    )

    return epochs


def read_csv_data(csv_file, verbose=True):
    """
    Read EEG data from CSV file with validation.

    Parameters:
        csv_file: path to .csv file
        verbose: print debugging information

    Returns:
        Tuple of (data, ch_names, metadata_dict)

    Raises:
        FileValidationError: If file validation fails
    """
    import csv

    csv_file = Path(csv_file)

    if verbose:
        logger.debug(f"Loading CSV file: {csv_file}")

    # Validate file before loading
    try:
        file_type, validation_info = FileValidator.validate_file(str(csv_file))
        if file_type != 'csv':
            raise FileValidationError(f"Expected .csv file, got .{file_type}")
        if verbose:
            logger.debug(f"File validation passed. Size: {validation_info['file_size_bytes']} bytes")
            logger.debug(f"CSV info - Rows: {validation_info['csv_rows']}, "
                        f"Headers: {validation_info['csv_headers']}")
    except FileValidationError as e:
        logger.error(f"File validation failed: {e}")
        raise

    # Load CSV data
    try:
        data_rows = []
        ch_names = None

        with open(csv_file, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f)

            # Read headers
            ch_names = next(reader)

            # Read data rows
            for row in reader:
                # Skip empty rows
                if not row or all(cell.strip() == '' for cell in row):
                    continue

                try:
                    # Convert to float
                    row_data = [float(cell.strip()) for cell in row]
                    data_rows.append(row_data)
                except ValueError as e:
                    logger.warning(f"Skipping row with non-numeric data: {row}")
                    continue

        if not data_rows:
            raise FileValidationError("CSV file contains no valid numeric data rows")

        # Convert to numpy array
        data = np.array(data_rows, dtype=np.float32)

        if verbose:
            logger.debug(f"Loaded CSV data with shape: {data.shape}")
            logger.debug(f"Channels: {ch_names}")
            logger.debug(f"Data range: [{data.min():.3f}, {data.max():.3f}]")

        # Create metadata
        metadata = {
            'file_path': str(csv_file),
            'n_channels': len(ch_names),
            'n_samples': len(data_rows),
            'data_min': float(data.min()),
            'data_max': float(data.max()),
            'data_mean': float(data.mean()),
        }

        return data, ch_names, metadata

    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        raise FileValidationError(f"Error reading CSV file: {e}")
