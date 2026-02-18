"""
Data processor for EEG epochs - averaging and signal processing
"""

import numpy as np


class EvokedData:
    """
    Container for averaged EEG data (evoked response).

    This represents the average of many epochs (trials), which shows
    the typical brain response to a stimulus (ERP/ErrP).

    Attributes:
        data: numpy array of shape (n_channels, n_times)
        ch_names: list of channel names
        sfreq: sampling frequency in Hz
        times: time vector
        info: dictionary containing metadata
    """

    def __init__(self, data, ch_names, sfreq, tmin, ch_types=None, ch_locs=None):
        """
        Initialize EvokedData object.

        Parameters:
            data: array (n_channels, n_times)
            ch_names: list of channel names
            sfreq: sampling frequency
            tmin: start time relative to event
            ch_types: list of channel types
            ch_locs: channel location information
        """
        self.data = data
        self.ch_names = ch_names
        self.sfreq = sfreq
        self.tmin = tmin
        self.ch_types = ch_types or ['eeg'] * len(ch_names)
        self.ch_locs = ch_locs

        # Create time vector
        n_times = data.shape[1]
        self.times = np.arange(n_times) / sfreq + tmin
        self.tmax = self.times[-1]

        self.info = {
            'sfreq': sfreq,
            'ch_names': ch_names,
            'nchan': len(ch_names),
            'ch_types': self.ch_types
        }

    def __repr__(self):
        n_channels, n_times = self.data.shape
        return (f"<EvokedData | {n_channels} channels, {n_times} time points, "
                f"{self.tmin:.3f} - {self.tmax:.3f} s, sfreq={self.sfreq} Hz>")

    def get_data(self):
        """Return the data array."""
        return self.data


def average_epochs(epochs, picks=None):
    """
    Average epochs to create an evoked response.

    This is the main processing step: averaging many trials together
    to see the typical brain response (ERP/ErrP).

    Parameters:
        epochs: EpochsData object with raw epochs
        picks: indices of channels to include (None = all)

    Returns:
        EvokedData object with averaged data
    """
    if picks is None:
        data_avg = np.mean(epochs.data, axis=0)
        ch_names = epochs.ch_names
        ch_types = epochs.ch_types
    else:
        data_avg = np.mean(epochs.data[:, picks, :], axis=0)
        ch_names = [epochs.ch_names[i] for i in picks]
        ch_types = [epochs.ch_types[i] for i in picks]

    return EvokedData(
        data=data_avg,
        ch_names=ch_names,
        sfreq=epochs.sfreq,
        tmin=epochs.tmin,
        ch_types=ch_types,
        ch_locs=epochs.ch_locs
    )


def select_channels(epochs, channel_indices):
    """
    Select specific channels from epochs data.

    Parameters:
        epochs: EpochsData object
        channel_indices: list of channel indices to keep

    Returns:
        New EpochsData object with selected channels
    """
    from .data_loader import EpochsData

    selected_data = epochs.data[:, channel_indices, :]
    selected_names = [epochs.ch_names[i] for i in channel_indices]
    selected_types = [epochs.ch_types[i] for i in channel_indices]

    selected_locs = None
    if epochs.ch_locs is not None:
        if isinstance(epochs.ch_locs, (list, np.ndarray)):
            selected_locs = [epochs.ch_locs[i] for i in channel_indices]

    return EpochsData(
        data=selected_data,
        ch_names=selected_names,
        sfreq=epochs.sfreq,
        tmin=epochs.tmin,
        events=epochs.events,
        event_id=epochs.event_id,
        ch_types=selected_types,
        ch_locs=selected_locs
    )


def select_time_window(epochs, tmin, tmax):
    """
    Select a specific time window from epochs.

    Parameters:
        epochs: EpochsData object
        tmin: start time in seconds
        tmax: end time in seconds

    Returns:
        New EpochsData object with selected time window
    """
    from .data_loader import EpochsData

    # Find time indices
    start_idx = np.argmin(np.abs(epochs.times - tmin))
    end_idx = np.argmin(np.abs(epochs.times - tmax)) + 1

    # Slice data
    windowed_data = epochs.data[:, :, start_idx:end_idx]

    return EpochsData(
        data=windowed_data,
        ch_names=epochs.ch_names,
        sfreq=epochs.sfreq,
        tmin=tmin,
        events=epochs.events,
        event_id=epochs.event_id,
        ch_types=epochs.ch_types,
        ch_locs=epochs.ch_locs
    )
