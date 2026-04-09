"""
Epoch-level processing utilities for EEG data.

The central operation is :func:`average_epochs`, which collapses the
epoch dimension of an :class:`~.data_loader.EpochsData` object to
produce an :class:`EvokedData` (the averaged event-related potential).

Helper functions :func:`select_channels` and :func:`select_time_window`
return new :class:`~.data_loader.EpochsData` objects restricted to a
subset of channels or a narrower time range, respectively.
"""

import numpy as np


class EvokedData:
    """Container for an averaged EEG response (ERP / ErrP).

    Produced by :func:`average_epochs`, this holds the mean waveform
    across all epochs for each channel — i.e. the classic
    *event-related potential*.

    Attributes:
        data (np.ndarray): Averaged sample matrix with shape
            ``(n_channels, n_times)``.  Values are in Volts.
        ch_names (list[str]): Channel labels matching the first axis of
            *data*.
        sfreq (float): Sampling frequency in Hz.
        tmin (float): Epoch start time in seconds.
        tmax (float): Epoch end time in seconds.
        times (np.ndarray): 1-D time vector for the epoch in seconds.
        ch_types (list[str]): Per-channel type strings (default ``'eeg'``).
        ch_locs: EEGLAB ``chanlocs`` structs, or ``None``.
        info (dict): Convenience metadata dict with keys ``sfreq``,
            ``ch_names``, ``nchan``, and ``ch_types``.
    """

    def __init__(self, data, ch_names, sfreq, tmin, ch_types=None, ch_locs=None):
        """
        Parameters:
            data (np.ndarray): Array with shape ``(n_channels, n_times)``.
            ch_names (list[str]): Channel labels.
            sfreq (float): Sampling frequency in Hz.
            tmin (float): Start time of the epoch in seconds.
            ch_types (list[str] | None): Per-channel type strings.
                Defaults to ``['eeg'] * n_channels``.
            ch_locs: EEGLAB ``chanlocs`` struct array, or ``None``.
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
        """Return the raw ``(n_channels, n_times)`` sample array."""
        return self.data


def average_epochs(epochs, picks=None):
    """Compute the mean waveform across epochs to produce an ERP.

    This is the core scientific operation: collapsing the epoch
    (trial) dimension via ``np.mean(data, axis=0)`` so that random
    noise cancels out and the time-locked brain response emerges.

    Parameters:
        epochs (EpochsData): Epoched dataset.
        picks (list[int] | None): Channel indices to include in the
            average.  When ``None`` (the default) all channels are used.

    Returns:
        EvokedData: Averaged waveform with shape
        ``(len(picks), n_times)`` (or ``(n_channels, n_times)`` when
        *picks* is ``None``).  Channel locations are copied through so
        that topographic plotting remains possible.
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
    """Return a new :class:`~.data_loader.EpochsData` restricted to the
    given channels.

    All other metadata (timing, events, sampling rate) is preserved.
    Channel locations are sliced to match when available.

    Parameters:
        epochs (EpochsData): Source epoched dataset.
        channel_indices (list[int]): Indices into ``epochs.ch_names``
            selecting the channels to keep.

    Returns:
        EpochsData: A new object whose ``data`` has shape
        ``(n_epochs, len(channel_indices), n_times)``.
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
    """Return a new :class:`~.data_loader.EpochsData` cropped to a time
    window.

    The nearest sample to *tmin* and *tmax* is found via
    ``np.argmin(|times − t|)``, so the actual window may be up to
    ±0.5 samples wider than requested.

    Parameters:
        epochs (EpochsData): Source epoched dataset.
        tmin (float): Desired start time in seconds.
        tmax (float): Desired end time in seconds.

    Returns:
        EpochsData: A new object whose time axis spans approximately
        ``[tmin, tmax]``.  Channel and event metadata are preserved.
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
