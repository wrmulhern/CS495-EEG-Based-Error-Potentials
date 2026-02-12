# TAKING ORIGINAL MNE DEPENDENT EXAMPLE AND MAKING IT WORK WITHOUT MNE BY REVERSE ENGINEERING NECESSARY CODE

# FUNCTIONS WE REVERSE ENGINEERED:


# LOADING
#   mne.io.read_epochs_eeglab() - Load EEGLAB .set files

# AVG
#   epochs.average() - Average across epochs

# VISUALIZING
#   epochs.plot() - Visualize the data
#   epochs.plot_topomap() - Topographic maps
#   epochs.plot_joint() - Combined visualizations





# Basic file for having user upload their own dataset, process the data, and plot -> With MNE

# 1) Import dataset by mounting in google drive
# 2) Determine dataset structure (folders, what type of data)
# 3) Once we have our data loaded, if we need to process (filter, reference, channels, epoching...) do it
# 4) Display our visualizations

# Packages

!pip install numpy scipy matplotlib pybv

# IMPORT

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import signal
from matplotlib import patches
import warnings

# REVERSE ENGINEERED MNE FUNCTIONS

# LOADING
#   OLD: mne.io.read_epochs_eeglab() - Load EEGLAB .set files
#   NEW: read_epochs_eeglab_minimal()

# Constants
CAL = 1e-6  # Calibration factor for EEG data


class Bunch(dict):
    """Dictionary that allows attribute-style access."""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def read_epochs_eeglab_minimal(set_file, verbose=True):
    """
    Read EEGLAB .set file containing epoched data.

    This is a minimal implementation that reads the essential fields
    from EEGLAB format without requiring MNE.

    Parameters:
        set_file: path to .set file
        verbose: print debugging information

    Returns:
        EpochsData object
    """
    set_file = Path(set_file)

    if verbose:
        print(f"Loading file: {set_file}")

    # Load MATLAB file
    mat = loadmat(set_file, squeeze_me=True, struct_as_record=False)

    if verbose:
        print(f"Keys in .mat file: {list(mat.keys())}")

    # Get EEG structure
    eeg = mat.get("EEG", mat)
    if isinstance(eeg, dict) and "EEG" in eeg:
        eeg = eeg["EEG"]
    if isinstance(eeg, dict):
        eeg = Bunch(**eeg)

    if verbose:
        print(f"EEG structure type: {type(eeg)}")
        if hasattr(eeg, '__dict__'):
            print(f"EEG attributes: {list(eeg.__dict__.keys())[:10]}...")  # First 10

    # Check if data is epoched
    trials = getattr(eeg, 'trials', 1)
    if verbose:
        print(f"Number of trials: {trials}")

    if int(trials) <= 1:
        raise ValueError("File does not contain epochs. This file appears to have continuous data.")

    # Extract data (already in epochs format)
    # EEGLAB format: (channels, timepoints, epochs)
    # We need: (epochs, channels, timepoints)

    if not hasattr(eeg, 'data'):
        raise ValueError("EEG structure does not contain 'data' field")

    data = eeg.data

    if verbose:
        print(f"Raw data type: {type(data)}")
        print(f"Data shape/size: {getattr(data, 'shape', getattr(data, 'size', 'unknown'))}")

    # Handle case where data is stored in separate .fdt file
    if isinstance(data, str):
        if verbose:
            print(f"Data is stored externally in file: {data}")

        # Get data file path
        data_file = Path(data)
        if not data_file.is_absolute():
            # Relative path - combine with .set file directory
            data_file = set_file.parent / data_file

        if verbose:
            print(f"Loading data from: {data_file}")

        # Check if file has .fdt extension
        if not data_file.suffix:
            data_file = data_file.with_suffix('.fdt')

        # Load binary data
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")

        # Get dimensions from EEG structure
        n_channels = int(eeg.nbchan)
        n_times = int(eeg.pnts)
        n_epochs = int(eeg.trials)

        if verbose:
            print(f"Expected dimensions: {n_channels} channels x {n_times} timepoints x {n_epochs} epochs")

        # Read binary data (EEGLAB uses float32)
        data = np.fromfile(str(data_file), dtype=np.float32)

        if verbose:
            print(f"Loaded {len(data)} values from .fdt file")

        # Reshape to EEGLAB format: (channels, timepoints, epochs)
        try:
            data = data.reshape(n_channels, n_times, n_epochs, order='F')  # Fortran order (column-major)
        except ValueError as e:
            # Try C order if Fortran fails
            try:
                data = data.reshape(n_channels, n_times, n_epochs, order='C')
                if verbose:
                    print("Note: Using C-order reshape instead of Fortran-order")
            except ValueError:
                raise ValueError(f"Cannot reshape data. Expected {n_channels}x{n_times}x{n_epochs} = "
                               f"{n_channels*n_times*n_epochs}, but got {len(data)} values")

    # Convert to numpy array if it isn't already
    if not isinstance(data, np.ndarray):
        data = np.array(data)
        if verbose:
            print(f"Converted to numpy array, shape: {data.shape}")

    # Ensure we have 3D data
    if data.ndim == 2:
        # Single epoch case: (channels, timepoints) -> (1, channels, timepoints)
        if verbose:
            print("Data is 2D, treating as single epoch")
        data = data[np.newaxis, :, :]
    elif data.ndim == 3:
        # Multiple epochs: transpose from EEGLAB format
        # EEGLAB: (channels, timepoints, epochs)
        # Our format: (epochs, channels, timepoints)
        if verbose:
            print(f"Data is 3D with shape {data.shape}, transposing to (epochs, channels, times)")
        data = np.transpose(data, (2, 0, 1))
    else:
        raise ValueError(f"Unexpected data dimensions: {data.shape}")

    if verbose:
        print(f"Final data shape: {data.shape} (epochs, channels, times)")

    # Extract channel information
    ch_names = []
    ch_locs = None

    if hasattr(eeg, 'chanlocs') and eeg.chanlocs is not None:
        chanlocs = eeg.chanlocs

        # Handle different formats
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

        if verbose:
            print(f"Found {len(ch_names)} channels: {ch_names[:5]}..." if len(ch_names) > 5 else f"Channels: {ch_names}")
    else:
        # No channel info, create default names
        n_channels = data.shape[1]
        ch_names = [f'Ch{i+1}' for i in range(n_channels)]
        if verbose:
            print(f"No channel info found, created {len(ch_names)} default channel names")

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
        if verbose:
            print("Warning: Could not find xmin/tmin, assuming 0.0")

    if verbose:
        print(f"Sampling rate: {sfreq} Hz")
        print(f"Epoch start time: {xmin} seconds")

    # Extract events
    events = None
    event_id = {}

    if hasattr(eeg, 'event') and eeg.event is not None:
        # Parse events (this would need more work for full compatibility)
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

# REVERSE ENGINEERED MNE FUNCTIONS

# LOADING
#   OLD: mne.io.read_epochs_eeglab() - Load EEGLAB .set files
#   NEW: read_epochs_eeglab_minimal()

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

    def average(self, picks=None):
        """
        Average epochs to create an evoked response.

        Parameters:
            picks: indices of channels to include (None = all)

        Returns:
            EvokedData object with averaged data
        """
        if picks is None:
            data_avg = np.mean(self.data, axis=0)
            ch_names = self.ch_names
            ch_types = self.ch_types
        else:
            data_avg = np.mean(self.data[:, picks, :], axis=0)
            ch_names = [self.ch_names[i] for i in picks]
            ch_types = [self.ch_types[i] for i in picks]

        return EvokedData(
            data=data_avg,
            ch_names=ch_names,
            sfreq=self.sfreq,
            tmin=self.tmin,
            ch_types=ch_types,
            ch_locs=self.ch_locs
        )

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

    def plot(self, picks=None, scalings='auto', title=None, show=True):
        """
        Plot all channels for all epochs (butterfly plot).

        Parameters:
            picks: channels to plot (None = all)
            scalings: scaling factor for display
            title: plot title
            show: whether to display the plot
        """
        if picks is None:
            picks = range(len(self.ch_names))

        fig, ax = plt.subplots(figsize=(12, 6))

        for epoch_idx in range(self.data.shape[0]):
            for ch_idx in picks:
                ax.plot(self.times * 1000,  # Convert to ms
                       self.data[epoch_idx, ch_idx, :] * 1e6,  # Convert to µV
                       alpha=0.3, linewidth=0.5)

        ax.axvline(0, color='k', linestyle='--', linewidth=1, label='Event onset')
        ax.axhline(0, color='k', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude (µV)')
        ax.set_title(title or f'Epochs ({self.data.shape[0]} epochs)')
        ax.grid(True, alpha=0.3)

        if show:
            plt.tight_layout()
            plt.show()

        return fig


class EvokedData:
    """
    Container for averaged EEG data (evoked response).

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

    def plot(self, picks=None, spatial_colors=False, gfp=False,
             window_title=None, scalings=None, titles=None, show=True):
        """
        Plot evoked response (ERP/ErrP).

        Parameters:
            picks: channels to plot (None = all)
            spatial_colors: use different colors per channel
            gfp: show global field power
            window_title: figure window title
            scalings: dict of scaling factors
            titles: channel titles
            show: whether to display the plot
        """
        if picks is None:
            picks = range(len(self.ch_names))

        fig, ax = plt.subplots(figsize=(12, 6))

        # Get colors
        if spatial_colors:
            colors = plt.cm.viridis(np.linspace(0, 1, len(picks)))
        else:
            colors = ['C0'] * len(picks)

        # Plot each channel
        for idx, ch_idx in enumerate(picks):
            label = self.ch_names[ch_idx] if len(picks) <= 20 else None
            ax.plot(self.times * 1000,  # Convert to ms
                   self.data[ch_idx, :] * 1e6,  # Convert to µV
                   color=colors[idx], label=label, alpha=0.8)

        # Add GFP if requested
        if gfp:
            gfp_data = np.std(self.data[picks, :], axis=0) * 1e6
            ax.plot(self.times * 1000, gfp_data, 'k--', linewidth=2,
                   label='GFP', alpha=0.6)

        ax.axvline(0, color='k', linestyle='--', linewidth=1)
        ax.axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude (µV)')
        ax.set_title(window_title or 'Evoked Response (Average)')
        ax.grid(True, alpha=0.3)

        if len(picks) <= 20:
            ax.legend(loc='best', fontsize=8)

        if show:
            plt.tight_layout()
            plt.show()

        return fig

    def plot_topomap(self, times, ch_type='eeg', colorbar=True,
                     cmap='RdBu_r', sensors=True, contours=6, show=True):
        """
        Plot topographic maps at specific time points.

        Parameters:
            times: list of time points to plot (in seconds)
            ch_type: channel type to plot
            colorbar: whether to show colorbar
            cmap: colormap name
            sensors: whether to show sensor positions
            contours: number of contour lines
            show: whether to display the plot
        """
        if self.ch_locs is None:
            print("Warning: No channel locations available. Cannot create topomap.")
            print("Showing simple time-series plot instead.")
            return self.plot(show=show)

        if not isinstance(times, (list, np.ndarray)):
            times = [times]

        n_times = len(times)
        fig, axes = plt.subplots(1, n_times, figsize=(4 * n_times, 4))
        if n_times == 1:
            axes = [axes]

        # Get channel positions
        pos = self._get_channel_positions()

        for idx, time in enumerate(times):
            # Find closest time index
            time_idx = np.argmin(np.abs(self.times - time))
            actual_time = self.times[time_idx]

            # Get data at this time point
            data = self.data[:, time_idx] * 1e6  # Convert to µV

            # Create topomap
            self._plot_topomap_single(
                data, pos, axes[idx],
                title=f'{actual_time*1000:.0f} ms',
                cmap=cmap, sensors=sensors, contours=contours
            )

        if colorbar:
            # Add shared colorbar
            vmin, vmax = data.min(), data.max()
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=axes, orientation='vertical',
                              fraction=0.05, pad=0.04)
            cbar.set_label('Amplitude (µV)')

        plt.suptitle('Topographic Maps', fontsize=14, y=1.02)

        if show:
            plt.tight_layout()
            plt.show()

        return fig

    def plot_joint(self, times=None, title='', ts_args=None,
                   topomap_args=None, show=True):
        """
        Plot evoked response with topomaps at specific time points.

        Parameters:
            times: list of time points for topomaps
            title: figure title
            ts_args: arguments for time series plot
            topomap_args: arguments for topomap
            show: whether to display the plot
        """
        if times is None:
            # Default times: peaks in GFP
            gfp = np.std(self.data, axis=0)
            peak_indices = signal.find_peaks(gfp, distance=int(0.05 * self.sfreq))[0]
            if len(peak_indices) > 0:
                times = self.times[peak_indices[:3]]  # Take first 3 peaks
            else:
                times = [0.1, 0.2, 0.3]

        if not isinstance(times, (list, np.ndarray)):
            times = [times]

        # Create figure with custom layout
        n_topos = len(times)
        fig = plt.figure(figsize=(14, 8))

        # Create grid: top row for time series, bottom for topomaps
        gs = fig.add_gridspec(2, n_topos, height_ratios=[2, 1],
                             hspace=0.3, wspace=0.3)

        # Time series plot (spans all columns)
        ax_ts = fig.add_subplot(gs[0, :])

        # Plot all channels
        for ch_idx in range(len(self.ch_names)):
            ax_ts.plot(self.times * 1000, self.data[ch_idx, :] * 1e6,
                      alpha=0.5, linewidth=0.8)

        # Mark time points
        for time in times:
            ax_ts.axvline(time * 1000, color='r', linestyle='--',
                         linewidth=1.5, alpha=0.7)

        ax_ts.axvline(0, color='k', linestyle='-', linewidth=1)
        ax_ts.axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
        ax_ts.set_xlabel('Time (ms)')
        ax_ts.set_ylabel('Amplitude (µV)')
        ax_ts.set_title(title or 'Evoked Response with Topographic Maps')
        ax_ts.grid(True, alpha=0.3)

        # Topomaps
        if self.ch_locs is not None:
            pos = self._get_channel_positions()

            for idx, time in enumerate(times):
                ax_topo = fig.add_subplot(gs[1, idx])

                # Find closest time index
                time_idx = np.argmin(np.abs(self.times - time))
                actual_time = self.times[time_idx]

                # Get data
                data = self.data[:, time_idx] * 1e6

                # Plot topomap
                self._plot_topomap_single(
                    data, pos, ax_topo,
                    title=f'{actual_time*1000:.0f} ms',
                    cmap='RdBu_r', sensors=True, contours=6
                )
        else:
            # If no channel locations, show time slices as bar plots
            for idx, time in enumerate(times):
                ax_bar = fig.add_subplot(gs[1, idx])
                time_idx = np.argmin(np.abs(self.times - time))
                actual_time = self.times[time_idx]
                data = self.data[:, time_idx] * 1e6

                ax_bar.barh(range(len(self.ch_names)), data)
                ax_bar.set_yticks(range(len(self.ch_names)))
                ax_bar.set_yticklabels(self.ch_names, fontsize=6)
                ax_bar.set_xlabel('µV')
                ax_bar.set_title(f'{actual_time*1000:.0f} ms')
                ax_bar.axvline(0, color='k', linestyle='-', linewidth=0.5)

        if show:
            plt.tight_layout()
            plt.show()

        return fig

    def _get_channel_positions(self):
        """Extract 2D channel positions from ch_locs."""
        if self.ch_locs is None:
            return None

        pos = np.zeros((len(self.ch_names), 2))

        for idx, ch_name in enumerate(self.ch_names):
            if hasattr(self.ch_locs, '__getitem__'):
                # ch_locs is array-like
                loc = self.ch_locs[idx]
                if hasattr(loc, 'X') and hasattr(loc, 'Y'):
                    pos[idx] = [loc.X, loc.Y]
                elif hasattr(loc, 'theta') and hasattr(loc, 'radius'):
                    # Convert polar to cartesian
                    theta = np.deg2rad(loc.theta)
                    pos[idx] = [loc.radius * np.cos(theta),
                               loc.radius * np.sin(theta)]

        return pos

    def _plot_topomap_single(self, data, pos, ax, title='', cmap='RdBu_r',
                            sensors=True, contours=6):
        """
        Plot a single topomap.

        Parameters:
            data: channel values to plot
            pos: channel positions (n_channels, 2)
            ax: matplotlib axis
            title: subplot title
            cmap: colormap
            sensors: whether to show sensor positions
            contours: number of contour lines
        """
        # Create interpolation grid
        xi = np.linspace(pos[:, 0].min() - 0.1, pos[:, 0].max() + 0.1, 100)
        yi = np.linspace(pos[:, 1].min() - 0.1, pos[:, 1].max() + 0.1, 100)
        Xi, Yi = np.meshgrid(xi, yi)

        # Interpolate data
        from scipy.interpolate import griddata
        Zi = griddata(pos, data, (Xi, Yi), method='cubic')

        # Plot
        vmax = np.abs(data).max()
        im = ax.contourf(Xi, Yi, Zi, levels=contours, cmap=cmap,
                        vmin=-vmax, vmax=vmax)

        # Add head outline (circle)
        head_radius = 1.0
        circle = patches.Circle((0, 0), head_radius, fill=False,
                               edgecolor='k', linewidth=2)
        ax.add_patch(circle)

        # Add nose
        nose = patches.Wedge((0, head_radius), 0.2, 60, 120,
                            facecolor='k', edgecolor='k')
        ax.add_patch(nose)

        # Show sensors
        if sensors:
            ax.plot(pos[:, 0], pos[:, 1], 'ko', markersize=4)

        ax.set_xlim([xi.min(), xi.max()])
        ax.set_ylim([yi.min(), yi.max()])
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title)

# UPLOAD A DATASET FILE

# mounting drive -> file is in my drive

from google.colab import drive
drive.mount('/content/drive')

# My path

DATA_ROOT = "/content/drive/MyDrive/Capstone_data/EEG_dataset"

# LOAD OUR DATASET

# I am using this one: https://openneuro.org/datasets/ds004771/versions/1.0.0

# Assuming file is EDF
# Preload true for filtering enablement

subject = "sub-002"
folder = "eeg"
file_name = subject + "_task-PY_eeg.set"
eeg_file = os.path.join(DATA_ROOT, subject, folder, file_name)


# this dataset already has epochs, so no need to preprocess


#
#
# MNE FUNCTION. REVERSE ENGINEER #1
#
#
#epochs = mne.io.read_epochs_eeglab(eeg_file)

# NEW:
# from errp_plotter_no_mne import read_epochs_eeglab_minimal
epochs = read_epochs_eeglab_minimal(eeg_file)
# errp_evoked = epochs.average()
# errp_evoked.plot()
# errp_evoked.plot_topomap(times=[0.1, 0.2, 0.3])
# errp_evoked.plot_joint(times=[0.1, 0.2, 0.3])

# checking data details



#
#
# MNE FUNCTION. REVERSE ENGINEER #1
#
#
# epochs.info['sfreq']
# epochs.ch_names




# NEW:
epochs.info['sfreq']
epochs.ch_names
epochs.event_id
epochs.average().plot()
epochs.to_data_frame()

# AVERAGE ErrP

# print(epochs.to_data_frame().shape)

#
#
# MNE FUNCTION. REVERSE ENGINEER #1
#
#
errp_evoked = epochs.average()



errp_evoked.plot()  # important channls

# I am confused

# TOPOMAP


#
#
# MNE FUNCTION. REVERSE ENGINEER #1
#
#
errp_evoked.plot_topomap(times=[0.1, 0.2, 0.3])

errp_evoked.plot_joint(times=[0.1, 0.2, 0.3])