"""
Visualization functions for EEG data
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy import signal
from scipy.interpolate import griddata


def plot_epochs(epochs, picks=None, scalings='auto', title=None, show=True):
    """
    Plot all channels for all epochs (butterfly plot).

    Parameters:
        epochs: EpochsData object
        picks: channels to plot (None = all)
        scalings: scaling factor for display
        title: plot title
        show: whether to display the plot

    Returns:
        matplotlib figure
    """
    if picks is None:
        picks = range(len(epochs.ch_names))

    fig, ax = plt.subplots(figsize=(12, 6))

    for epoch_idx in range(epochs.data.shape[0]):
        for ch_idx in picks:
            ax.plot(epochs.times * 1000,  # Convert to ms
                   epochs.data[epoch_idx, ch_idx, :] * 1e6,  # Convert to uV
                   alpha=0.3, linewidth=0.5)

    ax.axvline(0, color='k', linestyle='--', linewidth=1, label='Event onset')
    ax.axhline(0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (uV)')
    ax.set_title(title or f'Epochs ({epochs.data.shape[0]} epochs)')
    ax.grid(True, alpha=0.3)

    if show:
        plt.tight_layout()
        plt.show()

    return fig


def plot_evoked(evoked, picks=None, spatial_colors=False, gfp=False,
                window_title=None, scalings=None, titles=None, show=True):
    """
    Plot evoked response (ERP/ErrP).

    Parameters:
        evoked: EvokedData object
        picks: channels to plot (None = all)
        spatial_colors: use different colors per channel
        gfp: show global field power
        window_title: figure window title
        scalings: dict of scaling factors
        titles: channel titles
        show: whether to display the plot

    Returns:
        matplotlib figure
    """
    if picks is None:
        picks = range(len(evoked.ch_names))

    fig, ax = plt.subplots(figsize=(12, 6))

    # Get colors
    if spatial_colors:
        colors = plt.cm.viridis(np.linspace(0, 1, len(picks)))
    else:
        colors = ['C0'] * len(picks)

    # Plot each channel
    for idx, ch_idx in enumerate(picks):
        label = evoked.ch_names[ch_idx] if len(picks) <= 20 else None
        ax.plot(evoked.times * 1000,  # Convert to ms
               evoked.data[ch_idx, :] * 1e6,  # Convert to uV
               color=colors[idx], label=label, alpha=0.8)

    # Add GFP if requested
    if gfp:
        gfp_data = np.std(evoked.data[picks, :], axis=0) * 1e6
        ax.plot(evoked.times * 1000, gfp_data, 'k--', linewidth=2,
               label='GFP', alpha=0.6)

    ax.axvline(0, color='k', linestyle='--', linewidth=1)
    ax.axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (uV)')
    ax.set_title(window_title or 'Evoked Response (Average)')
    ax.grid(True, alpha=0.3)

    if len(picks) <= 20:
        ax.legend(loc='best', fontsize=8)

    if show:
        plt.tight_layout()
        plt.show()

    return fig


def plot_topomap(evoked, times, ch_type='eeg', colorbar=True,
                 cmap='RdBu_r', sensors=True, contours=6, show=True):
    """
    Plot topographic maps at specific time points.

    Parameters:
        evoked: EvokedData object
        times: list of time points to plot (in seconds)
        ch_type: channel type to plot
        colorbar: whether to show colorbar
        cmap: colormap name
        sensors: whether to show sensor positions
        contours: number of contour lines
        show: whether to display the plot

    Returns:
        matplotlib figure
    """
    if evoked.ch_locs is None:
        print("Warning: No channel locations available. Cannot create topomap.")
        print("Showing simple time-series plot instead.")
        return plot_evoked(evoked, show=show)

    if not isinstance(times, (list, np.ndarray)):
        times = [times]

    n_times = len(times)
    fig, axes = plt.subplots(1, n_times, figsize=(4 * n_times, 4))
    if n_times == 1:
        axes = [axes]

    # Get channel positions
    pos = _get_channel_positions(evoked.ch_locs, evoked.ch_names)

    for idx, time in enumerate(times):
        # Find closest time index
        time_idx = np.argmin(np.abs(evoked.times - time))
        actual_time = evoked.times[time_idx]

        # Get data at this time point
        data = evoked.data[:, time_idx] * 1e6  # Convert to uV

        # Create topomap
        _plot_topomap_single(
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
        cbar.set_label('Amplitude (uV)')

    plt.suptitle('Topographic Maps', fontsize=14, y=1.02)

    if show:
        plt.tight_layout()
        plt.show()

    return fig


def plot_joint(evoked, times=None, title='', ts_args=None,
               topomap_args=None, show=True):
    """
    Plot evoked response with topomaps at specific time points.

    Parameters:
        evoked: EvokedData object
        times: list of time points for topomaps
        title: figure title
        ts_args: arguments for time series plot
        topomap_args: arguments for topomap
        show: whether to display the plot

    Returns:
        matplotlib figure
    """
    if times is None:
        # Default times: peaks in GFP
        gfp = np.std(evoked.data, axis=0)
        peak_indices = signal.find_peaks(gfp, distance=int(0.05 * evoked.sfreq))[0]
        if len(peak_indices) > 0:
            times = evoked.times[peak_indices[:3]]  # Take first 3 peaks
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
    for ch_idx in range(len(evoked.ch_names)):
        ax_ts.plot(evoked.times * 1000, evoked.data[ch_idx, :] * 1e6,
                  alpha=0.5, linewidth=0.8)

    # Mark time points
    for time in times:
        ax_ts.axvline(time * 1000, color='r', linestyle='--',
                     linewidth=1.5, alpha=0.7)

    ax_ts.axvline(0, color='k', linestyle='-', linewidth=1)
    ax_ts.axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
    ax_ts.set_xlabel('Time (ms)')
    ax_ts.set_ylabel('Amplitude (uV)')
    ax_ts.set_title(title or 'Evoked Response with Topographic Maps')
    ax_ts.grid(True, alpha=0.3)

    # Topomaps
    if evoked.ch_locs is not None:
        pos = _get_channel_positions(evoked.ch_locs, evoked.ch_names)

        for idx, time in enumerate(times):
            ax_topo = fig.add_subplot(gs[1, idx])

            # Find closest time index
            time_idx = np.argmin(np.abs(evoked.times - time))
            actual_time = evoked.times[time_idx]

            # Get data
            data = evoked.data[:, time_idx] * 1e6

            # Plot topomap
            _plot_topomap_single(
                data, pos, ax_topo,
                title=f'{actual_time*1000:.0f} ms',
                cmap='RdBu_r', sensors=True, contours=6
            )
    else:
        # If no channel locations, show time slices as bar plots
        for idx, time in enumerate(times):
            ax_bar = fig.add_subplot(gs[1, idx])
            time_idx = np.argmin(np.abs(evoked.times - time))
            actual_time = evoked.times[time_idx]
            data = evoked.data[:, time_idx] * 1e6

            ax_bar.barh(range(len(evoked.ch_names)), data)
            ax_bar.set_yticks(range(len(evoked.ch_names)))
            ax_bar.set_yticklabels(evoked.ch_names, fontsize=6)
            ax_bar.set_xlabel('uV')
            ax_bar.set_title(f'{actual_time*1000:.0f} ms')
            ax_bar.axvline(0, color='k', linestyle='-', linewidth=0.5)

    if show:
        plt.tight_layout()
        plt.show()

    return fig


# Helper functions

def _get_channel_positions(ch_locs, ch_names):
    """Extract 2D channel positions from ch_locs."""
    if ch_locs is None:
        return None

    pos = np.zeros((len(ch_names), 2))

    for idx, ch_name in enumerate(ch_names):
        if hasattr(ch_locs, '__getitem__'):
            # ch_locs is array-like
            loc = ch_locs[idx]
            if hasattr(loc, 'X') and hasattr(loc, 'Y'):
                pos[idx] = [loc.X, loc.Y]
            elif hasattr(loc, 'theta') and hasattr(loc, 'radius'):
                # Convert polar to cartesian
                theta = np.deg2rad(loc.theta)
                pos[idx] = [loc.radius * np.cos(theta),
                           loc.radius * np.sin(theta)]

    return pos


def _plot_topomap_single(data, pos, ax, title='', cmap='RdBu_r',
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