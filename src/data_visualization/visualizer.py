"""
Matplotlib-based plotting functions for EEG data.

All public functions accept a *theme* parameter (``"light"`` or
``"dark"``) and return a :class:`matplotlib.figure.Figure` that can be
embedded in a Qt canvas via ``FigureCanvasQTAgg``.  When *show* is
``True`` the figure is also displayed interactively with
``plt.show()``.

Plot types
----------
* :func:`plot_epochs` — butterfly overlay of every epoch/channel trace.
* :func:`plot_evoked` — averaged ERP waveform with optional ERN / Pe
  event-window overlays.
* :func:`plot_topomap` — static scalp-voltage maps at up to three
  user-chosen time points.
* :func:`plot_joint` — combined time-series (top) and topomap row
  (bottom).
* :func:`plot_topomap_frame` — single-frame topomap used by the
  animated topomap feature in the GUI.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy import signal
from scipy.interpolate import griddata

logger = logging.getLogger(__name__)

def _apply_mpl_theme(fig, axes, theme: str = "light"):
    """Apply the app's light or dark colour palette to a Matplotlib figure.

    Updates the figure background, suptitle colour, and every supplied
    axes' background, tick / label / spine colours, and grid style.
    This is called once when a figure is first created and again whenever
    the user toggles dark mode.

    Parameters:
        fig (matplotlib.figure.Figure): Target figure.
        axes: A single ``Axes`` or an iterable of ``Axes`` objects.
        theme (str): ``"light"`` or ``"dark"``.
    """
    if isinstance(axes, (list, tuple, np.ndarray)):
        axes_list = list(axes)
    else:
        axes_list = [axes]

    if theme == "dark":
        bg_color = "#121212"
        axis_bg = "#121212"
        text_color = "#e8eaed"
        grid_color = "#3c4043"
    else:
        bg_color = "#ffffff"
        axis_bg = "#ffffff"
        text_color = "#202124"
        grid_color = "#dadce0"

    fig.patch.set_facecolor(bg_color)

    if hasattr(fig, "_suptitle") and fig._suptitle is not None:
        fig._suptitle.set_color(text_color)

    for ax in axes_list:
        ax.set_facecolor(axis_bg)
        ax.tick_params(colors=text_color)
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)
        ax.title.set_color(text_color)
        ax.grid(color=grid_color, alpha=0.3)
        for spine in ax.spines.values():
            spine.set_color(grid_color)


def plot_epochs(epochs, picks=None, scalings='auto', title=None, show=True, theme: str = "light"):
    """Butterfly plot: overlay every epoch/channel trace on a single axis.

    Useful for a quick visual inspection of raw epoch quality before
    averaging.  Each trace is drawn with low alpha so dense regions
    (consistent activity) appear darker than noisy outliers.

    Parameters:
        epochs (EpochsData): Epoched dataset.
        picks (list[int] | None): Channel indices to plot.  ``None``
            plots all channels.
        scalings: Unused — kept for API compatibility.
        title (str | None): Custom axis title.
        show (bool): If ``True``, call ``plt.show()`` after rendering.
        theme (str): ``"light"`` or ``"dark"``.

    Returns:
        matplotlib.figure.Figure
    """
    if picks is None:
        picks = range(len(epochs.ch_names))

    fig, ax = plt.subplots(figsize=(12, 6))

    for epoch_idx in range(epochs.data.shape[0]):
        for ch_idx in picks:
            ax.plot(epochs.times * 1000,
                   epochs.data[epoch_idx, ch_idx, :] * 1e6,
                   alpha=0.3, linewidth=0.5)

    ax.axvline(0, color='k', linestyle='--', linewidth=1, label='Event onset')
    ax.axhline(0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (uV)')
    ax.set_title(title or f'Epochs ({epochs.data.shape[0]} epochs)')
    ax.grid(True, alpha=0.3)

    _apply_mpl_theme(fig, ax, theme=theme)

    if show:
        plt.tight_layout()
        plt.show()

    return fig


def plot_evoked(evoked, picks=None, spatial_colors=False, gfp=False,
                window_title=None, scalings=None, titles=None,
                display_events_responses=False, show=True, theme: str = "light",
                selected_sensors=None):
    """Plot the averaged ERP waveform ("ErrP Time Series" in the GUI).

    Each channel is drawn as a separate line.  When
    *display_events_responses* is ``True``, two shaded bands and
    hover-activated annotation boxes mark the physiologically relevant
    windows:

    * **ERN / Ne** (50–150 ms, blue) — the error-related negativity.
    * **Pe** (200–400 ms, green) — the error positivity.

    Parameters:
        evoked (EvokedData): Averaged waveform to plot.
        picks (list[int] | None): Channel indices.  ``None`` = all.
        spatial_colors (bool): Colour traces by channel position using
            the *viridis* colourmap.
        gfp (bool): Overlay the Global Field Power (spatial ``std``).
        window_title (str | None): Axis title override.
        scalings: Unused — kept for API compatibility.
        titles: Unused — kept for API compatibility.
        display_events_responses (bool): Draw ERN/Pe bands and hover
            annotations.
        show (bool): Call ``plt.show()`` after rendering.
        theme (str): ``"light"`` or ``"dark"``.
        selected_sensors (list[str] | None): Not used directly here
            (channel filtering is applied upstream), but accepted for
            signature consistency with :func:`plot_topomap`.

    Returns:
        matplotlib.figure.Figure
    """
    if picks is None:
        picks = range(len(evoked.ch_names))

    fig, ax = plt.subplots(figsize=(12, 6))

    if spatial_colors:
        colors = plt.cm.viridis(np.linspace(0, 1, len(picks)))
    else:
        colors = None

    for idx, ch_idx in enumerate(picks):
        label = evoked.ch_names[ch_idx] if len(picks) <= 20 else None
        kwargs = dict(label=label, alpha=0.8)
        if colors is not None:
            kwargs['color'] = colors[idx]
        ax.plot(evoked.times * 1000,
               evoked.data[ch_idx, :] * 1e6,
               linewidth=0.8,
               **kwargs)

    if gfp:
        gfp_data = np.std(evoked.data[picks, :], axis=0) * 1e6
        ax.plot(evoked.times * 1000, gfp_data, 'k--', linewidth=2,
               label='GFP', alpha=0.6)

    ax.axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)

    if display_events_responses:
        time_min_ms = evoked.times[0] * 1000
        time_max_ms = evoked.times[-1] * 1000

        if time_min_ms <= 0 <= time_max_ms:
            ax.axvline(0, color='red', linestyle='--', linewidth=2,
                       label='Event', zorder=5)

        ern_start, ern_end = 50, 150
        ern_annotation = None
        if ern_start < time_max_ms and ern_end > time_min_ms:
            ern_display_start = max(ern_start, time_min_ms)
            ern_display_end = min(ern_end, time_max_ms)
            if time_min_ms <= ern_start <= time_max_ms:
                ax.axvline(ern_start, color='blue', linestyle='--', linewidth=1.5, alpha=0.7)
            if time_min_ms <= ern_end <= time_max_ms:
                ax.axvline(ern_end, color='blue', linestyle='--', linewidth=1.5, alpha=0.7)
            ax.axvspan(ern_display_start, ern_display_end, alpha=0.15, color='lightblue', zorder=1)
            ern_annotation = ax.annotate('ERN/Ne\n(50-150ms)\nNegative peak',
                                         xy=((ern_display_start + ern_display_end) / 2, 0),
                                         xytext=((ern_display_start + ern_display_end) / 2,
                                                ax.get_ylim()[0] * 0.7),
                                         ha='center', va='bottom', fontsize=10, color='darkblue',
                                         bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                                                  edgecolor='blue', alpha=0.95, linewidth=2),
                                         visible=False, zorder=10)

        pe_start, pe_end = 200, 400
        pe_annotation = None
        if pe_start < time_max_ms and pe_end > time_min_ms:
            pe_display_start = max(pe_start, time_min_ms)
            pe_display_end = min(pe_end, time_max_ms)
            if time_min_ms <= pe_start <= time_max_ms:
                ax.axvline(pe_start, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
            if time_min_ms <= pe_end <= time_max_ms:
                ax.axvline(pe_end, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
            ax.axvspan(pe_display_start, pe_display_end, alpha=0.15, color='lightgreen', zorder=1)
            pe_annotation = ax.annotate('Pe\n(200-400ms)\nPositive peak',
                                        xy=((pe_display_start + pe_display_end) / 2, 0),
                                        xytext=((pe_display_start + pe_display_end) / 2,
                                               ax.get_ylim()[1] * 0.7),
                                        ha='center', va='top', fontsize=10, color='darkgreen',
                                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                                                 edgecolor='green', alpha=0.95, linewidth=2),
                                        visible=False, zorder=10)

        if ern_annotation is not None or pe_annotation is not None:
            def on_hover(event):
                if event.inaxes != ax:
                    if ern_annotation and ern_annotation.get_visible():
                        ern_annotation.set_visible(False)
                        fig.canvas.draw_idle()
                    if pe_annotation and pe_annotation.get_visible():
                        pe_annotation.set_visible(False)
                        fig.canvas.draw_idle()
                    return
                mouse_x = event.xdata
                if ern_annotation and ern_start <= mouse_x <= ern_end:
                    ern_annotation.set_visible(True)
                    if pe_annotation:
                        pe_annotation.set_visible(False)
                    fig.canvas.draw_idle()
                elif pe_annotation and pe_start <= mouse_x <= pe_end:
                    if ern_annotation:
                        ern_annotation.set_visible(False)
                    pe_annotation.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    changed = False
                    if ern_annotation and ern_annotation.get_visible():
                        ern_annotation.set_visible(False)
                        changed = True
                    if pe_annotation and pe_annotation.get_visible():
                        pe_annotation.set_visible(False)
                        changed = True
                    if changed:
                        fig.canvas.draw_idle()
            fig.canvas.mpl_connect('motion_notify_event', on_hover)

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (uV)')
    ax.set_title(window_title or 'Evoked Response (Average)')
    ax.grid(True, alpha=0.3)

    _apply_mpl_theme(fig, ax, theme=theme)

    if len(picks) <= 20:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8, framealpha=0.95)
    
    if show:
        plt.tight_layout()
        plt.show()

    return fig


def plot_topomap(evoked, times, ch_type='eeg', colorbar=True,
                 cmap='RdBu_r', sensors=True, contours=6, show=True, theme: str = "light",
                 selected_sensors=None):
    """Plot static scalp-voltage maps at one or more time points.

    Each time point produces a filled-contour interpolation of the
    voltage across all electrode positions (cubic ``griddata``), drawn
    inside a head outline with a nose marker.

    Falls back to :func:`plot_evoked` if no channel locations are
    available.

    Parameters:
        evoked (EvokedData): Averaged waveform.
        times (float | list[float]): Time point(s) in seconds at which
            to draw the map.
        ch_type (str): Unused — kept for API compatibility.
        colorbar (bool): Append a colour bar to the right of the maps.
        cmap (str): Matplotlib colourmap name.
        sensors (bool): Draw electrode dots on the map.
        contours (int): Number of contour levels.
        show (bool): Call ``plt.show()`` after rendering.
        theme (str): ``"light"`` or ``"dark"``.
        selected_sensors (list[str] | None): Passed through for
            signature consistency; the topomap always uses all channels
            for interpolation.

    Returns:
        matplotlib.figure.Figure
    """
    if evoked.ch_locs is None:
        logger.warning("No channel locations available; cannot create topomap. Showing simple time-series plot instead.")
        return plot_evoked(evoked, show=show, theme=theme, selected_sensors=selected_sensors)

    if not isinstance(times, (list, np.ndarray)):
        times = [times]

    n_times = len(times)
    fig, axes = plt.subplots(1, n_times, figsize=(4 * n_times, 4))
    if n_times == 1:
        axes = [axes]

    pos = _get_channel_positions(evoked.ch_locs, evoked.ch_names)

    for idx, time in enumerate(times):
        time_idx = np.argmin(np.abs(evoked.times - time))
        actual_time = evoked.times[time_idx]
        data = evoked.data[:, time_idx] * 1e6
        _plot_topomap_single(
            data, pos, axes[idx],
            title=f'{actual_time*1000:.0f} ms',
            cmap=cmap, sensors=sensors, contours=contours
        )

    if colorbar:
        vmin, vmax = data.min(), data.max()
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, orientation='vertical',
                          fraction=0.05, pad=0.04)
        cbar.set_label('Amplitude (uV)')

    plt.suptitle('Topographic Maps', fontsize=14, y=0.95)

    _apply_mpl_theme(fig, axes, theme=theme)

    if show:
        plt.tight_layout()
        plt.show()

    return fig


def plot_joint(evoked, times=None, title='', ts_args=None,
               topomap_args=None, display_events_responses=False,
               show=True, theme: str = "light", selected_sensors=None):
    """Combined time-series and topomap figure ("Joint Maps" in the GUI).

    The figure has a 2-row ``GridSpec``:

    * **Top row** — averaged ERP waveform (same content as
      :func:`plot_evoked`) with vertical markers at each topomap time.
    * **Bottom row** — one topomap per requested time point.  Times
      outside the epoch range are rendered as hatched "Out of range"
      placeholders via :func:`_plot_out_of_range`.

    When *times* is ``None``, up to three peaks are auto-detected from
    the Global Field Power using ``scipy.signal.find_peaks``.

    Parameters:
        evoked (EvokedData): Averaged waveform.
        times (list[float] | None): Topomap time points in seconds.
        title (str): Axis title for the time-series row.
        ts_args: Unused — kept for API compatibility.
        topomap_args: Unused — kept for API compatibility.
        display_events_responses (bool): Draw ERN / Pe overlays on the
            time-series row (same logic as :func:`plot_evoked`).
        show (bool): Call ``plt.show()`` after rendering.
        theme (str): ``"light"`` or ``"dark"``.
        selected_sensors (list[str] | None): If provided (and not
            ``["All Channels"]``), only these channels are drawn in
            the time-series row.  The topomap row always uses all
            channels.

    Returns:
        matplotlib.figure.Figure
    """
    if times is None:
        gfp = np.std(evoked.data, axis=0)
        peak_indices = signal.find_peaks(gfp, distance=int(0.05 * evoked.sfreq))[0]
        if len(peak_indices) > 0:
            times = evoked.times[peak_indices[:3]]
        else:
            times = [0.1, 0.2, 0.3]

    if not isinstance(times, (list, np.ndarray)):
        times = [times]

    # Requested epoch times
    t_min_s = evoked.times[0]
    t_max_s = evoked.times[-1]

    n_topos = len(times)
    fig = plt.figure(figsize=(14, 8))

    gs = fig.add_gridspec(2, n_topos, height_ratios=[2, 1],
                         hspace=0.55, wspace=0.3)

    # ---- Time series (top row, spans all columns) ----
    ax_ts = fig.add_subplot(gs[0, :])

    # Determine which channels to plot in time series
    if selected_sensors and selected_sensors != ["All Channels"]:
        # Plot only selected sensors
        channels_to_plot = []
        for sensor_name in selected_sensors:
            if sensor_name in evoked.ch_names:
                channels_to_plot.append(evoked.ch_names.index(sensor_name))
    else:
        # Plot all channels
        channels_to_plot = list(range(len(evoked.ch_names)))

    for ch_idx in channels_to_plot:
        label = evoked.ch_names[ch_idx] if len(channels_to_plot) <= 20 else None
        ax_ts.plot(evoked.times * 1000, evoked.data[ch_idx, :] * 1e6,
                  alpha=0.5, linewidth=0.8, label=label)

    # Mark only in-range topo times with a red dashed line
    time_min_ms = evoked.times[0] * 1000
    time_max_ms = evoked.times[-1] * 1000
    for time in times:
        time_ms = time * 1000
        if t_min_s <= time <= t_max_s:
            ax_ts.axvline(time_ms, color="r", linestyle="--", linewidth=1.5, alpha=0.7)

    if time_min_ms <= 0 <= time_max_ms:
        ax_ts.axvline(0, color='k', linestyle='-', linewidth=1)
    ax_ts.axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
    ax_ts.set_xlabel('Time (ms)')
    ax_ts.set_ylabel('Amplitude (uV)')
    ax_ts.set_title(title or 'Evoked Response with Topographic Maps')
    ax_ts.grid(True, alpha=0.3)

    # ---- Events / response band overlays ----
    if display_events_responses:
        if time_min_ms <= 0 <= time_max_ms:
            ax_ts.axvline(0, color='red', linestyle='--', linewidth=2,
                           label='Event', zorder=5)

        ern_start, ern_end = 50, 150
        ern_annotation = None
        if ern_start < time_max_ms and ern_end > time_min_ms:
            ern_display_start = max(ern_start, time_min_ms)
            ern_display_end = min(ern_end, time_max_ms)
            if time_min_ms <= ern_start <= time_max_ms:
                ax_ts.axvline(ern_start, color='blue', linestyle='--', linewidth=1.5, alpha=0.7)
            if time_min_ms <= ern_end <= time_max_ms:
                ax_ts.axvline(ern_end, color='blue', linestyle='--', linewidth=1.5, alpha=0.7)
            ax_ts.axvspan(ern_display_start, ern_display_end, alpha=0.15, color='lightblue', zorder=1)
            ern_annotation = ax_ts.annotate('ERN/Ne\n(50-150ms)\nNegative peak',
                                             xy=((ern_display_start + ern_display_end) / 2, 0),
                                             xytext=((ern_display_start + ern_display_end) / 2,
                                                    ax_ts.get_ylim()[0] * 0.7),
                                             ha='center', va='bottom', fontsize=10, color='darkblue',
                                             bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                                                      edgecolor='blue', alpha=0.95, linewidth=2),
                                             visible=False, zorder=10)

        pe_start, pe_end = 200, 400
        pe_annotation = None
        if pe_start < time_max_ms and pe_end > time_min_ms:
            pe_display_start = max(pe_start, time_min_ms)
            pe_display_end = min(pe_end, time_max_ms)
            if time_min_ms <= pe_start <= time_max_ms:
                ax_ts.axvline(pe_start, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
            if time_min_ms <= pe_end <= time_max_ms:
                ax_ts.axvline(pe_end, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
            ax_ts.axvspan(pe_display_start, pe_display_end, alpha=0.15, color='lightgreen', zorder=1)
            pe_annotation = ax_ts.annotate('Pe\n(200-400ms)\nPositive peak',
                                            xy=((pe_display_start + pe_display_end) / 2, 0),
                                            xytext=((pe_display_start + pe_display_end) / 2,
                                                   ax_ts.get_ylim()[1] * 0.7),
                                            ha='center', va='top', fontsize=10, color='darkgreen',
                                            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                                                     edgecolor='green', alpha=0.95, linewidth=2),
                                            visible=False, zorder=10)

        if ern_annotation is not None or pe_annotation is not None:
            def on_hover(event):
                if event.inaxes != ax_ts:
                    if ern_annotation and ern_annotation.get_visible():
                        ern_annotation.set_visible(False)
                        fig.canvas.draw_idle()
                    if pe_annotation and pe_annotation.get_visible():
                        pe_annotation.set_visible(False)
                        fig.canvas.draw_idle()
                    return
                mouse_x = event.xdata
                if ern_annotation and ern_start <= mouse_x <= ern_end:
                    ern_annotation.set_visible(True)
                    if pe_annotation:
                        pe_annotation.set_visible(False)
                    fig.canvas.draw_idle()
                elif pe_annotation and pe_start <= mouse_x <= pe_end:
                    if ern_annotation:
                        ern_annotation.set_visible(False)
                    pe_annotation.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    changed = False
                    if ern_annotation and ern_annotation.get_visible():
                        ern_annotation.set_visible(False)
                        changed = True
                    if pe_annotation and pe_annotation.get_visible():
                        pe_annotation.set_visible(False)
                        changed = True
                    if changed:
                        fig.canvas.draw_idle()
            fig.canvas.mpl_connect('motion_notify_event', on_hover)

    # ---- Topomap row (bottom) ----
    if evoked.ch_locs is not None:
        pos = _get_channel_positions(evoked.ch_locs, evoked.ch_names)
    else:
        pos = None

    for idx, time in enumerate(times):
        ax_topo = fig.add_subplot(gs[1, idx])
        in_range = t_min_s <= time <= t_max_s

        if not in_range:
            # Topo not in range, render a placeholder
            _plot_out_of_range(ax_topo, time_ms=time * 1000,
                               t_min_ms=t_min_s * 1000, t_max_ms=t_max_s * 1000)
        elif pos is not None:
            time_idx = np.argmin(np.abs(evoked.times - time))
            actual_time = evoked.times[time_idx]
            data = evoked.data[:, time_idx] * 1e6
            _plot_topomap_single(
                data, pos, ax_topo,
                title=f'{actual_time * 1000:.0f} ms',
                cmap='RdBu_r', sensors=True, contours=6
            )
        else:
            # No channel locations — fall back to bar plot
            time_idx = np.argmin(np.abs(evoked.times - time))
            actual_time = evoked.times[time_idx]
            data = evoked.data[:, time_idx] * 1e6
            ax_topo.barh(range(len(evoked.ch_names)), data)
            ax_topo.set_yticks(range(len(evoked.ch_names)))
            ax_topo.set_yticklabels(evoked.ch_names, fontsize=6)
            ax_topo.set_xlabel('uV')
            ax_topo.set_title(f'{actual_time * 1000:.0f} ms')
            ax_topo.axvline(0, color='k', linestyle='-', linewidth=0.5)

    # Add legend to time series plot if not too many channels
    if len(channels_to_plot) <= 20:
        ax_ts.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8, framealpha=0.95)

    _apply_mpl_theme(fig, fig.get_axes(), theme=theme)

    if show:
        plt.tight_layout()
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Helper: out-of-range placeholder
# ---------------------------------------------------------------------------

def _plot_out_of_range(ax, time_ms: float, t_min_ms: float, t_max_ms: float):
    """Draw a hatched "Out of range" placeholder in a topomap cell.

    Used by :func:`plot_joint` when a user-specified topomap time falls
    outside the epoch window.  The placeholder shows the requested time
    and the valid range so the user knows what values to enter.

    Parameters:
        ax (matplotlib.axes.Axes): Target axes (will be cleared).
        time_ms (float): The out-of-range time that was requested, in ms.
        t_min_ms (float): Start of the valid epoch window in ms.
        t_max_ms (float): End of the valid epoch window in ms.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')

    # Grey hatched rectangle filling the axes
    rect = patches.FancyBboxPatch(
        (0.05, 0.05), 0.90, 0.90,
        boxstyle="round,pad=0.02",
        linewidth=1.5,
        edgecolor="#aaaaaa",
        facecolor="#f0f0f0",
        hatch="///",
        zorder=1,
    )
    ax.add_patch(rect)

    # Main label
    ax.text(
        0.5, 0.58,
        "Out of range",
        ha="center", va="center",
        fontsize=9, fontweight="bold",
        color="#888888",
        zorder=2,
    )

    # Sub-label: the requested time
    ax.text(
        0.5, 0.40,
        f"{time_ms:.0f} ms",
        ha="center", va="center",
        fontsize=8,
        color="#aaaaaa",
        zorder=2,
    )

    # Title showing the valid window so the user knows what to type
    ax.set_title(
        f"valid: {t_min_ms:.0f}–{t_max_ms:.0f} ms",
        fontsize=7,
        color="#aaaaaa",
        pad=4,
    )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _get_channel_positions(ch_locs, ch_names):
    """Extract 2-D scalp positions from EEGLAB ``chanlocs``.

    Two coordinate systems are supported:

    * **Cartesian** — each element has ``.X`` and ``.Y`` attributes.
    * **Polar** — each element has ``.theta`` (degrees) and ``.radius``
      attributes, converted via ``x = r·cos(θ)``, ``y = r·sin(θ)``.

    Parameters:
        ch_locs: EEGLAB ``chanlocs`` struct array (indexable sequence).
        ch_names (list[str]): Channel labels (only used for length).

    Returns:
        np.ndarray | None: Array of shape ``(n_channels, 2)`` with
        ``[x, y]`` rows, or ``None`` if *ch_locs* is ``None``.
    """
    if ch_locs is None:
        return None

    pos = np.zeros((len(ch_names), 2))

    for idx, ch_name in enumerate(ch_names):
        if hasattr(ch_locs, '__getitem__'):
            loc = ch_locs[idx]
            if hasattr(loc, 'X') and hasattr(loc, 'Y'):
                pos[idx] = [loc.X, loc.Y]
            elif hasattr(loc, 'theta') and hasattr(loc, 'radius'):
                theta = np.deg2rad(loc.theta)
                pos[idx] = [loc.radius * np.cos(theta),
                           loc.radius * np.sin(theta)]

    return pos


def _plot_topomap_single(data, pos, ax, title='', cmap='RdBu_r',
                        sensors=True, contours=6):
    """Render one filled-contour scalp map onto an existing axes.

    The voltage values in *data* are interpolated onto a 100×100 grid
    using ``scipy.interpolate.griddata`` (cubic method), then drawn
    with ``contourf``.  A head-outline circle and nose wedge are
    overlaid for anatomical reference.

    Parameters:
        data (np.ndarray): 1-D array of voltages (µV), one per channel.
        pos (np.ndarray): Electrode positions with shape
            ``(n_channels, 2)``.
        ax (matplotlib.axes.Axes): Target axes.
        title (str): Axes title (usually the time point).
        cmap (str): Matplotlib colourmap name.
        sensors (bool): Draw black dots at electrode positions.
        contours (int): Number of contour fill levels.
    """
    xi = np.linspace(pos[:, 0].min() - 0.1, pos[:, 0].max() + 0.1, 100)
    yi = np.linspace(pos[:, 1].min() - 0.1, pos[:, 1].max() + 0.1, 100)
    Xi, Yi = np.meshgrid(xi, yi)

    Zi = griddata(pos, data, (Xi, Yi), method='cubic')

    vmax = np.abs(data).max()
    im = ax.contourf(Xi, Yi, Zi, levels=contours, cmap=cmap,
                    vmin=-vmax, vmax=vmax)

    head_radius = 1.0
    circle = patches.Circle((0, 0), head_radius, fill=False,
                           edgecolor='k', linewidth=2)
    ax.add_patch(circle)

    nose = patches.Wedge((0, head_radius), 0.2, 60, 120,
                        facecolor='k', edgecolor='k')
    ax.add_patch(nose)

    if sensors:
        ax.plot(pos[:, 0], pos[:, 1], 'ko', markersize=4)

    ax.set_xlim([xi.min(), xi.max()])
    ax.set_ylim([yi.min(), yi.max()])
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title)


def plot_topomap_frame(evoked, time, fig=None, cmap='RdBu_r',
                       sensors=True, contours=6, theme='light',
                       global_vmax=None):
    """Render a single animated-topomap frame onto an existing figure.

    Designed to be called repeatedly by the GUI's ``QTimer`` loop.  On
    each call the figure is cleared and redrawn at the new *time*.
    Reusing the same ``Figure`` object avoids the overhead of creating
    and destroying canvases every frame.

    The colour scale is symmetrically clamped to ±\ *global_vmax* so
    the palette stays fixed across the entire animation, preventing
    misleading colour flicker between frames.  Compute it once before
    the first frame with ``np.abs(evoked.data).max() * 1e6``.

    Layout is manually controlled via ``subplots_adjust`` and
    ``set_anchor('C')`` to keep the round topomap centred in the
    canvas even after the colour-bar steals horizontal space.

    Parameters:
        evoked (EvokedData): Averaged waveform.
        time (float): Time point in seconds.
        fig (matplotlib.figure.Figure | None): Figure to reuse.
            A new one is created if ``None``.
        cmap (str): Matplotlib colourmap name.
        sensors (bool): Draw electrode dots.
        contours (int): Number of contour fill levels.
        theme (str): ``"light"`` or ``"dark"``.
        global_vmax (float | None): Symmetric colour-scale limit in µV.

    Returns:
        matplotlib.figure.Figure: The (possibly newly created) figure.
    """
    if evoked.ch_locs is None:
        logger.warning("No channel locations; cannot render animated topomap.")
        return fig

    pos = _get_channel_positions(evoked.ch_locs, evoked.ch_names)

    time_idx = np.argmin(np.abs(evoked.times - time))
    actual_time = evoked.times[time_idx]
    data = evoked.data[:, time_idx] * 1e6

    if fig is None:
        fig = plt.figure(figsize=(6, 5))
    else:
        fig.clear()

    ax = fig.add_subplot(111)

    vmax = global_vmax if global_vmax is not None else np.abs(data).max()

    xi = np.linspace(pos[:, 0].min() - 0.1, pos[:, 0].max() + 0.1, 100)
    yi = np.linspace(pos[:, 1].min() - 0.1, pos[:, 1].max() + 0.1, 100)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata(pos, data, (Xi, Yi), method='cubic')

    ax.contourf(Xi, Yi, Zi, levels=contours, cmap=cmap,
                vmin=-vmax, vmax=vmax)

    circle = patches.Circle((0, 0), 1.0, fill=False,
                            edgecolor='k', linewidth=2)
    ax.add_patch(circle)
    nose = patches.Wedge((0, 1.0), 0.2, 60, 120,
                         facecolor='k', edgecolor='k')
    ax.add_patch(nose)

    if sensors:
        ax.plot(pos[:, 0], pos[:, 1], 'ko', markersize=4)

    ax.set_xlim([xi.min(), xi.max()])
    ax.set_ylim([yi.min(), yi.max()])
    ax.set_aspect('equal')
    ax.set_anchor('C')
    ax.axis('off')
    ax.set_title(f'{actual_time * 1000:.1f} ms', fontsize=14)

    fig.suptitle('Topographic Map', fontsize=14, y=0.96)

    # Set subplot margins BEFORE colorbar so the colorbar steals from the
    # right edge of a centred region and set_anchor('C') is not overridden.
    fig.subplots_adjust(left=0.08, right=0.88, top=0.86, bottom=0.04)

    norm = plt.Normalize(vmin=-vmax, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical',
                        fraction=0.046, pad=0.04)
    cbar.set_label('Amplitude (µV)')

    _apply_mpl_theme(fig, [ax, cbar.ax], theme=theme)

    return fig
