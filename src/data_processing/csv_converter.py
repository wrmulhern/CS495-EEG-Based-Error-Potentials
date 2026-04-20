"""Convert OpenBCI Ganglion CSV recordings to EEGLAB ``.set`` format."""

import logging

import numpy as np
import pandas as pd
from scipy.io import savemat

from src.config import GANGLION, MARKERS, EPOCH

logger = logging.getLogger(__name__)

GANGLION_SFREQ = GANGLION.sfreq
GANGLION_N_CHANNELS = GANGLION.n_channels
GANGLION_CH_LOCS = list(GANGLION.ch_locs)

EVENT_ID = MARKERS.event_id
CODE_NAME = MARKERS.code_name


def convert_ganglion_csv_to_set(csv_path: str) -> str:
    """Convert an OpenBCI Ganglion ``.csv`` to EEGLAB ``.set`` format.

    If the CSV contains event markers in column 14 (BrainFlow
    layout), the data is epoched around stimulus-onset events
    (markers 1 = congruent, 2 = incongruent) using a -200 ms to
    +800 ms window -- a 1-second epoch that captures both the ERN
    (50-150 ms) and Pe (200-400 ms) ErrP components.

    If no usable markers are found (fewer than 2 stimulus events),
    the full recording is saved as continuous (``trials=1``) data.

    Hardcoded for the 4-channel Ganglion at 200 Hz with approximate
    scalp positions for TP9, AF7, AF8, TP10.

    Parameters:
        csv_path (str): Path to the source ``.csv`` file.

    Returns:
        str: Path to the newly created ``*_converted.set`` file
        (same directory as the input).
    """
    df = pd.read_csv(csv_path, comment='%', header=0, skipinitialspace=True)

    try:
        float(df.columns[0])
        df = pd.read_csv(csv_path, comment='%', header=None, skipinitialspace=True)
        df = df.iloc[1:].reset_index(drop=True)
    except (ValueError, IndexError):
        pass

    eeg_start, eeg_end = GANGLION.eeg_cols[0], GANGLION.eeg_cols[-1] + 1
    raw = df.iloc[:, eeg_start:eeg_end].values.T.astype(np.float64)
    raw = raw * GANGLION.uv_to_v_scale

    markers_raw = df.iloc[:, GANGLION.marker_col].values.astype(np.float64)

    n_channels = GANGLION_N_CHANNELS
    sfreq      = GANGLION_SFREQ
    n_samples  = raw.shape[1]

    stim_codes = MARKERS.stimulus_codes
    stim_samples = [
        (i, int(markers_raw[i]))
        for i in range(n_samples)
        if markers_raw[i] in stim_codes
    ]

    has_markers = len(stim_samples) >= EPOCH.min_stimulus_events

    if has_markers:
        pre_ms, post_ms = EPOCH.pre_stimulus_ms, EPOCH.post_stimulus_ms
        pre_samp  = int(pre_ms  / 1000 * sfreq)
        post_samp = int(post_ms / 1000 * sfreq)
        epoch_len = pre_samp + post_samp
        tmin_s    = -pre_ms  / 1000
        tmax_s    =  post_ms / 1000

        epochs_list = []
        event_rows  = []

        for onset_sample, code in stim_samples:
            start = onset_sample - pre_samp
            end   = onset_sample + post_samp
            if start < 0 or end > n_samples:
                continue

            epoch = raw[:, start:end]
            epochs_list.append(epoch)
            event_rows.append([onset_sample, 0, code])

        if len(epochs_list) < EPOCH.min_valid_epochs:
            has_markers = False
        else:
            epoched = np.stack(epochs_list, axis=0)
            data_3d = np.transpose(epoched, (1, 2, 0)).astype(np.float32)

            n_epochs = data_3d.shape[2]

            eeg_events = [
                {
                    'type':    float(row[2]),
                    'latency': float(row[0]) + 1,
                    'label':   CODE_NAME.get(int(row[2]), str(int(row[2]))),
                }
                for row in event_rows
            ]

            EEG = {
                'data':    data_3d,
                'setname': EPOCH.epoched_set_name,
                'nbchan':  n_channels,
                'pnts':    epoch_len,
                'trials':  n_epochs,
                'srate':   float(sfreq),
                'xmin':    tmin_s,
                'xmax':    tmax_s,
                'times':   (np.arange(epoch_len) / sfreq + tmin_s).tolist(),
                'chanlocs': GANGLION_CH_LOCS,
                'ref':     EPOCH.eeglab_ref,
                'event':   eeg_events,
                'epoch':   [{'event': i + 1} for i in range(n_epochs)],
                'eventdescription': list(CODE_NAME.values()),
            }

            logger.info(
                f"Epoched {n_epochs} trials "
                f"({sum(1 for _, c in stim_samples if c == 1)} congruent, "
                f"{sum(1 for _, c in stim_samples if c == 2)} incongruent) "
                f"window {tmin_s*1000:.0f} to {tmax_s*1000:.0f} ms"
            )

    if not has_markers:
        EEG = {
            'data':    raw.astype(np.float32),
            'setname': EPOCH.continuous_set_name,
            'nbchan':  n_channels,
            'pnts':    n_samples,
            'trials':  1,
            'srate':   float(sfreq),
            'xmin':    0.0,
            'xmax':    n_samples / sfreq,
            'times':   (np.arange(n_samples) / sfreq).tolist(),
            'chanlocs': GANGLION_CH_LOCS,
            'ref':     EPOCH.eeglab_ref,
        }
        logger.debug(
            f"No markers found — saved as continuous "
            f"({n_samples/sfreq:.1f}s, {n_samples} samples)"
        )

    output_path = csv_path.replace('.csv', EPOCH.converted_suffix)
    savemat(output_path, {'EEG': EEG}, appendmat=False)
    return output_path
