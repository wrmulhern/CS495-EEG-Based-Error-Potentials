"""Convert OpenBCI Ganglion CSV recordings to EEGLAB ``.set`` format."""

import logging

import numpy as np
import pandas as pd
from scipy.io import savemat

logger = logging.getLogger(__name__)

GANGLION_SFREQ = 200
GANGLION_N_CHANNELS = 4
GANGLION_CH_LOCS = [
    {'labels': 'TP9',  'X': -0.87, 'Y': -0.31, 'Z': 0.0, 'theta': -110.0, 'radius': 0.9},
    {'labels': 'AF7',  'X': -0.6,  'Y': 0.87,  'Z': 0.0, 'theta': -55.0,  'radius': 0.9},
    {'labels': 'AF8',  'X': 0.6,   'Y': 0.87,  'Z': 0.0, 'theta': 55.0,   'radius': 0.9},
    {'labels': 'TP10', 'X': 0.87,  'Y': -0.31, 'Z': 0.0, 'theta': 110.0,  'radius': 0.9},
]

EVENT_ID = {
    'congruent':    1,
    'incongruent':  2,
    'correct':      3,
    'error':        4,
    'no_response':  5,
}
CODE_NAME = {v: k for k, v in EVENT_ID.items()}


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

    raw = df.iloc[:, 1:5].values.T.astype(np.float64)
    raw = raw / 1e6   # uV -> V

    markers_raw = df.iloc[:, 14].values.astype(np.float64)

    n_channels = GANGLION_N_CHANNELS
    sfreq      = GANGLION_SFREQ
    n_samples  = raw.shape[1]

    stim_codes  = {1, 2}
    stim_samples = [
        (i, int(markers_raw[i]))
        for i in range(n_samples)
        if markers_raw[i] in stim_codes
    ]

    has_markers = len(stim_samples) >= 2

    if has_markers:
        pre_ms, post_ms = 200, 800
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

        if len(epochs_list) < 2:
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
                'setname': 'Flanker_ErrP',
                'nbchan':  n_channels,
                'pnts':    epoch_len,
                'trials':  n_epochs,
                'srate':   float(sfreq),
                'xmin':    tmin_s,
                'xmax':    tmax_s,
                'times':   (np.arange(epoch_len) / sfreq + tmin_s).tolist(),
                'chanlocs': GANGLION_CH_LOCS,
                'ref':     'common',
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
            'setname': 'Ganglion_Recording',
            'nbchan':  n_channels,
            'pnts':    n_samples,
            'trials':  1,
            'srate':   float(sfreq),
            'xmin':    0.0,
            'xmax':    n_samples / sfreq,
            'times':   (np.arange(n_samples) / sfreq).tolist(),
            'chanlocs': GANGLION_CH_LOCS,
            'ref':     'common',
        }
        logger.debug(
            f"No markers found — saved as continuous "
            f"({n_samples/sfreq:.1f}s, {n_samples} samples)"
        )

    output_path = csv_path.replace('.csv', '_converted.set')
    savemat(output_path, {'EEG': EEG}, appendmat=False)
    return output_path
