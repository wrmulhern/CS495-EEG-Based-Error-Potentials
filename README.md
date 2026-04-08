# EEG Based Error Potentials Capstone

Project site: https://aliburkemper12.github.io/Capstone-Project-Site/

## Installation and Deployment Guide

The environment utilizes **uv** as the Python package manager. To install uv, visit this link: [UV Install](https://docs.astral.sh/uv/getting-started/installation/) and follow the instructions for your operating system and terminal.

Once installed, clone the repository using either HTTPS or SSH if you have an SSH key on GitHub:

```bash
git clone (either https or ssh link)
```

To install the required packages, run:

```bash
cd CS495-EEG-Based-Error-Potentials
uv sync
```

Once the above is completed, you can run the application using this command:

```bash
uv run -m src.main
```

However, if you would like to create an executable for deployment instead, you can run this command:
```bash
uv run pyinstaller --onefile --noconsole --name (executable_name) src/main.py
```
which creates an executable of the application for your operating system

---

## ErrP Visualizer — Quick Guide

### What is an ErrP?

An **Error-Related Potential (ErrP)** is a brain signal that appears in EEG when a person perceives or makes an error. Two main components:

- **ERN / Ne** (50–150 ms) — negative deflection shortly after the error, generated in the anterior cingulate cortex.
- **Pe** (200–400 ms) — positive deflection reflecting conscious error awareness.

---

### End-to-End Workflow

1. Click **Record EEG** in the top bar. This opens a web application where you can run a **Flanker Task** — a standard cognitive paradigm that reliably elicits ErrP signals using a connected BCI headset.
2. Complete the task. The web app exports your session as a `.set` or `.csv` file.
3. Drop that file into this app to visualize your ErrP.

---

### Loading Files

- Drag and drop one or more files onto the drop zone, or click **Browse (…)**.
- `.set` files are loaded directly. `.csv` files (e.g. from OpenBCI Ganglion) are **automatically converted** to `.set` format — no manual steps required. A converted file is saved alongside the original CSV.
- Each file opens in its own **tab**. Tabs are fully independent.
- Files load **lazily**: data is only read when you first click **Visualize** on that tab.
- Close a single tab with its **×** button, or remove all tabs with **Clear All**.

---

### Graph Types

- **ErrP Time Series** — averaged ERP waveform across all (or selected) channels. Best for inspecting the ERN and Pe components over time.
- **Topographic Map** — scalp voltage map at up to three time points. Requires ≥19 channels. The epoch window is fixed to the full range.
- **Joint Maps** — time series and topomaps combined in one figure. Topomap times outside the epoch window show as *Out of range* placeholders.

---

### Graph Options

- **Epoch (ms)** — crop the time axis. Leave blank for the full epoch. Disabled automatically for Topographic Map.
- **Sensor** — plot a single channel instead of all channels (Time Series only).
- **Topomap times (s)** — three time points (in seconds) for the scalp maps.
- **Display Events and Responses** — overlays the ERN window (blue, 50–150 ms) and Pe window (green, 200–400 ms) with hover-activated labels.

---

### Downloading a Graph

Click **Download Graph** in the bottom bar to save the currently displayed figure as a high-resolution PNG (300 dpi).

---

### Dark Mode

Toggle **Dark mode** in the top bar. The theme applies to both the Qt UI and the embedded Matplotlib figures.

---

### Supported File Formats

- **EEGLAB .set** — epoched data with ≥2 trials. Companion `.fdt` files are handled automatically.
- **.csv** — OpenBCI Ganglion format. Automatically converted to `.set` on load.
