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
uv run pyinstaller ErrPVisualizer.spec
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
### Record EEG — Flanker Task
 
The **Record EEG** button opens a built-in Flanker Task that simultaneously streams and records EEG data from an OpenBCI Ganglion board.
 
#### Hardware Required
 
- OpenBCI Ganglion board
- OpenBCI USB Bluetooth dongle (plugged into your computer)
- 3 EEG electrodes (forehead placement recommended — left brow, center forehead, right brow)
- 1 earclip electrode (used as reference, connected to the **D_G** pin on the Ganglion)
 
#### Ganglion LED Status
 
| LED behavior | Meaning |
|---|---|
| **Blinking blue** | Powered on, advertising, ready to connect |
| **Solid blue** | Connected to your computer |
| **No light** | Board is off — check power switch |
 
The board must be **blinking blue** before you attempt to connect. If it is solid blue it may already be connected to another session — power cycle it (flip the switch off and back on).
 
#### Finding Your Serial Port
 
The app uses **pyserial** to automatically detect available ports and populate the dropdown. If no ports appear, make sure the USB dongle is plugged in before opening the app, then use the **↺** refresh button.
 
**macOS** — the port will look like:
```
/dev/cu.usbmodem11
/dev/cu.usbserial-XXXXXXXX
```
 
**Windows** — the port will look like:
```
COM3
COM4
```
 
If you are unsure which port is correct, open the **OpenBCI GUI**, attempt to connect, and note which port it uses successfully. Use that exact port in the Record EEG setup screen.
 
#### Electrode Placement
 
For ErrP recording, place electrodes across the **forehead**:
 
| Electrode | Placement | Ganglion Pin |
|---|---|---|
| Left forehead | Above left eyebrow | **1** |
| Center forehead | Middle of forehead | **2** |
| Right forehead | Above right eyebrow | **3** |
| Earclip (reference) | Either earlobe | **D_G** |
 
 
#### Running the Task
 
1. Plug in the USB dongle and power on the Ganglion board. Wait for the blue LED to **blink**
2. Click **Record EEG** in the top bar
3. Select your serial port from the dropdown (use **↺** to refresh if needed)
4. Set the number of trials (default 100, ~8 minutes)
5. Click **Connect & Start** — the app scans for the Ganglion over Bluetooth. You may be prompted to allow Bluetooth access — click Allow.
6. Once connected the LED will turn **solid blue**. Read the on-screen instructions and click **Begin Task**
7. Press **← left arrow** if the center arrow points left, **→ right arrow** if it points right
8. Respond as quickly and accurately as possible. The incongruent trials (flanking arrows pointing opposite to the center) will naturally cause some errors, which is what generates the ErrP signal
9. When the task finishes the recording is saved and automatically loaded into the visualizer
 
#### What the App Does With the Data
 
- **During the task**: EEG is streamed continuously at 200 Hz into a memory buffer. Event markers are stamped into the data stream at the exact sample when each stimulus appears and each response is made.
- **After the task**: the full buffer is saved to a `.csv` file, then automatically converted to `.set` format.
- **Conversion pipeline**: the raw signal is epoched into −200ms to +800ms windows around each stimulus onset.
- **Visualizing**: the resulting file shows the **average across all trials**.
 
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

---

### Running Tests

To run the **unit tests**:

```bash
uv run pytest -q
```
To run a specific unit test:

```bash
uv run pytest tests/<test_file>.py -v
```
