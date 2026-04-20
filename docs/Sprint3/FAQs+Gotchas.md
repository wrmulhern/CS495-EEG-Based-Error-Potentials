# FAQs/Gotchas Documentation – EEG Based Error Potentials

---

## Frequently Asked Questions

### Q1: How does the ErrP Visualizer work?

The ErrP Visualizer is a desktop application for analyzing Error-Related Potential (ErrP) brain signals from EEG data. Here's the workflow:

1. **Load Data:** The application has the ability to take in EEG data with the **Record EEG** button and the ability to select prerecorded EEG data. The Record EEG button takes you to a window where, using an OpenBCI Ganglion device, you can complete a flanker task — a standard cognitive paradigm that elicits ErrP signals by flashing arrows and asking you to type the correct one. To use prerecorded EEG data, simply drag-and-drop or browse for a `.set` or `.csv` file.
2. **Auto-Processing:** CSV files are automatically converted to the standard EEGLAB format (`.set`) with proper channel locations and sampling rates.
3. **Configure Visualization:** Choose your graph type (time series, topographic map, or joint maps), select time windows, and pick specific channels to analyze.
4. **Visualize:** Click the **Visualize** button (or press Enter) to generate scientific plots showing brain activity patterns.
5. **Analyze:** Look for ErrP signatures after error responses.

---

### Q2: What do the different graph types show?

- **ErrP Time Series:** Shows amplitude (microvolts) over time for selected channels. Use this to see the waveform shape and identify ERP components. The "Display Events and Responses" checkbox highlights the ERN/Ne (50–150ms, blue) and Pe (200–400ms, green) time windows with hover annotations.

- **Topographic Map:** Shows spatial distribution of brain activity at specific time points (e.g., 100ms, 200ms, 300ms). Warmer colors = more positive voltage; cooler colors = more negative. Requires ≥19 channels.

- **Joint Maps:** Combines both views — time series on top and topographic maps on bottom at selected times. Best for comprehensive ERP analysis showing both temporal dynamics and spatial patterns.

---

### Q3: Why isn't my EEG data showing any clear ErrP signals?

There are several common reasons why ErrP signals may not appear clearly:

- **Too few trials:** ErrPs are small signals that require averaging across many trials (typically 80–100+) to become visible.
- **Low signal quality:** Poor electrode contact or placement (especially on the forehead) can introduce noise and obscure the signal. Make sure electrodes are secure and properly positioned.
- **Incorrect task performance:** ErrPs are only generated when errors occur. If you made very few mistakes during the Flanker Task, there may not be enough error trials to produce a clear signal.
- **Incorrect channel selection:** ErrPs are strongest in frontal regions (e.g., center forehead). Make sure you're visualizing the relevant channels.
- **Timing expectations:** The ERN appears around 50–150 ms and the Pe around 200–400 ms after the response — check that you are looking in the correct time window.

---

## Gotchas

### Installation

**Issue:** `ModuleNotFoundError: No module named 'src'`  
**Cause:** Running the app from the wrong directory or project structure changed  
**Fix:**
```bash
cd path/to/errp-visualizer
uv run -m src.main
```

---

**Issue:** `uv: command not found`  
**Cause:** UV package manager not installed  
**Fix:**
```bash
pip install uv
```

---

**Issue:** PyQt5 fails to install on macOS  
**Cause:** Missing system dependencies  
**Fix:**
```bash
xcode-select --install
uv sync
```

---

**Issue:** Qt platform plugin could not be initialized  
**Cause:** Missing Qt platform libraries  
**Fix:**
```bash
uv pip uninstall PyQt5
uv pip install PyQt5
```

---

### External Resources

**Issue:** UV package manager conflicts with existing pip installations  
**Cause:** Mixing UV and pip in the same environment  
**Fix:** Use UV exclusively — do not mix `pip install` with `uv add`