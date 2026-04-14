# Development Guide

This section is intended for developers who want to modify, extend, or contribute to the ErrP Visualizer codebase.

---

## Language and Compiler

The project is written entirely in **Python 3.13**. Python is an interpreted language and does not require a compiler. The application is run directly through the Python interpreter via uv (see setup instructions in the README). If creating a standalone executable, PyInstaller compiles the application into a binary — see the section of the README.

---

## Build Management

The project uses **[uv](https://docs.astral.sh/uv/)** as the Python package and environment manager. uv replaces the traditional pip + virtualenv workflow with a faster, more deterministic alternative.

**Key uv commands:**

```bash
uv sync              # install all dependencies into the virtual environment
uv add <package>     # add a new dependency
uv remove <package>  # remove a dependency
uv run <command>     # run a command inside the virtual environment
```

There is no Makefile or other build system. All build and run commands are issued through uv directly.

---

## Dependencies

All dependencies are declared in **`pyproject.toml`** in the project root. The lockfile **`uv.lock`** pins exact versions for reproducible installs across machines. Both files are committed to the repository — do not edit them manually. Use `uv add` and `uv remove` to manage dependencies, which will update both files automatically.

**Core dependencies:**

| Package | Purpose |
|---|---|
| `PyQt5` | GUI framework — all windows, widgets, dialogs |
| `matplotlib` | EEG waveform and topomap plotting |
| `numpy` | Array operations, signal math |
| `scipy` | Bandpass/notch filtering, `.mat`/`.set` file I/O |
| `pandas` | CSV reading during Ganglion data conversion |
| `brainflow` | OpenBCI Ganglion EEG streaming and marker injection |
| `pyserial` | Serial port detection for the Ganglion dongle |
| `pyinstaller` | Building standalone executables |
| `markdown` | Rendering the README in the Help dialog |

---

## Project Structure

```
CS495-EEG-Based-Error-Potentials/
├── src/
│   ├── main.py                          # entry point
│   ├── gui/
│   │   ├── file_window.py               # main window, tab management, CSV conversion
│   │   ├── flanker_window.py            # Record EEG dialog and Flanker task UI
│   │   ├── themes/
│   │   │   ├── light_theme.py
│   │   │   └── dark_theme.py
│   │   └── utils/
│   │       ├── drag_and_drop.py         # file drop zone widget
│   │       └── checkbox.py              # custom toggle switch widget
│   ├── data_processing/
│   │   ├── data_loader.py               # reads EEGLAB .set files → EpochsData
│   │   ├── data_processor.py            # averaging, time windowing → EvokedData
│   │   └── eeg_recorder.py              # Brainflow streaming thread
│   └── data_visualization/
│       └── visualizer.py                # matplotlib plot functions (stateless)
├── pyproject.toml                        # dependencies
├── uv.lock                               # locked dependency versions
├── ErrPVisualizer.spec                   # PyInstaller build configuration
└── README.md
```

---

## Code Style

The project follows standard Python conventions. Contributions should adhere to the following:

- **[PEP 8](https://peps.python.org/pep-0008/)** style guidelines — 4-space indentation, snake_case for functions and variables, PascalCase for classes
- **Type hints** on function signatures where practical
- **Docstrings** on all public classes and methods — describe what the function does, its parameters, and its return value
- **Logging** via the standard `logging` module — use `logger.info()` for significant events, `logger.debug()` for verbose detail, `logger.warning()` for recoverable issues. Do not use `print()` for application output.
- **No unused imports** — keep imports clean and grouped (standard library, third-party, local)
- GUI code lives in `src/gui/`, data logic lives in `src/data_processing/`, plotting logic lives in `src/data_visualization/`. Do not mix concerns across these boundaries — the visualizer functions are stateless by design and should remain so.

---

## Making Changes

### Adding a new graph type

1. Add a new plot function to `src/data_visualization/visualizer.py` — it should accept an `EvokedData` object and a `matplotlib.figure.Figure` and return nothing (draw onto the figure directly)
2. Add the new type name to the `graph_type_combo` list in `FileTab._build_options_panel()` in `file_window.py`
3. Handle the new type in the `visualize()` method of `FileTab`

### Adding a new file format

1. Add a reader function in `src/data_processing/data_loader.py` that returns an `EpochsData` object
2. Handle the new file extension in `FileWindow.add_files()` in `file_window.py`

### Modifying the Flanker task

The task logic lives entirely in `src/gui/flanker_window.py`. Timing constants (`FIXATION_MS`, `STIMULUS_MS`, `RESPONSE_MS`, etc.) are defined at the top of the file and can be adjusted without touching the task state machine. Stimulus definitions (`STIMULI` list) can be extended to add new trial types. Event marker values are defined in `src/data_processing/eeg_recorder.py` and must stay in sync with the converter in `file_window.py`.

---

## Backlog and Bug Tracking

Sprint backlogs and known issues are tracked on the **[project site](https://aliburkemper12.github.io/Capstone-Project-Site/#/)** under the backlog section for each sprint. This is the authoritative source for planned work and open issues.

---

## Testing

There is currently no automated test suite. Testing is performed manually by running the application and exercising features directly. 

**Key manual test scenarios:**

- Load a `.set` file and verify all three graph types render correctly
- Load a `.csv` file and verify it converts and visualizes without error
- Run the Flanker task with the Ganglion disconnected and verify the error message is shown gracefully
- Run the Flanker task end to end with the Ganglion connected and verify the output file loads in the visualizer
- Toggle dark mode and verify all UI elements and plots update correctly
- Open the Help dialog and verify the README loads from GitHub

Automated testing is a known gap. Future contributors should consider adding unit tests for the data processing layer (`data_loader.py`, `data_processor.py`) using **pytest**, which is the standard Python testing framework and integrates well with uv:

```bash
uv add --dev pytest
uv run pytest
```

---

## Known Issues

### ⚠ EEG Signal Quality — High Priority

The current EEG recording pipeline produces **raw, unprocessed data** that is typically too noisy to clearly resolve ErrP components (ERN and Pe). The following improvements are needed before reliable ErrP detection is achievable:

- **Artifact rejection is rudimentary** — the current 100 µV peak-to-peak threshold is a blunt instrument. Independent Component Analysis (ICA) would allow systematic removal of eye blink and muscle artifacts while preserving brain signal.
- **No re-referencing** — the signal is referenced to a single earlobe electrode. Average reference or linked mastoid reference would improve signal quality.
- **No trial-type separation** — all trials (correct and error) are currently averaged together, which dilutes the ErrP. The pipeline should be extended to epoch error trials and correct trials separately so they can be compared. The ErrP is only present in error trials.
- **Low channel count** — with 3–4 forehead electrodes the spatial resolution is very limited. A full 8–16 channel setup with proper scalp placement would yield substantially cleaner data.
