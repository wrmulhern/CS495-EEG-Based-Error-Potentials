# Testing Documentation for ErrP Visualizer

## Overview

This document outlines the testing strategy for the ErrP Visualizer application, a Python-based EEG data analysis tool for error-related potentials. The testing suite covers all major components and ensures reliability, correctness, and maintainability of the codebase.

## Components/Services to Test

### 1. Configuration Module (`src/config.py`)
- **Purpose**: Centralized configuration for all domain constants and tunable parameters
- **Components**:
  - `GANGLION`: OpenBCI Ganglion board parameters
  - `MARKERS`: Flanker-task event marker codes
  - `EPOCH`: Epoch extraction parameters
  - `ERP`: ERP component time windows
  - `FLANKER`: Flanker task timing configurations
  - `VALIDATION`: File and EEG data validation thresholds
  - `EXPORT`: Export/save defaults
  - `RECORDER`: CSV output settings
  - `PLOT`: Visualization defaults

### 2. Data Processing Module (`src/data_processing/`)
- **Purpose**: Core scientific operations for EEG data analysis
- **Components**:
  - `data_loader.py`: Data loading and epoching functionality
  - `data_processor.py`: Averaging, channel selection, time windowing
  - `csv_converter.py`: CSV to EEGLAB .set file conversion
  - `file_validator.py`: File validation utilities

### 3. Data Visualization Module (`src/data_visualization/`)
- **Purpose**: Matplotlib-based plotting and visualization
- **Components**:
  - `visualizer.py`: Plot generation and theming

### 4. GUI Module (`src/gui/`)
- **Purpose**: PyQt5-based user interface
- **Components**:
  - `file_window.py`: Main file selection interface
  - `flanker_window.py`: EEG recording interface
  - `help_dialog.py`: Help documentation dialog
  - `themes/`: Color schemes and theming
  - `utils/`: Custom UI components (checkbox, drag-and-drop, multi-select)

### 5. EEG Recorder Module (`src/eeg_recorder.py`)
- **Purpose**: Real-time EEG data acquisition using BrainFlow
- **Components**: Serial port management and data streaming

### 6. Main Application (`src/main.py`)
- **Purpose**: Application entry point and CLI argument parsing

## Significant Test Case Descriptions

### Configuration Tests (`tests/test_config.py`)
- **test_ganglion_config**: Verifies Ganglion board parameters (sfreq, n_channels, column indices)
- **test_markers_config**: Validates event marker codes for stimulus/response events
- **test_epoch_config**: Checks epoch extraction parameters (timing, event counts)
- **test_erp_config**: Confirms ERP component time windows (ERN, Pe)
- **test_flanker_config**: Tests flanker task timing configurations
- **test_validation_config**: Validates file size limits and allowed extensions
- **test_export_config**: Checks export settings (DPI, formats)
- **test_recorder_config**: Verifies CSV output formatting
- **test_plot_config**: Confirms plotting defaults (figure sizes, styling)

### Data Processing Tests
- **test_bunch** (`tests/test_data_loader.py`): Tests MNE Epochs object creation
- **test_epochs_data_init**: Validates custom EpochsData initialization
- **test_evoked_data_init** (`tests/test_data_processor.py`): Tests EvokedData creation
- **test_average_epochs**: Verifies epoch averaging (noise cancellation)
- **test_select_channels**: Tests channel selection and metadata preservation
- **test_select_time_window**: Validates time-based data cropping
- **test_convert_ganglion_csv_to_set** (`tests/test_csv_converter.py`): Tests CSV to .set conversion

### File Validation Tests (`tests/test_file_validator.py`)
- **test_validate_file_path_valid/invalid**: Path existence and type checking
- **test_validate_file_extension_valid/invalid**: Extension validation against allowed types

### EEG Recorder Tests (`tests/test_eeg_recorder.py`)
- **test_eeg_recorder_init_valid/invalid_port**: Recorder initialization with valid/invalid ports
- **test_list_ports**: Serial port enumeration

### GUI Tests (`tests/test_gui.py`)
- **test_file_window**: File selection window instantiation and properties
- **test_flanker_window**: EEG recording interface creation
- **test_help_dialog**: Help documentation dialog functionality
- **test_themes**: Theme application to QApplication
- **test_colors**: Color palette retrieval and attributes
- **test_utils_checkbox**: Custom checkbox widget
- **test_utils_drag_and_drop**: Drag-and-drop functionality
- **test_utils_multi_select**: Multi-select dropdown component

### Main Application Tests (`tests/test_main.py`)
- **test_setup_logging**: Logging configuration verification
- **test_main**: QApplication creation and window display

### Visualization Tests (`tests/test_visualizer.py`)
- **test_apply_mpl_theme**: Matplotlib theme application for light/dark modes

## Test Scripts

### Primary Test Runner (`tests/run_tests.py`)
The main test execution script that runs all tests programmatically:

```python
#!/usr/bin/env python3

import sys
import os

# Change to project root
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

# Add src and tests to path
sys.path.insert(0, 'src')
sys.path.insert(0, 'tests')

def run_test(module_name, test_func):
    try:
        # Set path for src
        import sys
        if 'src' not in sys.path:
            sys.path.insert(0, 'src')
        # Import the test module
        exec(f"import {module_name}")
        # Call the function
        exec(f"{module_name}.{test_func}()")
        print(f"✓ {module_name}.{test_func} passed")
        return True
    except Exception as e:
        print(f"✗ {module_name}.{test_func} failed: {e}")
        return False

def main():
    tests = [
        # Configuration tests
        ('test_config', 'test_ganglion_config'),
        ('test_config', 'test_markers_config'),
        ('test_config', 'test_epoch_config'),
        ('test_config', 'test_erp_config'),
        ('test_config', 'test_flanker_config'),
        ('test_config', 'test_validation_config'),
        ('test_config', 'test_export_config'),
        ('test_config', 'test_recorder_config'),
        ('test_config', 'test_plot_config'),
        # Data processing tests
        ('test_data_loader', 'test_bunch'),
        ('test_data_loader', 'test_epochs_data_init'),
        ('test_data_processor', 'test_evoked_data_init'),
        ('test_data_processor', 'test_average_epochs'),
        ('test_data_processor', 'test_select_channels'),
        ('test_data_processor', 'test_select_time_window'),
        # File handling tests
        ('test_eeg_recorder', 'test_eeg_recorder_init_valid'),
        ('test_eeg_recorder', 'test_eeg_recorder_init_invalid_port'),
        ('test_eeg_recorder', 'test_list_ports'),
        ('test_file_validator', 'test_validate_file_path_valid'),
        ('test_file_validator', 'test_validate_file_path_invalid'),
        ('test_file_validator', 'test_validate_file_extension_valid'),
        ('test_file_validator', 'test_validate_file_extension_invalid'),
        ('test_csv_converter', 'test_convert_ganglion_csv_to_set'),
        # Visualization tests
        ('test_visualizer', 'test_apply_mpl_theme'),
        # Main application tests
        ('test_main', 'test_setup_logging'),
        ('test_main', 'test_main'),
        # GUI tests
        ('test_gui', 'test_file_window'),
        ('test_gui', 'test_flanker_window'),
        ('test_gui', 'test_help_dialog'),
        ('test_gui', 'test_themes'),
        ('test_gui', 'test_colors'),
        ('test_gui', 'test_utils_checkbox'),
        ('test_gui', 'test_utils_drag_and_drop'),
        ('test_gui', 'test_utils_multi_select'),
    ]

    passed = 0
    total = len(tests)
    for module, func in tests:
        if run_test(module, func):
            passed += 1

    print(f"\nPassed: {passed}/{total}")
    if passed == total:
        print("All tests passed!")
    else:
        print("Some tests failed.")

if __name__ == '__main__':
    main()
```

### Pytest Integration
All tests are also compatible with pytest for modern testing workflows:

```bash
# Run all tests
uv run pytest tests/

# Run with verbose output
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_config.py

# Run specific test function
uv run pytest tests/test_config.py::test_ganglion_config

# Generate coverage report
uv run pytest tests/ --cov=src --cov-report=html
```

## Running Tests

### Prerequisites
1. Install dependencies: `uv sync`
2. Install test dependencies: `uv sync --dev`

### Execution Methods

#### Method 1: Custom Test Runner (Recommended for CI)
```bash
python tests/run_tests.py
```

#### Method 2: Pytest (Recommended for development)
```bash
uv run pytest tests/
```

#### Method 3: Individual Test Files
```bash
python tests/test_config.py
python tests/test_gui.py
# etc.
```

### Test Environment Setup
- Tests requiring PyQt5 automatically skip if not available
- Matplotlib tests use non-GUI backend ('Agg') to avoid display issues
- Mock objects are used for external dependencies (serial ports, file I/O)

## Test Coverage

Current test coverage includes:
- **Configuration validation**: All config constants and defaults
- **Data processing pipeline**: Loading, epoching, averaging, filtering
- **File I/O operations**: CSV reading, .set file creation, validation
- **GUI components**: Window creation, theming, custom widgets
- **EEG acquisition**: Port detection, recorder initialization
- **Visualization**: Plot generation and styling
- **Application lifecycle**: Startup, logging, argument parsing

### Coverage Goals
- Maintain >90% code coverage across all modules
- Test both success and failure paths
- Include integration tests for critical workflows
- Validate GUI behavior without requiring display

### Continuous Integration
Tests should be run on:
- All commits to main branch
- Pull requests
- Release builds
- After dependency updates

## Test Maintenance

### Adding New Tests
1. Create test file in `tests/` directory following naming convention `test_<module>.py`
2. Import required modules and test dependencies
3. Write test functions starting with `test_`
4. Add test to `tests/run_tests.py` if using custom runner
5. Update this documentation

### Test Data
- Use synthetic/mock data for deterministic testing
- Avoid real EEG files in repository
- Generate test data programmatically when possible

### Best Practices
- Tests should be fast (<1 second per test)
- Use descriptive test names and docstrings
- Mock external dependencies (file I/O, network, hardware)
- Test edge cases and error conditions
- Keep tests independent and isolated