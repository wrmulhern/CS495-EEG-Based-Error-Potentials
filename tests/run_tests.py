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
        ('test_config', 'test_ganglion_config'),
        ('test_config', 'test_markers_config'),
        ('test_config', 'test_epoch_config'),
        ('test_config', 'test_erp_config'),
        ('test_config', 'test_flanker_config'),
        ('test_config', 'test_validation_config'),
        ('test_config', 'test_export_config'),
        ('test_config', 'test_recorder_config'),
        ('test_config', 'test_plot_config'),
        ('test_data_loader', 'test_bunch'),
        ('test_data_loader', 'test_epochs_data_init'),
        ('test_data_processor', 'test_evoked_data_init'),
        ('test_data_processor', 'test_average_epochs'),
        ('test_data_processor', 'test_select_channels'),
        ('test_data_processor', 'test_select_time_window'),
        ('test_eeg_recorder', 'test_eeg_recorder_init_valid'),
        ('test_eeg_recorder', 'test_eeg_recorder_init_invalid_port'),
        ('test_eeg_recorder', 'test_list_ports'),
        ('test_file_validator', 'test_validate_file_path_valid'),
        ('test_file_validator', 'test_validate_file_path_invalid'),
        ('test_file_validator', 'test_validate_file_extension_valid'),
        ('test_file_validator', 'test_validate_file_extension_invalid'),
        ('test_csv_converter', 'test_convert_ganglion_csv_to_set'),
        ('test_visualizer', 'test_apply_mpl_theme'),
        ('test_main', 'test_setup_logging'),
        ('test_main', 'test_main'),
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
        print("All successful tests passed!")
    else:
        print("Some tests failed.")

if __name__ == '__main__':
    main()