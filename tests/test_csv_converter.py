"""Tests for csv_converter module."""

import tempfile
import os
import pandas as pd
from src.data_processing.csv_converter import convert_ganglion_csv_to_set


def test_convert_ganglion_csv_to_set():
    """Test CSV to SET conversion."""
    # Create a mock CSV
    data = {i: [0.0] * 10 for i in range(15)}
    df = pd.DataFrame(data)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f, index=False, header=False)
        csv_path = f.name
    
    try:
        set_path = convert_ganglion_csv_to_set(csv_path)
        assert os.path.exists(set_path)
        assert set_path.endswith('_converted.set')
    finally:
        os.unlink(csv_path)
        if os.path.exists(set_path):
            os.unlink(set_path)