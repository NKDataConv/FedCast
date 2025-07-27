"""
MIT-BIH Arrhythmia Database ECG Dataset Loader

This module provides a loader for the MIT-BIH Arrhythmia Database from PhysioNet:
https://physionet.org/content/mitdb/1.0.0/

The dataset contains ECG recordings from 47 different patients, with each patient representing
a natural partition for federated learning experiments. This is particularly relevant for
medical federated learning where patient privacy is crucial.

The ECG signals are sampled at 360 Hz and contain two leads (MLII and V1/V2/V4/V5).
We use the first lead (MLII) for time series forecasting tasks.

This loader allows dynamic, on-demand loading of ECG data for a single patient, returning
windowed time series samples suitable for forecasting tasks.
"""

import os
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datasets import Dataset
import wfdb

WINDOW_SIZE = 20
DATA_DIR = Path("data/mitdb")
BASE_URL = "https://physionet.org/files/mitdb/1.0.0/"

# MIT-BIH record names (patient IDs)
RECORD_NAMES = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234"
]


def download_record(record_name):
    """Download a single ECG record from PhysioNet."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Files needed for each record
    extensions = ['.dat', '.hea', '.atr']
    
    for ext in extensions:
        filename = f"{record_name}{ext}"
        file_path = DATA_DIR / filename
        
        if not file_path.exists():
            url = f"{BASE_URL}{filename}"
            print(f"Downloading {filename}...")
            try:
                response = requests.get(url)
                response.raise_for_status()
                with open(file_path, 'wb') as f:
                    f.write(response.content)
            except requests.RequestException as e:
                print(f"Failed to download {filename}: {e}")
                raise


def load_ecg_record(record_name):
    """Load ECG data for a specific record using wfdb."""
    try:
        # First try to read locally
        record_path = str(DATA_DIR / record_name)
        record = wfdb.rdrecord(record_path)
        return record.p_signal[:, 0]  # Use first channel (MLII)
    except:
        # If local read fails, download and try again
        download_record(record_name)
        record_path = str(DATA_DIR / record_name)
        record = wfdb.rdrecord(record_path)
        return record.p_signal[:, 0]  # Use first channel (MLII)


def get_patient_ids():
    """Get list of available patient IDs (record names)."""
    return RECORD_NAMES


def load_dataset(partition_id: int, num_examples: int = 500):
    """
    Loads ECG time series data for a single patient and prepares it for forecasting.
    
    Args:
        partition_id: The patient index (0-based) to use as the partition.
        num_examples: Number of (x, y) pairs to generate.
        
    Returns:
        A Hugging Face Dataset with 'x' (input sequences) and 'y' (target values).
    """
    patient_ids = get_patient_ids()
    if partition_id < 0 or partition_id >= len(patient_ids):
        raise ValueError(f"partition_id must be between 0 and {len(patient_ids)-1}")
    
    record_name = patient_ids[partition_id]
    
    # Load ECG signal for the patient
    ecg_signal = load_ecg_record(record_name)
    
    # Remove NaN values and normalize
    ecg_signal = ecg_signal[~np.isnan(ecg_signal)]
    
    # Simple normalization (z-score)
    ecg_signal = (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)
    
    # Check if we have enough data
    total_points = num_examples + WINDOW_SIZE
    if len(ecg_signal) < total_points:
        # If not enough data, repeat the signal
        repeats = (total_points // len(ecg_signal)) + 1
        ecg_signal = np.tile(ecg_signal, repeats)
    
    # Use only the required number of points
    values = ecg_signal[:total_points]
    
    # Create input/output sequences using numpy
    X = np.lib.stride_tricks.sliding_window_view(values, WINDOW_SIZE)[:num_examples]
    y = values[WINDOW_SIZE:WINDOW_SIZE + num_examples]
    
    # Create DataFrame and convert to Dataset
    df_xy = pd.DataFrame({"x": list(X), "y": y})
    dataset = Dataset.from_pandas(df_xy)
    return dataset


if __name__ == "__main__":
    # Test the dataset loader
    print("Available patients:", len(get_patient_ids()))
    dataset = load_dataset(0, num_examples=10)
    print("Dataset:", dataset)
    print("First sample:", dataset[0])
    print("Input shape:", np.array(dataset[0]["x"]).shape)
    print("Target type:", type(dataset[0]["y"])) 