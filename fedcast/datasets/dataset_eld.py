"""
UCI Electricity Load Diagrams (ELD) Dataset Loader

This module provides a loader for the UCI Electricity Load Diagrams 2011-2014 dataset:
https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014

The dataset contains hourly electricity consumption data for 370 customers (clients) over several years.
Each client is represented by a column, making it naturally suited for federated learning experiments.

This loader allows dynamic, on-demand loading of data for a single client, returning windowed time series
samples suitable for forecasting tasks. The dataset is downloaded and cached locally if not already present.
"""

import os
import requests
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from datasets import Dataset

WINDOW_SIZE = 20
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip"
DATA_DIR = Path("data")
DATA_FILE = DATA_DIR / "LD2011_2014.txt"


def download_eld_dataset():
    DATA_DIR.mkdir(exist_ok=True)
    zip_path = DATA_DIR / "LD2011_2014.txt.zip"
    if not DATA_FILE.exists():
        print("Downloading ELD dataset...")
        r = requests.get(DATA_URL)
        with open(zip_path, "wb") as f:
            f.write(r.content)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(DATA_DIR)
        os.remove(zip_path)


def get_client_ids():
    download_eld_dataset()
    # Read only the header to get client IDs
    with open(DATA_FILE, "r") as f:
        header = f.readline().strip().split(";")
    # First column is '""', rest are client IDs (quoted), so strip quotes
    return [h.strip('"') for h in header[1:]]


def load_dataset(partition_id: int, num_examples: int = 500):
    """
    Loads time series data for a single client from the ELD dataset and prepares it for forecasting.
    Args:
        partition_id: The client index (0-based) to use as the partition.
        num_examples: Number of (x, y) pairs to generate.
    Returns:
        A Hugging Face Dataset with 'x' (input sequences) and 'y' (target values).
    """
    client_ids = get_client_ids()
    if partition_id < 0 or partition_id >= len(client_ids):
        raise ValueError(f"partition_id must be between 0 and {len(client_ids)-1}")
    client_id = client_ids[partition_id]

    # Read only the relevant column for the client
    # The first column is an empty string (""), so use that for the index
    df = pd.read_csv(
        DATA_FILE,
        sep=";",
        usecols=[client_id],
        decimal=","
    )
    # Convert to float and drop missing values
    series = pd.to_numeric(df[client_id], errors="coerce").dropna()
    # Use only the first num_examples + WINDOW_SIZE points
    total_points = num_examples + WINDOW_SIZE
    if len(series) < total_points:
        raise ValueError(f"Not enough data points for client {client_id}")
    values = series.values[:total_points]

    # Efficiently create input/output sequences using numpy
    X = np.lib.stride_tricks.sliding_window_view(values, WINDOW_SIZE)[:num_examples]
    y = values[WINDOW_SIZE:WINDOW_SIZE + num_examples]
    df_xy = pd.DataFrame({"x": list(X), "y": y})
    dataset = Dataset.from_pandas(df_xy)
    return dataset


if __name__ == "__main__":
    dataset = load_dataset(0)
    print(dataset)
    print(dataset[0]) 