"""
Intel Berkeley Research Lab IoT Sensor Dataset Loader

This module provides a loader for the Intel Berkeley Research Lab sensor dataset:
http://db.csail.mit.edu/labdata/labdata.html

The dataset contains sensor readings from 54 Mica2Dot sensors deployed in the Intel Berkeley 
Research lab between February 28th and April 5th, 2004. Each sensor collected timestamped 
temperature, humidity, light, and voltage values every 31 seconds.

Each sensor represents a natural partition for federated learning experiments, which is 
particularly relevant for IoT federated learning where sensor data privacy and locality 
are crucial considerations.

This loader focuses on temperature data for time series forecasting tasks, allowing 
dynamic, on-demand loading of sensor data for a single sensor location, returning 
windowed time series samples suitable for temperature prediction.
"""

import os
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from datasets import Dataset

# Constants
WINDOW_SIZE = 20  # Number of previous temperature readings to use for prediction
DATA_URL = "http://db.csail.mit.edu/labdata/data.txt.gz"
CACHE_DIR = Path.home() / ".cache" / "fedcast" / "intel_iot"

# List of 54 sensor IDs from the Intel Berkeley dataset
SENSOR_IDS = list(range(1, 55))  # Sensors 1-54

def get_sensor_ids() -> List[int]:
    """
    Get the list of available sensor IDs.
    
    Returns:
        List of sensor IDs (1-54)
    """
    return SENSOR_IDS.copy()

def _download_and_cache_data() -> Path:
    """
    Download the Intel Berkeley sensor data if not already cached.
    
    Returns:
        Path to the cached data file
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached_file = CACHE_DIR / "intel_sensors.csv"
    
    if cached_file.exists():
        return cached_file
    
    print(f"Downloading Intel Berkeley sensor data from {DATA_URL}...")
    
    try:
        # Download the gzipped data
        response = requests.get(DATA_URL, timeout=30)
        response.raise_for_status()
        
        # Save the gzipped data temporarily
        temp_gz_file = CACHE_DIR / "data.txt.gz"
        with open(temp_gz_file, 'wb') as f:
            f.write(response.content)
        
        # Extract and parse the data
        import gzip
        with gzip.open(temp_gz_file, 'rt') as f:
            lines = f.readlines()
        
        # Parse the data format: date time epoch moteid temperature humidity light voltage
        data_rows = []
        for line in lines:
            if line.strip() and not line.startswith('#'):
                parts = line.strip().split()
                if len(parts) >= 8:  # Ensure we have all required fields
                    try:
                        date = parts[0]
                        time = parts[1]
                        epoch = int(parts[2])
                        moteid = int(parts[3])
                        temperature = float(parts[4])
                        humidity = float(parts[5])
                        light = float(parts[6])
                        voltage = float(parts[7])
                        
                        # Only include valid sensor IDs and non-null temperature values
                        if 1 <= moteid <= 54 and not np.isnan(temperature):
                            data_rows.append({
                                'datetime': f"{date} {time}",
                                'epoch': epoch,
                                'sensor_id': moteid,
                                'temperature': temperature,
                                'humidity': humidity,
                                'light': light,
                                'voltage': voltage
                            })
                    except (ValueError, IndexError):
                        continue  # Skip malformed lines
        
        # Convert to DataFrame and save as CSV
        df = pd.DataFrame(data_rows)
        df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
        df = df.dropna(subset=['datetime', 'temperature'])
        df = df.sort_values(['sensor_id', 'datetime']).reset_index(drop=True)
        
        df.to_csv(cached_file, index=False)
        
        # Clean up temporary file
        temp_gz_file.unlink()
        
        print(f"Intel Berkeley sensor data downloaded and cached to {cached_file}")
        print(f"Total records: {len(df)}")
        print(f"Sensors with data: {sorted(df['sensor_id'].unique())}")
        
        return cached_file
        
    except Exception as e:
        print(f"Error downloading Intel Berkeley sensor data: {e}")
        raise

def _load_sensor_data(sensor_id: int) -> pd.DataFrame:
    """
    Load temperature data for a specific sensor.
    
    Args:
        sensor_id: The ID of the sensor (1-54)
        
    Returns:
        DataFrame with datetime and temperature columns for the sensor
    """
    if sensor_id not in SENSOR_IDS:
        raise ValueError(f"Invalid sensor_id {sensor_id}. Must be in range 1-54.")
    
    data_file = _download_and_cache_data()
    
    # Load the full dataset
    df = pd.read_csv(data_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Filter for the specific sensor
    sensor_df = df[df['sensor_id'] == sensor_id].copy()
    
    if len(sensor_df) == 0:
        raise ValueError(f"No data found for sensor {sensor_id}")
    
    # Sort by datetime and reset index
    sensor_df = sensor_df.sort_values('datetime').reset_index(drop=True)
    
    return sensor_df[['datetime', 'temperature']]

def _create_sliding_windows(temperatures: np.ndarray, window_size: int) -> List[Dict[str, Any]]:
    """
    Create sliding window samples from temperature time series.
    
    Args:
        temperatures: Array of temperature values
        window_size: Size of the input window
        
    Returns:
        List of dictionaries with 'x' (input window) and 'y' (target) keys
    """
    if len(temperatures) < window_size + 1:
        raise ValueError(f"Not enough data points. Need at least {window_size + 1}, got {len(temperatures)}")
    
    samples = []
    for i in range(len(temperatures) - window_size):
        x = temperatures[i:i + window_size].tolist()
        y = float(temperatures[i + window_size])
        samples.append({"x": x, "y": y})
    
    return samples

def load_dataset(partition_id: int, num_examples: int = 500) -> Dataset:
    """
    Load Intel Berkeley IoT sensor dataset for a specific sensor (partition).
    
    Args:
        partition_id: Sensor ID (0-53, mapped to sensor IDs 1-54)
        num_examples: Number of sliding window examples to generate
        
    Returns:
        Dataset object with 'x' (temperature windows) and 'y' (next temperature) columns
    """
    if not (0 <= partition_id < len(SENSOR_IDS)):
        raise ValueError(f"partition_id must be in range 0-{len(SENSOR_IDS)-1}, got {partition_id}")
    
    # Map partition_id to actual sensor_id (1-54)
    sensor_id = SENSOR_IDS[partition_id]
    
    # Load temperature data for this sensor
    sensor_df = _load_sensor_data(sensor_id)
    temperatures = sensor_df['temperature'].values
    
    # Normalize temperatures (z-score normalization per sensor)
    temp_mean = np.mean(temperatures)
    temp_std = np.std(temperatures)
    if temp_std > 0:
        temperatures = (temperatures - temp_mean) / temp_std
    
    # Create sliding windows
    samples = _create_sliding_windows(temperatures, WINDOW_SIZE)
    
    # Limit to requested number of examples
    if len(samples) > num_examples:
        # Take evenly spaced samples to cover the full time range
        indices = np.linspace(0, len(samples) - 1, num_examples, dtype=int)
        samples = [samples[i] for i in indices]
    
    return Dataset.from_list(samples) 