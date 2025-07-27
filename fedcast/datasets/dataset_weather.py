"""
210 U.S. Cities Historical Weather Dataset Loader

This module provides a loader for the 210 U.S. Cities historical weather dataset:
https://kilthub.cmu.edu/articles/dataset/Compiled_daily_temperature_and_precipitation_data_for_the_U_S_cities/7890488

The dataset contains historical daily temperature and precipitation data from 210 weather
stations across the United States, with records dating back to at least 1900 for most cities.
Data is sourced from NOAA's Global Historical Climatology Network (GHCN-D).

Each weather station represents a natural partition for federated learning experiments, 
which is particularly relevant for meteorological federated learning where weather data 
privacy and locality are crucial considerations.

This loader focuses on daily average temperature for time series forecasting tasks, 
allowing dynamic loading of weather data for a specific weather station, returning 
windowed time series samples suitable for temperature prediction.
"""

import os
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from datasets import Dataset
import warnings
warnings.filterwarnings('ignore')

# Constants
WINDOW_SIZE = 20  # Number of previous temperature readings to use for prediction
BASE_URL = "https://media.githubusercontent.com/media/radames/dataset-historical-daily-temperature-210-US/main/"
CACHE_DIR = Path.home() / ".cache" / "fedcast" / "weather"

# Weather station IDs from the 210 U.S. cities dataset
# Using a representative sample of 50 stations for manageable federated learning experiments
WEATHER_STATIONS = [
    'USC00042863', 'USC00166584', 'USC00280734', 'USC00286055', 'USC00356749', 'USC00380072',
    'USW00003017', 'USW00003103', 'USW00003145', 'USW00003171', 'USW00003812', 'USW00003813',
    'USW00003820', 'USW00003822', 'USW00003856', 'USW00003859', 'USW00003860', 'USW00003872',
    'USW00003904', 'USW00003927', 'USW00003937', 'USW00003940', 'USW00003945', 'USW00003947',
    'USW00003953', 'USW00003963', 'USW00003965', 'USW00004853', 'USW00012815', 'USW00012816',
    'USW00012835', 'USW00012836', 'USW00012839', 'USW00012842', 'USW00012844', 'USW00012921',
    'USW00012923', 'USW00012924', 'USW00012960', 'USW00013722', 'USW00013733', 'USW00013736',
    'USW00013737', 'USW00013739', 'USW00013740', 'USW00013743', 'USW00013781', 'USW00013833',
    'USW00013838', 'USW00013865'
]

def get_weather_stations() -> List[str]:
    """
    Get the list of available weather station IDs.
    
    Returns:
        List of weather station IDs
    """
    return WEATHER_STATIONS.copy()

def _download_and_cache_station_data(station_id: str) -> Path:
    """
    Download weather data for a specific station if not already cached.
    
    Args:
        station_id: The weather station ID
        
    Returns:
        Path to the cached data file
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached_file = CACHE_DIR / f"{station_id}.csv"
    
    if cached_file.exists():
        return cached_file
    
    print(f"Downloading weather data for station {station_id}...")
    
    try:
        # Download the CSV file
        file_url = BASE_URL + f"{station_id}.csv"
        response = requests.get(file_url, timeout=30)
        response.raise_for_status()
        
        # Save the data
        with open(cached_file, 'wb') as f:
            f.write(response.content)
        
        # Verify the data is valid
        df = pd.read_csv(cached_file)
        if len(df) == 0:
            raise ValueError(f"No data found in {station_id}.csv")
        
        print(f"Weather data for station {station_id} downloaded and cached")
        print(f"Records: {len(df)}, Date range: {df['Date'].min()} to {df['Date'].max()}")
        
        return cached_file
        
    except Exception as e:
        print(f"Error downloading weather data for station {station_id}: {e}")
        raise

def _load_station_data(station_id: str) -> pd.DataFrame:
    """
    Load weather data for a specific station.
    
    Args:
        station_id: The weather station ID
        
    Returns:
        DataFrame with date and temperature columns for the station
    """
    if station_id not in WEATHER_STATIONS:
        raise ValueError(f"Invalid station_id {station_id}. Must be one of the available stations.")
    
    data_file = _download_and_cache_station_data(station_id)
    
    # Load the data
    df = pd.read_csv(data_file)
    
    # Convert date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Calculate average temperature from max and min
    # Handle missing values (NA) by converting to NaN
    df['tmax'] = pd.to_numeric(df['tmax'], errors='coerce')
    df['tmin'] = pd.to_numeric(df['tmin'], errors='coerce')
    
    # Calculate average temperature (only where both tmax and tmin are available)
    df['tavg'] = (df['tmax'] + df['tmin']) / 2.0
    
    # Remove rows with missing temperature data
    df = df.dropna(subset=['tavg'])
    
    if len(df) == 0:
        raise ValueError(f"No valid temperature data found for station {station_id}")
    
    # Sort by date and reset index
    df = df.sort_values('Date').reset_index(drop=True)
    
    return df[['Date', 'tavg']]

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
    Load weather dataset for a specific weather station (partition).
    
    Args:
        partition_id: Station ID (0-49, mapped to weather station IDs)
        num_examples: Number of sliding window examples to generate
        
    Returns:
        Dataset object with 'x' (temperature windows) and 'y' (next temperature) columns
    """
    if not (0 <= partition_id < len(WEATHER_STATIONS)):
        raise ValueError(f"partition_id must be in range 0-{len(WEATHER_STATIONS)-1}, got {partition_id}")
    
    # Map partition_id to actual station_id
    station_id = WEATHER_STATIONS[partition_id]
    
    # Load temperature data for this station
    station_df = _load_station_data(station_id)
    temperatures = station_df['tavg'].values
    
    # Convert Fahrenheit to Celsius for better international compatibility
    temperatures = (temperatures - 32) * 5/9
    
    # Normalize temperatures (z-score normalization per station)
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