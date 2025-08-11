"""
UNSW-NB15 Network Traffic Dataset Loader

This module provides a loader for the UNSW-NB15 network traffic dataset:
https://research.unsw.edu.au/projects/unsw-nb15-dataset

The dataset contains network flow records captured from a testbed with nine types of 
attacks (Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, 
Shellcode, and Worms) plus normal traffic. The dataset has 2.5 million records 
with 49 features each.

Each attack type (plus normal traffic) represents a natural partition for federated 
learning experiments, which is particularly relevant for cybersecurity federated 
learning where network traffic data privacy is crucial.

This loader focuses on network flow volume (bytes) over time for traffic forecasting 
tasks, allowing dynamic loading of traffic data for a specific attack category, 
returning windowed time series samples suitable for network traffic prediction.
"""

import os
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from datasets import Dataset
import zipfile
import warnings
warnings.filterwarnings('ignore')

# Constants
WINDOW_SIZE = 20  # Number of previous traffic measurements to use for prediction
BASE_URL = "https://github.com/iammyr/encrypted-network-datasets/raw/master/"
CACHE_DIR = Path.home() / ".cache" / "fedcast" / "network_traffic"

# Network traffic categories (attack types + normal)
# Each category represents a different network location/organization
TRAFFIC_CATEGORIES = [
    'Normal',
    'Fuzzers', 
    'Analysis',
    'Backdoors',
    'DoS',
    'Exploits', 
    'Generic',
    'Reconnaissance',
    'Shellcode',
    'Worms'
]


def _normalize_attack_categories(series: pd.Series) -> pd.Series:
    """Normalize attack category labels from UNSW-NB15 to canonical names.

    - Strips whitespace
    - Maps known variants/sentinels (e.g., '0' -> 'Normal', 'Backdoor' -> 'Backdoors')
    - Replaces empty and NaN-like strings with 'Normal'
    """
    s = series.astype(str).str.strip()
    mapping = {
        "": "Normal",
        "nan": "Normal",
        "NaN": "Normal",
        "0": "Normal",
        "Backdoor": "Backdoors",
    }
    s = s.replace(mapping)
    return s

def get_traffic_categories() -> List[str]:
    """
    Get the list of available network traffic categories.
    
    Returns:
        List of traffic categories (attack types + normal)
    """
    return TRAFFIC_CATEGORIES.copy()

def _download_and_cache_data() -> Path:
    """
    Download the UNSW-NB15 network traffic data if not already cached.
    
    Returns:
        Path to the cached data file
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached_file = CACHE_DIR / "unsw_nb15_combined.csv"
    
    if cached_file.exists():
        return cached_file
    
    print(f"Downloading UNSW-NB15 network traffic data...")
    
    try:
        # Download and combine all 4 CSV files
        all_data = []
        
        for i in range(1, 5):
            filename = f"UNSW-NB15_{i}.csv.zip"
            file_url = BASE_URL + filename
            
            print(f"Downloading {filename}...")
            response = requests.get(file_url, timeout=60)
            response.raise_for_status()
            
            # Save the zip file temporarily
            temp_zip_file = CACHE_DIR / filename
            with open(temp_zip_file, 'wb') as f:
                f.write(response.content)
            
            # Extract and read the CSV
            with zipfile.ZipFile(temp_zip_file, 'r') as zip_ref:
                csv_filename = f"UNSW-NB15_{i}.csv"
                zip_ref.extractall(CACHE_DIR)
                
                # Read the CSV file
                csv_path = CACHE_DIR / csv_filename
                df_chunk = pd.read_csv(csv_path, header=None, low_memory=False)
                all_data.append(df_chunk)
                
                # Clean up temporary files
                csv_path.unlink()
            temp_zip_file.unlink()
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Define column names based on UNSW-NB15 feature descriptions
        columns = [
            'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes',
            'sttl', 'dttl', 'sloss', 'dloss', 'service', 'sload', 'dload', 'spkts', 'dpkts',
            'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth',
            'res_bdy_len', 'sjit', 'djit', 'stime', 'ltime', 'sintpkt', 'dintpkt',
            'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd',
            'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm',
            'ct_src_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm',
            'attack_cat', 'label'
        ]
        
        combined_df.columns = columns[:len(combined_df.columns)]
        
        # Clean and prepare the data
        # Convert relevant columns to numeric
        numeric_cols = ['dur', 'sbytes', 'dbytes', 'sload', 'dload', 'spkts', 'dpkts']
        for col in numeric_cols:
            if col in combined_df.columns:
                combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
        
        # Fill NaN values
        combined_df = combined_df.fillna(0)
        
        # Ensure we have attack categories
        if 'attack_cat' in combined_df.columns:
            combined_df['attack_cat'] = _normalize_attack_categories(combined_df['attack_cat']).fillna('Normal')
        else:
            # If no attack_cat column, create one based on label
            combined_df['attack_cat'] = combined_df.get('label', 0).apply(
                lambda x: 'Normal' if x == 0 else 'Attack'
            )
        
        # Create a time proxy by using row index (simulating temporal order)
        combined_df['time_proxy'] = range(len(combined_df))
        
        # Save the combined dataset
        combined_df.to_csv(cached_file, index=False)
        
        print(f"UNSW-NB15 network traffic data downloaded and cached to {cached_file}")
        print(f"Total records: {len(combined_df)}")
        # Convert attack categories to string for sorting
        unique_categories = [str(cat) for cat in combined_df['attack_cat'].unique()]
        print(f"Attack categories: {sorted(unique_categories)}")
        
        return cached_file
        
    except Exception as e:
        print(f"Error downloading UNSW-NB15 network traffic data: {e}")
        raise

def _load_traffic_data(category: str) -> pd.DataFrame:
    """
    Load network traffic data for a specific category.
    
    Args:
        category: The traffic category (attack type or 'Normal')
        
    Returns:
        DataFrame with time and traffic volume columns for the category
    """
    if category not in TRAFFIC_CATEGORIES:
        raise ValueError(f"Invalid category {category}. Must be one of {TRAFFIC_CATEGORIES}")
    
    data_file = _download_and_cache_data()
    
    # Load the full dataset
    df = pd.read_csv(data_file)
    if 'attack_cat' in df.columns:
        df['attack_cat'] = _normalize_attack_categories(df['attack_cat']).fillna('Normal')
    
    # Filter for the specific category
    category_df = df[df['attack_cat'] == category].copy()
    
    if len(category_df) == 0:
        raise ValueError(f"No data found for category {category}")
    
    # Create traffic volume feature (combine source and destination bytes)
    category_df['traffic_volume'] = category_df['sbytes'] + category_df['dbytes']
    
    # Sort by time proxy and reset index
    category_df = category_df.sort_values('time_proxy').reset_index(drop=True)
    
    # Group by time windows to create time series (aggregate every 100 records)
    window_size = 100
    aggregated_data = []
    
    for i in range(0, len(category_df), window_size):
        window_data = category_df.iloc[i:i+window_size]
        if len(window_data) > 0:
            aggregated_data.append({
                'time_window': i // window_size,
                'traffic_volume': window_data['traffic_volume'].sum()
            })
    
    agg_df = pd.DataFrame(aggregated_data)
    
    if len(agg_df) < WINDOW_SIZE + 1:
        # If not enough data, use the raw data without aggregation
        return category_df[['time_proxy', 'traffic_volume']].rename(
            columns={'time_proxy': 'time_window'}
        )
    
    return agg_df

def _create_sliding_windows(traffic_volumes: np.ndarray, window_size: int) -> List[Dict[str, Any]]:
    """
    Create sliding window samples from network traffic time series.
    
    Args:
        traffic_volumes: Array of traffic volume values
        window_size: Size of the input window
        
    Returns:
        List of dictionaries with 'x' (input window) and 'y' (target) keys
    """
    if len(traffic_volumes) < window_size + 1:
        raise ValueError(f"Not enough data points. Need at least {window_size + 1}, got {len(traffic_volumes)}")
    
    samples = []
    for i in range(len(traffic_volumes) - window_size):
        x = traffic_volumes[i:i + window_size].tolist()
        y = float(traffic_volumes[i + window_size])
        samples.append({"x": x, "y": y})
    
    return samples

def load_dataset(partition_id: int, num_examples: int = 500) -> Dataset:
    """
    Load UNSW-NB15 network traffic dataset for a specific traffic category (partition).
    
    Args:
        partition_id: Category ID (0-9, mapped to traffic categories)
        num_examples: Number of sliding window examples to generate
        
    Returns:
        Dataset object with 'x' (traffic volume windows) and 'y' (next traffic volume) columns
    """
    if not (0 <= partition_id < len(TRAFFIC_CATEGORIES)):
        raise ValueError(f"partition_id must be in range 0-{len(TRAFFIC_CATEGORIES)-1}, got {partition_id}")
    
    # Map partition_id to actual traffic category
    category = TRAFFIC_CATEGORIES[partition_id]
    
    # Load traffic data for this category
    category_df = _load_traffic_data(category)
    traffic_volumes = category_df['traffic_volume'].values
    
    # Normalize traffic volumes (log transform + z-score normalization)
    # Add small constant to avoid log(0)
    traffic_volumes = np.log1p(traffic_volumes)  # log(1 + x)
    
    # Z-score normalization per category
    volume_mean = np.mean(traffic_volumes)
    volume_std = np.std(traffic_volumes)
    if volume_std > 0:
        traffic_volumes = (traffic_volumes - volume_mean) / volume_std
    
    # Create sliding windows
    samples = _create_sliding_windows(traffic_volumes, WINDOW_SIZE)
    
    # Limit to requested number of examples
    if len(samples) > num_examples:
        # Take evenly spaced samples to cover the full time range
        indices = np.linspace(0, len(samples) - 1, num_examples, dtype=int)
        samples = [samples[i] for i in indices]
    
    return Dataset.from_list(samples) 