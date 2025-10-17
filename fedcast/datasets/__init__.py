"""
Dataset modules for FedCast.

This module provides various dataset implementations for time series forecasting
in federated learning scenarios.
"""

from .dataset_sinus import load_dataset as load_sinus_dataset
from .dataset_ecg import load_dataset as load_ecg_dataset
from .dataset_eld import load_dataset as load_eld_dataset
from .dataset_intel_iot import load_dataset as load_intel_iot_dataset
from .dataset_network_traffic import load_dataset as load_network_traffic_dataset
from .dataset_stocks import load_dataset as load_stocks_dataset
from .dataset_weather import load_dataset as load_weather_dataset

__all__ = [
    "load_sinus_dataset",
    "load_ecg_dataset", 
    "load_eld_dataset",
    "load_intel_iot_dataset",
    "load_network_traffic_dataset",
    "load_stocks_dataset",
    "load_weather_dataset",
]
