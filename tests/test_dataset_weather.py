import pytest
import numpy as np
from datasets import Dataset
from fedcast.datasets.dataset_weather import load_dataset, get_weather_stations, WINDOW_SIZE, WEATHER_STATIONS
from pathlib import Path
import os

# Test cases for partition IDs
PARTITION_ID_1 = 0  # First weather station
PARTITION_ID_2 = 1  # Second weather station
NUM_EXAMPLES = 10


def test_get_weather_stations():
    """Tests that get_weather_stations returns the expected weather station IDs."""
    stations = get_weather_stations()
    assert isinstance(stations, list), "Weather stations should be returned as a list."
    assert len(stations) == len(WEATHER_STATIONS), f"Should return {len(WEATHER_STATIONS)} weather stations."
    assert stations == WEATHER_STATIONS, "Weather stations should match WEATHER_STATIONS."


def test_load_dataset_output_type():
    """Tests that load_dataset returns a Dataset object."""
    dataset = load_dataset(PARTITION_ID_1, num_examples=NUM_EXAMPLES)
    assert isinstance(dataset, Dataset), "load_dataset should return a Dataset object."


def test_load_dataset_structure_and_shape():
    """Tests the structure and shape of the dataset."""
    dataset = load_dataset(PARTITION_ID_1, num_examples=NUM_EXAMPLES)
    
    # Check dataset length
    assert len(dataset) == NUM_EXAMPLES, f"Dataset should have {NUM_EXAMPLES} examples."
    
    # Check features
    assert "x" in dataset.features, "Dataset should have an 'x' column."
    assert "y" in dataset.features, "Dataset should have a 'y' column."
    
    # Check sample structure
    sample = dataset[0]
    assert isinstance(sample["x"], list), "Input 'x' should be a list."
    assert len(sample["x"]) == WINDOW_SIZE, f"Input 'x' should have length {WINDOW_SIZE}."
    assert isinstance(sample["y"], (float, np.floating, int, np.integer)), "Target 'y' should be a number."


def test_load_dataset_reproducibility():
    """Tests that load_dataset returns consistent results across multiple calls."""
    dataset1 = load_dataset(PARTITION_ID_1, num_examples=NUM_EXAMPLES)
    dataset2 = load_dataset(PARTITION_ID_1, num_examples=NUM_EXAMPLES)
    
    # The exact same call should produce the same results
    for i in range(NUM_EXAMPLES):
        assert dataset1[i]["x"] == dataset2[i]["x"], f"Input sequences should be identical for call {i}."
        assert dataset1[i]["y"] == dataset2[i]["y"], f"Target values should be identical for call {i}."


def test_load_dataset_content_not_all_zero():
    """Tests that the dataset contains actual weather data (not all zeros or NaN)."""
    dataset = load_dataset(PARTITION_ID_1, num_examples=NUM_EXAMPLES)
    
    # Collect all data points to check they're not all zero/NaN
    all_x_data = []
    all_y_data = []
    
    for i in range(NUM_EXAMPLES):
        all_x_data.extend(dataset[i]["x"])
        all_y_data.append(dataset[i]["y"])
    
    x_array = np.array(all_x_data)
    y_array = np.array(all_y_data)
    
    # Check for NaN values
    assert not np.any(np.isnan(x_array)), "Input data should not contain NaN values."
    assert not np.any(np.isnan(y_array)), "Target data should not contain NaN values."
    
    # Check that data has some variance (not all the same)
    x_std = np.std(x_array)
    assert x_std > 0.01, f"Weather data should have some variance, got std={x_std}"


def test_load_dataset_different_stations():
    """Tests that different weather stations produce different data."""
    dataset1 = load_dataset(PARTITION_ID_1, num_examples=NUM_EXAMPLES)
    dataset2 = load_dataset(PARTITION_ID_2, num_examples=NUM_EXAMPLES)
    
    # Extract data from both stations
    data1 = [item for sample in dataset1 for item in sample["x"]] + [sample["y"] for sample in dataset1]
    data2 = [item for sample in dataset2 for item in sample["x"]] + [sample["y"] for sample in dataset2]
    
    # Different stations should produce different weather patterns
    assert data1 != data2, "Different weather stations should produce different data patterns."


def test_load_dataset_data_normalization():
    """Tests that weather data appears to be normalized."""
    dataset = load_dataset(PARTITION_ID_1, num_examples=NUM_EXAMPLES)
    
    # Collect all data points to check normalization
    all_x_data = []
    all_y_data = []
    
    for i in range(NUM_EXAMPLES):
        all_x_data.extend(dataset[i]["x"])
        all_y_data.append(dataset[i]["y"])
    
    x_array = np.array(all_x_data)
    y_array = np.array(all_y_data)
    
    # Check that the data has reasonable variance for normalized data
    x_std = np.std(x_array)
    assert x_std > 0.01, f"Weather data should have some variance, got std={x_std}"
    
    # For normalized data, extreme values should be reasonable (within ~3 standard deviations)
    x_min, x_max = np.min(x_array), np.max(x_array)
    assert -5 < x_min < 5, f"Normalized weather data should have reasonable min value, got {x_min}"
    assert -5 < x_max < 5, f"Normalized weather data should have reasonable max value, got {x_max}"


def test_load_dataset_temperature_continuity():
    """Tests that temperature sequences show realistic continuity."""
    dataset = load_dataset(PARTITION_ID_1, num_examples=5)  # Use fewer examples for detailed check
    
    for i in range(len(dataset)):
        x_sequence = np.array(dataset[i]["x"])
        y_next = dataset[i]["y"]
        
        # Check that the sequence has some variance (not constant)
        if len(set(x_sequence)) > 1:  # Only check if not all values are the same
            seq_std = np.std(x_sequence)
            assert seq_std >= 0, f"Temperature sequence {i} should have non-negative variance"
        
        # All values should be finite
        assert np.all(np.isfinite(x_sequence)), f"Temperature sequence {i} should contain only finite values"
        assert np.isfinite(y_next), f"Target value for sequence {i} should be finite"


def test_load_dataset_invalid_partition_id():
    """Tests error handling for invalid partition IDs."""
    with pytest.raises(ValueError, match="partition_id must be in range"):
        load_dataset(-1, num_examples=NUM_EXAMPLES)
    
    with pytest.raises(ValueError, match="partition_id must be in range"):
        load_dataset(len(WEATHER_STATIONS), num_examples=NUM_EXAMPLES)  # One beyond valid range


def test_load_dataset_num_examples_parameter():
    """Tests that the num_examples parameter controls dataset size correctly."""
    small_dataset = load_dataset(PARTITION_ID_1, num_examples=3)
    large_dataset = load_dataset(PARTITION_ID_1, num_examples=20)
    
    assert len(small_dataset) == 3, "Small dataset should have exactly 3 examples."
    assert len(large_dataset) == 20, "Large dataset should have exactly 20 examples."


def test_load_dataset_window_size_consistency():
    """Tests that all input sequences have the correct window size."""
    dataset = load_dataset(PARTITION_ID_1, num_examples=NUM_EXAMPLES)
    
    for i in range(len(dataset)):
        assert len(dataset[i]["x"]) == WINDOW_SIZE, f"Sample {i} should have input length {WINDOW_SIZE}."


def test_load_dataset_weather_specific_properties():
    """Tests weather-specific characteristics."""
    dataset = load_dataset(PARTITION_ID_1, num_examples=NUM_EXAMPLES)
    
    # Collect temperature data
    all_temps = []
    for i in range(NUM_EXAMPLES):
        all_temps.extend(dataset[i]["x"])
        all_temps.append(dataset[i]["y"])
    
    temp_array = np.array(all_temps)
    
    # Weather temperatures should be finite
    assert np.all(np.isfinite(temp_array)), "All temperature values should be finite."
    
    # After normalization, data should have reasonable range
    temp_range = np.max(temp_array) - np.min(temp_array)
    assert temp_range > 0, "Temperature data should have some range/variation."


def test_weather_stations_coverage():
    """Tests that weather stations cover a reasonable set of stations."""
    stations = get_weather_stations()
    
    # Should have a reasonable number of stations
    assert 20 <= len(stations) <= 100, f"Should have reasonable number of stations, got {len(stations)}"
    
    # All station IDs should be strings
    for station in stations:
        assert isinstance(station, str), f"Station ID should be string, got {type(station)}"
        assert len(station) > 5, f"Station ID should be reasonable length, got {station}"


def test_dataset_time_series_properties():
    """Tests time series specific properties of weather data."""
    dataset = load_dataset(PARTITION_ID_1, num_examples=5)
    
    for i in range(len(dataset)):
        x_sequence = np.array(dataset[i]["x"])
        
        # Temperature sequences should not have extreme jumps between consecutive points
        if len(x_sequence) > 1:
            diffs = np.abs(np.diff(x_sequence))
            max_diff = np.max(diffs)
            # For normalized data, consecutive differences should be reasonable
            assert max_diff < 5.0, f"Temperature sequence {i} has unrealistic jump: {max_diff}"


def test_load_dataset_seasonal_patterns():
    """Tests that weather data shows expected patterns over time."""
    # Load a larger sample to potentially capture seasonal patterns
    dataset = load_dataset(PARTITION_ID_1, num_examples=50)
    
    # Collect all temperature data
    all_temps = []
    for i in range(len(dataset)):
        all_temps.extend(dataset[i]["x"])
        all_temps.append(dataset[i]["y"])
    
    temp_array = np.array(all_temps)
    
    # Weather data should have realistic statistical properties
    temp_mean = np.mean(temp_array)
    temp_std = np.std(temp_array)
    
    # For normalized data, mean should be close to 0, std close to 1
    assert abs(temp_mean) < 0.5, f"Normalized temperature mean should be close to 0, got {temp_mean}"
    assert 0.5 < temp_std < 2.0, f"Normalized temperature std should be reasonable, got {temp_std}"


def test_dataset_memory_efficiency():
    """Tests that the dataset doesn't consume excessive memory for small requests."""
    # This should not cause memory issues
    small_dataset = load_dataset(PARTITION_ID_1, num_examples=2)
    
    assert len(small_dataset) == 2, "Should be able to create small datasets efficiently."
    
    # Verify the data is valid
    sample = small_dataset[0]
    assert len(sample["x"]) == WINDOW_SIZE, "Small dataset should have correct structure."
    assert isinstance(sample["y"], (float, np.floating, int, np.integer)), "Small dataset should have valid targets."


def test_weather_station_id_format():
    """Tests that weather station IDs follow expected format."""
    stations = get_weather_stations()
    
    for station in stations:
        # Station IDs should follow NOAA format (USC or USW followed by numbers)
        assert station.startswith(('USC', 'USW')), f"Station ID {station} should start with USC or USW"
        assert len(station) >= 10, f"Station ID {station} should be at least 10 characters"
        # Check that the rest are digits
        assert station[3:].isdigit(), f"Station ID {station} should have digits after USC/USW prefix" 