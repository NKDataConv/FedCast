import pytest
import numpy as np
from datasets import Dataset
from fedcast.datasets.dataset_intel_iot import load_dataset, get_sensor_ids, WINDOW_SIZE, SENSOR_IDS
from pathlib import Path
import os

# Test cases for partition IDs
PARTITION_ID_1 = 0  # Sensor 1
PARTITION_ID_2 = 1  # Sensor 2
NUM_EXAMPLES = 10


def test_get_sensor_ids():
    """Tests that get_sensor_ids returns the expected sensor IDs."""
    sensor_ids = get_sensor_ids()
    assert isinstance(sensor_ids, list), "Sensor IDs should be returned as a list."
    assert len(sensor_ids) == len(SENSOR_IDS), f"Should return {len(SENSOR_IDS)} sensor IDs."
    assert sensor_ids == SENSOR_IDS, "Sensor IDs should match SENSOR_IDS."


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
    """Tests that the dataset contains actual temperature data (not all zeros or NaN)."""
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
    
    # Check that not all values are zero (temperature data should vary)
    assert not np.all(x_array == 0), "Input data should not be all zeros."
    assert not np.all(y_array == 0), "Target data should not be all zeros."


def test_load_dataset_different_sensors():
    """Tests that different sensors produce different temperature data."""
    dataset1 = load_dataset(PARTITION_ID_1, num_examples=NUM_EXAMPLES)
    dataset2 = load_dataset(PARTITION_ID_2, num_examples=NUM_EXAMPLES)
    
    # Extract data from both sensors
    data1 = [item for sample in dataset1 for item in sample["x"]] + [sample["y"] for sample in dataset1]
    data2 = [item for sample in dataset2 for item in sample["x"]] + [sample["y"] for sample in dataset2]
    
    # Different sensors should produce different temperature readings
    assert data1 != data2, "Different sensors should produce different temperature data."


def test_load_dataset_data_normalization():
    """Tests that temperature data appears to be normalized."""
    dataset = load_dataset(PARTITION_ID_1, num_examples=NUM_EXAMPLES)
    
    # Collect all data points to check normalization
    all_x_data = []
    all_y_data = []
    
    for i in range(NUM_EXAMPLES):
        all_x_data.extend(dataset[i]["x"])
        all_y_data.append(dataset[i]["y"])
    
    x_array = np.array(all_x_data)
    y_array = np.array(all_y_data)
    
    # Check that the data has some variance (not constant)
    x_std = np.std(x_array)
    assert x_std > 0.01, f"Temperature data should have some variance, got std={x_std}"
    
    # For normalized data, extreme values should be reasonable
    x_min, x_max = np.min(x_array), np.max(x_array)
    assert -10 < x_min < 10, f"Normalized temperature data should have reasonable min value, got {x_min}"
    assert -10 < x_max < 10, f"Normalized temperature data should have reasonable max value, got {x_max}"


def test_load_dataset_time_series_continuity():
    """Tests that temperature sequences show realistic continuity (no sudden jumps)."""
    dataset = load_dataset(PARTITION_ID_1, num_examples=5)  # Use fewer examples for detailed check
    
    for i in range(len(dataset)):
        x_sequence = np.array(dataset[i]["x"])
        y_next = dataset[i]["y"]
        
        # Check for reasonable continuity in temperature readings
        # Temperature shouldn't have extreme jumps between consecutive readings
        diffs = np.abs(np.diff(x_sequence))
        max_diff = np.max(diffs)
        
        # For normalized data, consecutive differences should be reasonable
        assert max_diff < 5.0, f"Temperature sequence {i} has unrealistic jump: {max_diff}"
        
        # The predicted next value should be reasonably close to the last input value
        last_input = x_sequence[-1]
        next_diff = abs(y_next - last_input)
        assert next_diff < 5.0, f"Prediction discontinuity too large in sequence {i}: {next_diff}"


def test_load_dataset_invalid_partition_id():
    """Tests error handling for invalid partition IDs."""
    with pytest.raises(ValueError, match="partition_id must be in range"):
        load_dataset(-1, num_examples=NUM_EXAMPLES)
    
    with pytest.raises(ValueError, match="partition_id must be in range"):
        load_dataset(len(SENSOR_IDS), num_examples=NUM_EXAMPLES)  # One beyond valid range


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


def test_load_dataset_sensor_data_characteristics():
    """Tests IoT-specific characteristics of sensor temperature data."""
    dataset = load_dataset(PARTITION_ID_1, num_examples=NUM_EXAMPLES)
    
    # Collect temperature data
    all_temps = []
    for i in range(NUM_EXAMPLES):
        all_temps.extend(dataset[i]["x"])
        all_temps.append(dataset[i]["y"])
    
    temp_array = np.array(all_temps)
    
    # Check that temperature values are reasonable for indoor sensors
    # After normalization, most values should be within a reasonable range
    temp_std = np.std(temp_array)
    assert 0.1 < temp_std < 5.0, f"Temperature standard deviation should be reasonable, got {temp_std}"
    
    # Temperature data should not have any infinite values
    assert np.all(np.isfinite(temp_array)), "All temperature values should be finite." 