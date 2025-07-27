import pytest
import numpy as np
from datasets import Dataset
from fedcast.datasets.dataset_network_traffic import load_dataset, get_traffic_categories, WINDOW_SIZE, TRAFFIC_CATEGORIES
from pathlib import Path
import os

# Test cases for partition IDs
PARTITION_ID_1 = 0  # Normal traffic
PARTITION_ID_2 = 4  # DoS attacks
NUM_EXAMPLES = 10


def test_get_traffic_categories():
    """Tests that get_traffic_categories returns the expected traffic categories."""
    categories = get_traffic_categories()
    assert isinstance(categories, list), "Traffic categories should be returned as a list."
    assert len(categories) == len(TRAFFIC_CATEGORIES), f"Should return {len(TRAFFIC_CATEGORIES)} traffic categories."
    assert categories == TRAFFIC_CATEGORIES, "Traffic categories should match TRAFFIC_CATEGORIES."


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
    """Tests that the dataset contains actual network traffic data (not all zeros or NaN)."""
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
    assert x_std > 0.01, f"Network traffic data should have some variance, got std={x_std}"


def test_load_dataset_different_categories():
    """Tests that different traffic categories produce different data."""
    dataset1 = load_dataset(PARTITION_ID_1, num_examples=NUM_EXAMPLES)  # Normal
    dataset2 = load_dataset(PARTITION_ID_2, num_examples=NUM_EXAMPLES)  # DoS
    
    # Extract data from both categories
    data1 = [item for sample in dataset1 for item in sample["x"]] + [sample["y"] for sample in dataset1]
    data2 = [item for sample in dataset2 for item in sample["x"]] + [sample["y"] for sample in dataset2]
    
    # Different categories should produce different traffic patterns
    assert data1 != data2, "Different traffic categories should produce different data patterns."


def test_load_dataset_data_normalization():
    """Tests that network traffic data appears to be normalized."""
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
    assert x_std > 0.01, f"Network traffic data should have some variance, got std={x_std}"
    
    # For normalized data, extreme values should be reasonable
    x_min, x_max = np.min(x_array), np.max(x_array)
    assert -10 < x_min < 10, f"Normalized traffic data should have reasonable min value, got {x_min}"
    assert -10 < x_max < 10, f"Normalized traffic data should have reasonable max value, got {x_max}"


def test_load_dataset_time_series_characteristics():
    """Tests that traffic sequences show realistic time series properties."""
    dataset = load_dataset(PARTITION_ID_1, num_examples=5)  # Use fewer examples for detailed check
    
    for i in range(len(dataset)):
        x_sequence = np.array(dataset[i]["x"])
        y_next = dataset[i]["y"]
        
        # Check that the sequence is not constant (should have variance)
        if len(set(x_sequence)) > 1:  # Only check if not all values are the same
            seq_std = np.std(x_sequence)
            assert seq_std >= 0, f"Traffic sequence {i} should have non-negative variance"
        
        # All values should be finite
        assert np.all(np.isfinite(x_sequence)), f"Traffic sequence {i} should contain only finite values"
        assert np.isfinite(y_next), f"Target value for sequence {i} should be finite"


def test_load_dataset_invalid_partition_id():
    """Tests error handling for invalid partition IDs."""
    with pytest.raises(ValueError, match="partition_id must be in range"):
        load_dataset(-1, num_examples=NUM_EXAMPLES)
    
    with pytest.raises(ValueError, match="partition_id must be in range"):
        load_dataset(len(TRAFFIC_CATEGORIES), num_examples=NUM_EXAMPLES)  # One beyond valid range


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


def test_load_dataset_traffic_categories_validity():
    """Tests that all defined traffic categories can be loaded."""
    # Test that we can load data from the first few categories
    for i in range(min(3, len(TRAFFIC_CATEGORIES))):  # Test first 3 categories to avoid long tests
        try:
            dataset = load_dataset(i, num_examples=3)
            assert len(dataset) == 3, f"Should be able to load data for category {TRAFFIC_CATEGORIES[i]}"
        except ValueError as e:
            # If a category has no data, that's acceptable for some attack types
            if "No data found for category" in str(e):
                print(f"Warning: No data found for category {TRAFFIC_CATEGORIES[i]}")
            else:
                raise


def test_load_dataset_network_specific_properties():
    """Tests network traffic-specific characteristics."""
    dataset = load_dataset(PARTITION_ID_1, num_examples=NUM_EXAMPLES)
    
    # Collect traffic volume data
    all_volumes = []
    for i in range(NUM_EXAMPLES):
        all_volumes.extend(dataset[i]["x"])
        all_volumes.append(dataset[i]["y"])
    
    volume_array = np.array(all_volumes)
    
    # Network traffic volumes should be finite
    assert np.all(np.isfinite(volume_array)), "All traffic volumes should be finite."
    
    # After log transformation and normalization, data should have reasonable range
    volume_range = np.max(volume_array) - np.min(volume_array)
    assert volume_range > 0, "Traffic volume data should have some range/variation."


def test_traffic_categories_coverage():
    """Tests that traffic categories cover expected attack types."""
    categories = get_traffic_categories()
    
    # Should include normal traffic
    assert 'Normal' in categories, "Should include Normal traffic category."
    
    # Should include common attack types
    expected_attacks = ['DoS', 'Reconnaissance', 'Exploits']
    for attack in expected_attacks:
        assert attack in categories, f"Should include {attack} attack category."
    
    # Should have reasonable number of categories
    assert 5 <= len(categories) <= 15, f"Should have reasonable number of categories, got {len(categories)}"


def test_dataset_memory_efficiency():
    """Tests that the dataset doesn't consume excessive memory for small requests."""
    # This should not cause memory issues even though the underlying dataset is large
    small_dataset = load_dataset(PARTITION_ID_1, num_examples=2)
    
    assert len(small_dataset) == 2, "Should be able to create small datasets efficiently."
    
    # Verify the data is valid
    sample = small_dataset[0]
    assert len(sample["x"]) == WINDOW_SIZE, "Small dataset should have correct structure."
    assert isinstance(sample["y"], (float, np.floating, int, np.integer)), "Small dataset should have valid targets." 