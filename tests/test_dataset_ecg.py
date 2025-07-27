import pytest
import numpy as np
from datasets import Dataset
from fedcast.datasets.dataset_ecg import load_dataset, get_patient_ids, WINDOW_SIZE, RECORD_NAMES
from pathlib import Path
import os

# Test cases for partition IDs
PARTITION_ID_1 = 0
PARTITION_ID_2 = 1
NUM_EXAMPLES = 10


def test_get_patient_ids():
    """Tests that get_patient_ids returns the expected patient IDs."""
    patient_ids = get_patient_ids()
    assert isinstance(patient_ids, list), "Patient IDs should be returned as a list."
    assert len(patient_ids) == len(RECORD_NAMES), f"Should return {len(RECORD_NAMES)} patient IDs."
    assert patient_ids == RECORD_NAMES, "Patient IDs should match RECORD_NAMES."


def test_load_dataset_output_type():
    """Tests that load_dataset returns a Dataset object."""
    dataset = load_dataset(PARTITION_ID_1, num_examples=NUM_EXAMPLES)
    assert isinstance(dataset, Dataset), "The function should return a Dataset object."


def test_load_dataset_structure_and_shape():
    """Tests the structure and shape of the returned dataset."""
    dataset = load_dataset(PARTITION_ID_1, num_examples=NUM_EXAMPLES)

    # Check number of examples
    assert len(dataset) == NUM_EXAMPLES, f"Dataset should have {NUM_EXAMPLES} examples."

    # Check column names
    assert "x" in dataset.features, "Dataset should have an 'x' column."
    assert "y" in dataset.features, "Dataset should have a 'y' column."

    # Check shape of a single example
    sample = dataset[0]
    assert len(sample["x"]) == WINDOW_SIZE, f"Input 'x' should have length {WINDOW_SIZE}."
    assert isinstance(sample["y"], (float, np.floating, int, np.integer)), "Target 'y' should be a number."


def test_load_dataset_reproducibility():
    """Tests that load_dataset is reproducible for the same partition_id."""
    dataset1 = load_dataset(PARTITION_ID_1, num_examples=NUM_EXAMPLES)
    dataset2 = load_dataset(PARTITION_ID_1, num_examples=NUM_EXAMPLES)

    # Compare a few samples to ensure they are identical
    for i in range(min(5, NUM_EXAMPLES)):
        assert np.allclose(dataset1[i]["x"], dataset2[i]["x"]), f"Sample {i} 'x' data should be identical."
        assert np.isclose(dataset1[i]["y"], dataset2[i]["y"]), f"Sample {i} 'y' data should be identical."


def test_load_dataset_content_not_all_zero():
    """Tests that the ECG data is not all zeros."""
    dataset = load_dataset(PARTITION_ID_1, num_examples=NUM_EXAMPLES)
    
    # At least one value in the first sample should be nonzero for real ECG data
    x_data = np.array(dataset[0]["x"])
    y_data = dataset[0]["y"]
    
    assert np.any(x_data != 0) or y_data != 0, "ECG data should not be all zeros."
    assert not np.all(np.isnan(x_data)), "ECG data should not contain NaN values."
    assert not np.isnan(y_data), "Target value should not be NaN."


def test_load_dataset_different_patients():
    """Tests that different patients produce different data."""
    dataset1 = load_dataset(PARTITION_ID_1, num_examples=NUM_EXAMPLES)
    dataset2 = load_dataset(PARTITION_ID_2, num_examples=NUM_EXAMPLES)
    
    # Different patients should have different ECG patterns
    x1 = np.array(dataset1[0]["x"])
    x2 = np.array(dataset2[0]["x"])
    
    # At least some values should be different between patients
    assert not np.allclose(x1, x2), "Different patients should have different ECG patterns."


def test_load_dataset_data_normalization():
    """Tests that ECG data appears to be normalized and has reasonable range."""
    dataset = load_dataset(PARTITION_ID_1, num_examples=NUM_EXAMPLES)
    
    # Collect all data points to check normalization properties
    all_x_data = []
    all_y_data = []
    
    for i in range(NUM_EXAMPLES):
        all_x_data.extend(dataset[i]["x"])
        all_y_data.append(dataset[i]["y"])
    
    x_array = np.array(all_x_data)
    y_array = np.array(all_y_data)
    
    # Check that data has reasonable range for normalized ECG data
    # ECG data after normalization should be roughly within [-5, 5] range
    assert np.all(x_array > -10) and np.all(x_array < 10), "Normalized ECG data should be within reasonable range."
    assert np.all(y_array > -10) and np.all(y_array < 10), "Normalized ECG targets should be within reasonable range."
    
    # Check that the data has some variance (not constant)
    x_std = np.std(x_array)
    assert x_std > 0.01, f"ECG data should have some variance, got std={x_std}"


def test_load_dataset_sequential_data():
    """Tests that the data follows time series sequential pattern."""
    dataset = load_dataset(PARTITION_ID_1, num_examples=NUM_EXAMPLES)
    
    # Get first two samples
    sample1 = dataset[0]
    sample2 = dataset[1]
    
    x1 = np.array(sample1["x"])
    x2 = np.array(sample2["x"])
    y1 = sample1["y"]
    
    # The second sample's input should be the first sample's input shifted by one
    # (x2[0:19] should equal x1[1:20], and x2[19] should equal y1)
    assert np.allclose(x2[:-1], x1[1:]), "Sequential samples should have overlapping windows."
    assert np.isclose(x2[-1], y1), "Next sample's last input should be current sample's target."


def test_load_dataset_invalid_partition_id():
    """Tests error handling for invalid partition IDs."""
    num_patients = len(get_patient_ids())
    
    # Test negative partition ID
    with pytest.raises(ValueError, match="partition_id must be between"):
        load_dataset(-1, num_examples=NUM_EXAMPLES)
    
    # Test partition ID that's too large
    with pytest.raises(ValueError, match="partition_id must be between"):
        load_dataset(num_patients, num_examples=NUM_EXAMPLES)


def test_load_dataset_varying_num_examples():
    """Tests that the dataset can handle different numbers of examples."""
    small_dataset = load_dataset(PARTITION_ID_1, num_examples=5)
    large_dataset = load_dataset(PARTITION_ID_1, num_examples=50)
    
    assert len(small_dataset) == 5, "Small dataset should have 5 examples."
    assert len(large_dataset) == 50, "Large dataset should have 50 examples."
    
    # First examples should be identical
    assert np.allclose(small_dataset[0]["x"], large_dataset[0]["x"]), "First samples should be identical."
    assert np.isclose(small_dataset[0]["y"], large_dataset[0]["y"]), "First targets should be identical."


def test_load_dataset_data_continuity():
    """Tests that ECG data shows physiological continuity (no extreme jumps)."""
    dataset = load_dataset(PARTITION_ID_1, num_examples=NUM_EXAMPLES)
    
    # Check that values don't have extreme jumps (typical for ECG signals)
    for i in range(NUM_EXAMPLES):
        x_data = np.array(dataset[i]["x"])
        
        # Calculate differences between consecutive points
        diffs = np.diff(x_data)
        
        # ECG signals shouldn't have extreme jumps (after normalization)
        # Allow some tolerance for artifacts but catch obvious errors
        assert np.all(np.abs(diffs) < 10), f"Sample {i} has extreme jumps in ECG data."


def test_multiple_patients_coverage():
    """Tests that we can load data from multiple different patients."""
    num_patients = min(5, len(get_patient_ids()))  # Test first 5 patients
    datasets = []
    
    for patient_id in range(num_patients):
        dataset = load_dataset(patient_id, num_examples=5)
        datasets.append(dataset)
        
        # Each dataset should be valid
        assert len(dataset) == 5, f"Patient {patient_id} dataset should have 5 examples."
        assert "x" in dataset.features and "y" in dataset.features, f"Patient {patient_id} dataset should have x and y columns."
    
    # Verify that patients have different data patterns
    for i in range(1, num_patients):
        x0 = np.array(datasets[0][0]["x"])
        xi = np.array(datasets[i][0]["x"])
        assert not np.allclose(x0, xi, rtol=1e-3), f"Patient 0 and patient {i} should have different ECG patterns." 