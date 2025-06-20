import pytest
import numpy as np
from datasets import Dataset
from fedcast.datasets.dataset_sinus import get_sine_params, load_dataset, WINDOW_SIZE

# Test cases for partition IDs
PARTITION_ID_1 = 1
PARTITION_ID_2 = 2
NUM_EXAMPLES = 100


def test_get_sine_params_determinism():
    """Tests that get_sine_params is deterministic for the same partition_id."""
    params1 = get_sine_params(PARTITION_ID_1)
    params2 = get_sine_params(PARTITION_ID_1)
    assert params1 == params2, "Parameters should be identical for the same partition ID."


def test_get_sine_params_uniqueness():
    """Tests that get_sine_params generates unique parameters for different partition_ids."""
    params1 = get_sine_params(PARTITION_ID_1)
    params2 = get_sine_params(PARTITION_ID_2)
    assert params1 != params2, "Parameters should be different for different partition IDs."


def test_get_sine_params_output_type():
    """Tests the output type of get_sine_params."""
    params = get_sine_params(PARTITION_ID_1)
    assert isinstance(params, tuple), "Output should be a tuple."
    assert len(params) == 4, "Tuple should contain 4 elements."
    assert all(
        isinstance(p, (float, np.floating)) for p in params
    ), "All elements should be floats."


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
    assert isinstance(
        sample["y"], (float, np.floating)
    ), "Target 'y' should be a float."


def test_load_dataset_reproducibility():
    """Tests that load_dataset is reproducible for the same partition_id."""
    dataset1 = load_dataset(PARTITION_ID_1, num_examples=NUM_EXAMPLES)
    dataset2 = load_dataset(PARTITION_ID_1, num_examples=NUM_EXAMPLES)

    # Compare a few samples to ensure they are identical
    for i in range(min(5, NUM_EXAMPLES)):
        assert np.array_equal(
            dataset1[i]["x"], dataset2[i]["x"]
        ), f"Sample {i} 'x' data should be identical."
        assert np.isclose(
            dataset1[i]["y"], dataset2[i]["y"]
        ), f"Sample {i} 'y' data should be identical."


def test_load_dataset_content():
    """
    Tests if the data in the dataset correctly corresponds to the generated sine wave.
    """
    params = get_sine_params(PARTITION_ID_1)
    amplitude, frequency, phase, vertical_shift = params

    dataset = load_dataset(PARTITION_ID_1, num_examples=NUM_EXAMPLES)

    # Re-generate the first part of the time series to verify the dataset's content
    total_points = WINDOW_SIZE + 1  # Enough points for the first sample
    x_time = np.arange(total_points)
    time_series = amplitude * np.sin(frequency * x_time + phase) + vertical_shift

    expected_x = time_series[:WINDOW_SIZE]
    expected_y = time_series[WINDOW_SIZE]

    actual_x = dataset[0]["x"]
    actual_y = dataset[0]["y"]

    assert np.allclose(
        actual_x, expected_x
    ), "The 'x' data does not match the expected sine wave."
    assert np.isclose(
        actual_y, expected_y
    ), "The 'y' data does not match the expected sine wave." 