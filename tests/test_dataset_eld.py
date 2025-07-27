import pytest
import numpy as np
from datasets import Dataset
from fedcast.datasets.dataset_eld import load_dataset, WINDOW_SIZE

PARTITION_ID = 0
NUM_EXAMPLES = 10

def test_load_dataset_output_type():
    dataset = load_dataset(PARTITION_ID, num_examples=NUM_EXAMPLES)
    assert isinstance(dataset, Dataset), "The function should return a Dataset object."

def test_load_dataset_structure_and_shape():
    dataset = load_dataset(PARTITION_ID, num_examples=NUM_EXAMPLES)
    assert len(dataset) == NUM_EXAMPLES, f"Dataset should have {NUM_EXAMPLES} examples."
    assert "x" in dataset.features, "Dataset should have an 'x' column."
    assert "y" in dataset.features, "Dataset should have a 'y' column."
    sample = dataset[0]
    assert len(sample["x"]) == WINDOW_SIZE, f"Input 'x' should have length {WINDOW_SIZE}."
    assert isinstance(sample["y"], (float, np.floating, int, np.integer)), "Target 'y' should be a number."

def test_load_dataset_reproducibility():
    dataset1 = load_dataset(PARTITION_ID, num_examples=NUM_EXAMPLES)
    dataset2 = load_dataset(PARTITION_ID, num_examples=NUM_EXAMPLES)
    for i in range(min(5, NUM_EXAMPLES)):
        assert np.allclose(dataset1[i]["x"], dataset2[i]["x"]), f"Sample {i} 'x' data should be identical."
        assert np.isclose(dataset1[i]["y"], dataset2[i]["y"]), f"Sample {i} 'y' data should be identical."
