import importlib
import pkgutil
import pytest
import numpy as np
from datasets import Dataset
import fedcast.datasets

NUM_EXAMPLES = 5

def get_dataset_modules():
    package = fedcast.datasets
    for _, name, ispkg in pkgutil.iter_modules(package.__path__):
        if not ispkg and name.startswith("dataset_"):
            yield name

dataset_module_names = list(get_dataset_modules())

@pytest.mark.parametrize("module_name", dataset_module_names)
def test_dataset_loader(module_name):
    module = importlib.import_module(f"fedcast.datasets.{module_name}")
    assert hasattr(module, "load_dataset"), f"{module_name} should have a load_dataset function."
    dataset = module.load_dataset(0, num_examples=NUM_EXAMPLES)
    assert isinstance(dataset, Dataset), f"{module_name} should return a Dataset object."
    assert "x" in dataset.features, f"{module_name} should have an 'x' column."
    assert "y" in dataset.features, f"{module_name} should have a 'y' column."
    sample = dataset[0]
    window_size = getattr(module, "WINDOW_SIZE", 20)
    assert len(sample["x"]) == window_size, f"Input 'x' in {module_name} should have length {window_size}."
    assert isinstance(sample["y"], (float, np.floating, int, np.integer)), f"Target 'y' in {module_name} should be a number." 