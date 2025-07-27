import pytest
import numpy as np
from datasets import Dataset
from fedcast.datasets.dataset_stocks import load_dataset, get_stock_symbols, WINDOW_SIZE, STOCK_SYMBOLS
from pathlib import Path
import os

# Test cases for partition IDs
PARTITION_ID_1 = 0  # AAPL
PARTITION_ID_2 = 1  # MSFT
NUM_EXAMPLES = 10


def test_get_stock_symbols():
    """Tests that get_stock_symbols returns the expected stock symbols."""
    stock_symbols = get_stock_symbols()
    assert isinstance(stock_symbols, list), "Stock symbols should be returned as a list."
    assert len(stock_symbols) == len(STOCK_SYMBOLS), f"Should return {len(STOCK_SYMBOLS)} stock symbols."
    assert stock_symbols == STOCK_SYMBOLS, "Stock symbols should match STOCK_SYMBOLS."


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
    """Tests that the stock data is not all zeros."""
    dataset = load_dataset(PARTITION_ID_1, num_examples=NUM_EXAMPLES)
    
    # At least one value in the first sample should be nonzero for real stock data
    x_data = np.array(dataset[0]["x"])
    y_data = dataset[0]["y"]
    
    assert np.any(x_data != 0) or y_data != 0, "Stock data should not be all zeros."
    assert not np.all(np.isnan(x_data)), "Stock data should not contain NaN values."
    assert not np.isnan(y_data), "Target value should not be NaN."


def test_load_dataset_different_stocks():
    """Tests that different stocks produce different data."""
    dataset1 = load_dataset(PARTITION_ID_1, num_examples=NUM_EXAMPLES)
    dataset2 = load_dataset(PARTITION_ID_2, num_examples=NUM_EXAMPLES)
    
    # Different stocks should have different price patterns
    x1 = np.array(dataset1[0]["x"])
    x2 = np.array(dataset2[0]["x"])
    
    # At least some values should be different between stocks
    assert not np.allclose(x1, x2), "Different stocks should have different return patterns."


def test_load_dataset_returns_normalization():
    """Tests that stock returns appear to be normalized and have reasonable range."""
    dataset = load_dataset(PARTITION_ID_1, num_examples=NUM_EXAMPLES)
    
    # Collect all data points to check normalization properties
    all_x_data = []
    all_y_data = []
    
    for i in range(NUM_EXAMPLES):
        all_x_data.extend(dataset[i]["x"])
        all_y_data.append(dataset[i]["y"])
    
    x_array = np.array(all_x_data)
    y_array = np.array(all_y_data)
    
    # Check that data has reasonable range for normalized stock returns
    # Stock returns after normalization should be roughly within [-5, 5] range
    assert np.all(x_array > -10) and np.all(x_array < 10), "Normalized stock returns should be within reasonable range."
    assert np.all(y_array > -10) and np.all(y_array < 10), "Normalized stock return targets should be within reasonable range."
    
    # Check that the data has some variance (not constant)
    x_std = np.std(x_array)
    assert x_std > 0.01, f"Stock return data should have some variance, got std={x_std}"


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
    num_stocks = len(get_stock_symbols())
    
    # Test negative partition ID
    with pytest.raises(ValueError, match="partition_id must be between"):
        load_dataset(-1, num_examples=NUM_EXAMPLES)
    
    # Test partition ID that's too large
    with pytest.raises(ValueError, match="partition_id must be between"):
        load_dataset(num_stocks, num_examples=NUM_EXAMPLES)


def test_load_dataset_varying_num_examples():
    """Tests that the dataset can handle different numbers of examples."""
    small_dataset = load_dataset(PARTITION_ID_1, num_examples=5)
    large_dataset = load_dataset(PARTITION_ID_1, num_examples=50)
    
    assert len(small_dataset) == 5, "Small dataset should have 5 examples."
    assert len(large_dataset) == 50, "Large dataset should have 50 examples."
    
    # First examples should be identical
    assert np.allclose(small_dataset[0]["x"], large_dataset[0]["x"]), "First samples should be identical."
    assert np.isclose(small_dataset[0]["y"], large_dataset[0]["y"]), "First targets should be identical."


def test_load_dataset_financial_properties():
    """Tests that stock data exhibits typical financial time series properties."""
    dataset = load_dataset(PARTITION_ID_1, num_examples=NUM_EXAMPLES)
    
    # Check that returns don't have extreme values (typical for log returns)
    for i in range(NUM_EXAMPLES):
        x_data = np.array(dataset[i]["x"])
        y_data = dataset[i]["y"]
        
        # Log returns typically stay within [-5, 5] after normalization
        assert np.all(np.abs(x_data) < 20), f"Sample {i} has extreme values in stock returns."
        assert np.abs(y_data) < 20, f"Sample {i} target has extreme value in stock returns."


def test_load_dataset_returns_stationarity():
    """Tests that stock returns show properties of stationarity (mean-reverting)."""
    dataset = load_dataset(PARTITION_ID_1, num_examples=NUM_EXAMPLES)
    
    # Collect several windows of data
    all_data = []
    for i in range(NUM_EXAMPLES):
        all_data.extend(dataset[i]["x"])
        all_data.append(dataset[i]["y"])
    
    returns = np.array(all_data)
    
    # Check that the mean is close to zero (characteristic of stationary returns)
    mean_return = np.mean(returns)
    assert abs(mean_return) < 2.0, f"Mean of normalized returns should be close to 0, got {mean_return}"


def test_multiple_stocks_coverage():
    """Tests that we can load data from multiple different stocks."""
    num_stocks = min(5, len(get_stock_symbols()))  # Test first 5 stocks
    datasets = []
    
    for stock_id in range(num_stocks):
        dataset = load_dataset(stock_id, num_examples=5)
        datasets.append(dataset)
        
        # Each dataset should be valid
        assert len(dataset) == 5, f"Stock {stock_id} dataset should have 5 examples."
        assert "x" in dataset.features and "y" in dataset.features, f"Stock {stock_id} dataset should have x and y columns."
    
    # Verify that stocks have different data patterns
    for i in range(1, num_stocks):
        x0 = np.array(datasets[0][0]["x"])
        xi = np.array(datasets[i][0]["x"])
        assert not np.allclose(x0, xi, rtol=1e-3), f"Stock 0 and stock {i} should have different return patterns."


def test_stock_symbols_validity():
    """Tests that stock symbols are valid format."""
    symbols = get_stock_symbols()
    
    for symbol in symbols[:10]:  # Test first 10
        # Stock symbols should be strings
        assert isinstance(symbol, str), f"Symbol {symbol} should be a string."
        # Should be uppercase and reasonable length
        assert symbol.isupper(), f"Symbol {symbol} should be uppercase."
        assert 1 <= len(symbol) <= 10, f"Symbol {symbol} should be between 1-10 characters."


def test_data_caching():
    """Tests that data caching works correctly."""
    # Load data twice for the same stock
    dataset1 = load_dataset(PARTITION_ID_1, num_examples=5)
    dataset2 = load_dataset(PARTITION_ID_1, num_examples=5)
    
    # Results should be identical (cached data)
    for i in range(5):
        assert np.allclose(dataset1[i]["x"], dataset2[i]["x"]), f"Cached data should be identical for sample {i}."
        assert np.isclose(dataset1[i]["y"], dataset2[i]["y"]), f"Cached targets should be identical for sample {i}." 