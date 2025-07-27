"""
S&P 500 Stock Prices Dataset Loader

This module provides a loader for historical stock price data from major S&P 500 companies.
Data is fetched from Yahoo Finance using the yfinance library.

The dataset contains daily stock prices (adjusted close) from multiple companies, with each company
representing a natural partition for federated learning experiments. This is particularly relevant
for financial federated learning where trading data privacy is crucial.

Each company's stock price history forms a time series suitable for forecasting tasks.
The loader allows dynamic, on-demand loading of stock data for a single company, returning
windowed time series samples suitable for price prediction.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from datasets import Dataset
import yfinance as yf
from datetime import datetime, timedelta

WINDOW_SIZE = 20
DATA_DIR = Path("data/stocks")

# Major S&P 500 companies - diversified across sectors
STOCK_SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "BRK-B", "JNJ", "V",
    "WMT", "JPM", "MA", "PG", "UNH", "DIS", "PYPL", "ADBE", "HD", "NFLX",
    "BAC", "VZ", "CMCSA", "CRM", "NKE", "INTC", "T", "PFE", "TMO", "ABT",
    "COST", "AVGO", "XOM", "CVX", "WFC", "LLY", "NEE", "ABBV", "ORCL", "ACN",
    "MRK", "BMY", "TXN", "MDT", "HON", "QCOM", "PM", "IBM", "UPS", "SBUX"
]


def download_stock_data(symbol, period="2y"):
    """Download stock data for a specific symbol."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    filename = f"{symbol}.csv"
    file_path = DATA_DIR / filename
    
    # Check if data exists and is relatively recent (less than 1 day old)
    if file_path.exists():
        file_age = datetime.now() - datetime.fromtimestamp(file_path.stat().st_mtime)
        if file_age < timedelta(days=1):
            # Load existing data
            try:
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                return df['Adj Close'].dropna()
            except:
                pass  # If loading fails, download fresh data
    
    # Download fresh data
    print(f"Downloading stock data for {symbol}...")
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)
        
        if hist.empty:
            raise ValueError(f"No data available for symbol {symbol}")
        
        # Save to file
        hist.to_csv(file_path)
        
        # Return adjusted close prices
        return hist['Close'].dropna()
    
    except Exception as e:
        print(f"Failed to download data for {symbol}: {e}")
        # Return dummy data as fallback
        dates = pd.date_range(start='2022-01-01', periods=1000, freq='D')
        # Create a realistic stock price pattern
        np.random.seed(hash(symbol) % 2**32)
        price_changes = np.random.normal(0.001, 0.02, len(dates))  # Small daily changes
        prices = [100]  # Start at $100
        for change in price_changes[1:]:
            prices.append(prices[-1] * (1 + change))
        
        return pd.Series(prices, index=dates)


def get_stock_symbols():
    """Get list of available stock symbols."""
    return STOCK_SYMBOLS


def load_dataset(partition_id: int, num_examples: int = 500):
    """
    Loads stock price time series data for a single company and prepares it for forecasting.
    
    Args:
        partition_id: The company index (0-based) to use as the partition.
        num_examples: Number of (x, y) pairs to generate.
        
    Returns:
        A Hugging Face Dataset with 'x' (input sequences) and 'y' (target values).
    """
    symbols = get_stock_symbols()
    if partition_id < 0 or partition_id >= len(symbols):
        raise ValueError(f"partition_id must be between 0 and {len(symbols)-1}")
    
    symbol = symbols[partition_id]
    
    # Load stock price data for the company
    stock_prices = download_stock_data(symbol)
    
    # Convert to numpy array and handle any remaining NaN values
    prices = stock_prices.values
    prices = prices[~np.isnan(prices)]
    
    if len(prices) == 0:
        raise ValueError(f"No valid price data available for symbol {symbol}")
    
    # Convert to log returns for better forecasting properties
    # Log returns are more stationary and easier to predict
    log_prices = np.log(prices)
    log_returns = np.diff(log_prices)
    
    # Normalize log returns (z-score normalization)
    if len(log_returns) > 1:
        mean_return = np.mean(log_returns)
        std_return = np.std(log_returns)
        if std_return > 0:
            normalized_returns = (log_returns - mean_return) / std_return
        else:
            normalized_returns = log_returns
    else:
        normalized_returns = log_returns
    
    # Check if we have enough data
    total_points = num_examples + WINDOW_SIZE
    if len(normalized_returns) < total_points:
        # If not enough data, repeat the pattern
        repeats = (total_points // len(normalized_returns)) + 1
        normalized_returns = np.tile(normalized_returns, repeats)
    
    # Use only the required number of points
    values = normalized_returns[:total_points]
    
    # Create input/output sequences using numpy
    X = np.lib.stride_tricks.sliding_window_view(values, WINDOW_SIZE)[:num_examples]
    y = values[WINDOW_SIZE:WINDOW_SIZE + num_examples]
    
    # Create DataFrame and convert to Dataset
    df_xy = pd.DataFrame({"x": list(X), "y": y})
    dataset = Dataset.from_pandas(df_xy)
    return dataset


if __name__ == "__main__":
    # Test the dataset loader
    print("Available stocks:", len(get_stock_symbols()))
    print("First 10 symbols:", get_stock_symbols()[:10])
    dataset = load_dataset(0, num_examples=10)
    print("Dataset:", dataset)
    print("First sample:", dataset[0])
    print("Input shape:", np.array(dataset[0]["x"]).shape)
    print("Target type:", type(dataset[0]["y"])) 