import hashlib
import numpy as np
import pandas as pd
from datasets import Dataset

WINDOW_SIZE = 20
"""The number of time steps to use as input features for the model."""


def get_sine_params(partition_id: int):
    """
    Generates deterministic parameters for a sine wave based on the partition ID.

    This function uses the partition ID to seed a random number generator, ensuring
    that the same ID always produces the same set of parameters. This is crucial
    for creating reproducible, unique datasets for each client in a federated
    learning setup.

    Args:
        partition_id: An integer identifying the data partition.

    Returns:
        A tuple containing the amplitude, frequency, phase, and vertical shift
        for a sine wave.
    """
    # Use a seed based on the partition_id for reproducibility.
    # SHA256 is used to create a uniform and deterministic seed.
    seed = int(hashlib.sha256(str(partition_id).encode("utf-8")).hexdigest(), 16) % (
        2**32
    )
    rng = np.random.default_rng(seed)

    amplitude = rng.uniform(0.5, 5.0)
    frequency = rng.uniform(0.05, 0.2)
    phase = rng.uniform(0, 2 * np.pi)
    vertical_shift = rng.uniform(-2.0, 2.0)

    return amplitude, frequency, phase, vertical_shift


def load_dataset(partition_id: int, num_examples: int = 500):
    """
    Generates a synthetic time series dataset from a sine wave for a given partition.

    The function creates a time series prediction problem where the goal is to
    predict the next value in the series based on a window of preceding values.
    The characteristics of the sine wave are unique to each partition_id.

    Args:
        partition_id: The identifier for the data partition.
        num_examples: The number of training/testing examples to generate.

    Returns:
        A `datasets.Dataset` object containing 'x' (input sequences) and
        'y' (target values).
    """
    # 1. Get deterministic parameters for the sine wave.
    amplitude, frequency, phase, vertical_shift = get_sine_params(partition_id)

    # 2. Generate the sine wave.
    # Total number of points needed is num_examples + the window size for the last example.
    total_points = num_examples + WINDOW_SIZE
    x_time = np.arange(total_points)
    time_series = amplitude * np.sin(frequency * x_time + phase) + vertical_shift

    # 3. Create input/output sequences for time series prediction.
    X, y = [], []
    for i in range(num_examples):
        X.append(time_series[i : i + WINDOW_SIZE])
        y.append(time_series[i + WINDOW_SIZE])

    # 4. Create a pandas DataFrame.
    df = pd.DataFrame({"x": X, "y": y})

    # 5. Convert to a Hugging Face Dataset.
    dataset = Dataset.from_pandas(df)
    return dataset


if __name__ == "__main__":
    dataset = load_dataset(0)
    print(dataset)
    print(dataset[0])