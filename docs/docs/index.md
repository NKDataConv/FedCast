---
layout: default
title: Documentation
---

# FedCast Documentation

Welcome to the FedCast documentation! This comprehensive guide will help you get started with federated learning for time series forecasting.

## Getting Started

### Installation

#### Option 1: Install from PyPI (Recommended)
```bash
pip install fedcast
```

> **Note**: FedCast is currently in **Beta** (v0.1.1b1). While the core functionality is stable, some features may still be under development. We welcome feedback and contributions!

#### Option 2: Install from source
1. **Clone the repository:**
   ```bash
   git clone https://github.com/NKDataConv/FedCast.git
   cd FedCast
   ```

2. **Install dependencies:**
   This project uses [Poetry](https://python-poetry.org/) for dependency management and packaging.
   ```bash
   poetry install
   ```

   Or install directly with pip:
   ```bash
   pip install -e .
   ```

### Quick Start Example

```python
from fedcast.datasets import SinusDataset
from fedcast.cast_models import MLP
from fedcast.federated_learning_strategies import FedTrend
from fedcast.experiments import run_federated_experiment

# Load time series data
dataset = SinusDataset(num_clients=10, sequence_length=100)

# Define model architecture
model = MLP(input_size=100, hidden_size=64, output_size=1)

# Choose federated learning strategy
strategy = FedTrend()

# Run federated learning experiment
results = run_federated_experiment(
    dataset=dataset,
    model=model,
    strategy=strategy,
    num_rounds=50
)

# Results are automatically logged to MLflow
print(f"Final accuracy: {results['final_accuracy']}")
```

## Architecture Overview

FedCast is built on a modular architecture that seamlessly integrates with the Flower framework while providing specialized components for time series forecasting:

### Core Components

#### 1. **Flower Integration Layer**
- Direct integration with Flower's core functionality
- Custom client and server implementations
- Support for both synchronous and asynchronous federated learning
- Preservation of all Flower features and capabilities

#### 2. **Data Management**
- **Time Series Datasets**: Support for multiple data types (synthetic, energy, medical, financial, IoT, network, weather)
- **Data Validation**: Automatic data cleaning and validation
- **Transformation Pipelines**: Flexible data preprocessing
- **Heterogeneous Data Handling**: Support for varying data distributions across clients
- **Automatic Downloading**: Built-in data source connectors with caching

#### 3. **Model Management**
- **Model Registry**: Centralized model factory system
- **Version Control**: Model serialization and deserialization
- **Adaptation**: Model personalization and fine-tuning
- **Architecture Support**: MLP, Linear models, and extensible framework for custom models

#### 4. **Federated Learning Strategies**
- **Communication-Efficient Algorithms**: FedLAMA reduces communication overhead by up to 70%
- **Robust Aggregation**: FedNova addresses objective inconsistency in heterogeneous settings
- **Personalization**: FedTrend and other specialized strategies for time series
- **Standard Algorithms**: FedAvg, FedProx, FedOpt, SCAFFOLD, and more

#### 5. **Evaluation & Experimentation**
- **Time Series Metrics**: Specialized evaluation metrics for forecasting tasks
- **MLflow Integration**: Comprehensive experiment tracking and logging
- **Visualization**: Automatic plotting of training progress and results
- **Grid Experiments**: Automated testing across multiple configurations

#### 6. **Telemetry & Monitoring**
- **MLflow Logger**: Centralized experiment tracking
- **Performance Monitoring**: Real-time training metrics
- **Result Analysis**: Comparative analysis tools

## Running Experiments

### Basic Experiments
Run individual experiments with specific configurations:
```bash
# FedAvg experiment
poetry run python fedcast/experiments/basic_fedavg.py

# FedTrend experiment
poetry run python fedcast/experiments/basic_fedtrend.py
```

### Grid Search Experiments
Run comprehensive experiments across multiple configurations:
```bash
# Run all combinations of datasets, models, and strategies
poetry run python fedcast/experiments/grid_all.py
```

## Monitoring and Visualization

### MLflow UI
View experiment results, compare runs, and analyze performance:
```bash
mlflow ui --host 127.0.0.1 --port 5000
```

Access the UI at `http://127.0.0.1:5000` to:
- Track experiment parameters and metrics
- Compare different federated learning strategies
- Visualize training progress and convergence
- Download model artifacts and results

### Automatic Plotting
FedCast automatically generates plots for:
- Training and validation losses per round
- Client-specific performance metrics
- Communication efficiency comparisons
- Model convergence analysis

Plots are saved in `runs/<experiment_name>/` directory.

## Development

### Running Tests

To ensure the reliability and correctness of the framework, we use `pytest` for testing.

To run the full test suite, execute the following command from the root of the project:

```bash
poetry run pytest
```

This will automatically discover and run all tests located in the `tests/` directory.

## Contributing

We welcome contributions to FedCast! Please see our [Contributing Guide](CONTRIBUTING.md) for details on how to get started.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Support

If you encounter any issues or have questions, please:
1. Check the [GitHub Issues](https://github.com/NKDataConv/FedCast/issues)
2. Create a new issue with detailed information about your problem
3. Join our community discussions
