# FedCast: Federated Learning for Time Series Forecasting

<p align="center">
  <img src="https://raw.githubusercontent.com/NKDataConv/FedCast/main/assets/fedcast-logo.png" alt="FedCast Logo" width="100">
</p>

FedCast is a comprehensive Python framework designed for time series forecasting using federated learning. It leverages the powerful [Flower (flwr)](https://flower.ai/) framework to enable privacy-preserving, decentralized model training on distributed time series data.

## Project Overview

The core goal of FedCast is to provide a modular, extensible, and easy-to-use platform for researchers and practitioners to develop and evaluate personalized federated learning strategies for time series analysis. The framework addresses the unique challenges of time series forecasting in federated settings, where data privacy, communication efficiency, and model personalization are critical concerns.

### Problem Statement

Traditional centralized approaches to time series forecasting require all data to be collected at a central location, which poses significant challenges:
- **Privacy Concerns**: Sensitive time series data (medical, financial, IoT) cannot be shared
- **Communication Overhead**: Large-scale time series data is expensive to transmit
- **Heterogeneity**: Different clients may have varying data distributions and patterns
- **Personalization**: Global models may not perform well for individual client patterns

FedCast addresses these challenges through federated learning, enabling collaborative model training while keeping data distributed and private.

## Architecture

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

### Design Principles

- **Modularity**: Clear separation of concerns with independent, replaceable components
- **Extensibility**: Plugin architecture for easy integration of new algorithms and data sources
- **Privacy-First**: Built-in privacy preservation mechanisms
- **Performance**: Optimized for communication efficiency and computational speed
- **Reproducibility**: Comprehensive logging and experiment tracking

## Key Features

- **Federated Time Series Forecasting**: Train models on time-series data without centralizing it
- **Built on Flower**: Extends the robust and flexible Flower framework
- **Modular Architecture**: Easily customize components like data loaders, models, and aggregation strategies
- **Personalization**: Supports various strategies for building models tailored to individual clients
- **Communication Efficiency**: Advanced strategies like FedLAMA reduce communication overhead significantly
- **Comprehensive Evaluation**: Specialized metrics and visualization tools for time series forecasting
- **Experiment Tracking**: Full MLflow integration for reproducible research
- **Multiple Data Sources**: Support for synthetic, real-world, and domain-specific datasets

## Technical Stack

- **Python 3.12+**: Core programming language
- **Flower**: Federated learning framework foundation
- **PyTorch**: Deep learning model implementation
- **Pandas/NumPy**: Data manipulation and numerical computing
- **MLflow**: Experiment tracking and model management
- **Poetry**: Dependency management and packaging
- **Pytest**: Testing framework

## Quick Start Example

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

## Getting Started

### Installation

#### Option 1: Install from PyPI (Recommended)
```bash
pip install fedcast
```

> **Note**: FedCast is currently in **Beta** (v0.1.1b1). While the core functionality is stable, some features may still be under development. We welcome feedback and contributions!

#### Option 2: Install from source
1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd FedCast
    ```

2.  **Install dependencies:**
    This project uses [Poetry](https://python-poetry.org/) for dependency management and packaging.
    ```bash
    poetry install
    ```

    Or install directly with pip:
    ```bash
    pip install -e .
    ```

## Quick Start

After installation, you can start using FedCast:

```python
import fedcast
from fedcast.datasets import load_sinus_dataset
from fedcast.cast_models import MLPModel
from fedcast.federated_learning_strategies import build_fedavg_strategy

# Create a dataset
dataset = load_sinus_dataset(partition_id=0)

# Create a model
model = MLPModel()

# Create a federated learning strategy
strategy = build_fedavg_strategy()

# Your federated learning experiment here...
```

## Development

### Running Tests

To ensure the reliability and correctness of the framework, we use `pytest` for testing.

To run the full test suite, execute the following command from the root of the project:

```bash
poetry run pytest
```

This will automatically discover and run all tests located in the `tests/` directory.


### Running Experiments

FedCast provides several ways to run federated learning experiments:

#### 1. Basic Experiments
Run individual experiments with specific configurations:
```bash
# FedAvg experiment
poetry run python fedcast/experiments/basic_fedavg.py

# FedTrend experiment
poetry run python fedcast/experiments/basic_fedtrend.py
```

#### 2. Grid Search Experiments
Run comprehensive experiments across multiple configurations:
```bash
# Run all combinations of datasets, models, and strategies
poetry run python fedcast/experiments/grid_all.py
```

#### 3. Custom Experiments
Create your own experiment scripts by importing FedCast components:
```python
from fedcast.datasets import YourDataset
from fedcast.cast_models import YourModel
from fedcast.federated_learning_strategies import YourStrategy

# Implement your custom experiment logic
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


## Supporters

This project is supported by the Bundesministerium für Forschung, Technologie und Raumfahrt (BMFTR). We are grateful for their support, without which this project would not be possible.

<img src="https://raw.githubusercontent.com/NKDataConv/FedCast/main/assets/logo_bmftr.jpg" alt="BMFTR Logo" width=250>
