---
layout: default
title: FedCast
description: Federated Learning for Time Series Forecasting
---

# FedCast: Federated Learning for Time Series Forecasting

<div align="center">
  <img src="https://raw.githubusercontent.com/NKDataConv/FedCast/main/assets/fedcast-logo.png" alt="FedCast Logo" width="100">
</div>

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

## Key Features

- **Federated Time Series Forecasting**: Train models on time-series data without centralizing it
- **Built on Flower**: Extends the robust and flexible Flower framework
- **Modular Architecture**: Easily customize components like data loaders, models, and aggregation strategies
- **Personalization**: Supports various strategies for building models tailored to individual clients
- **Communication Efficiency**: Advanced strategies like FedLAMA reduce communication overhead significantly
- **Comprehensive Evaluation**: Specialized metrics and visualization tools for time series forecasting
- **Experiment Tracking**: Full MLflow integration for reproducible research
- **Multiple Data Sources**: Support for synthetic, real-world, and domain-specific datasets

## Quick Start

### Installation

```bash
pip install fedcast
```

> **Note**: FedCast is currently in **Beta** (v0.1.1b1). While the core functionality is stable, some features may still be under development.

### Basic Usage

```python
from fedcast.datasets import SinusDataset
from fedcast.cast_models import MLP
from fedcast.federated_learning_strategies import FedTrend

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
```

## Architecture

FedCast is built on a modular architecture that seamlessly integrates with the Flower framework:

### Core Components

1. **Flower Integration Layer**: Direct integration with Flower's core functionality
2. **Data Management**: Time series data loading, preprocessing, and validation
3. **Model Management**: Model registry, version control, and personalization
4. **Federated Learning Strategies**: Communication-efficient algorithms like FedLAMA and FedNova
5. **Evaluation & Experimentation**: Time series metrics and MLflow integration
6. **Telemetry & Monitoring**: Comprehensive experiment tracking

## Technical Stack

- **Python 3.12+**: Core programming language
- **Flower**: Federated learning framework foundation
- **PyTorch**: Deep learning model implementation
- **Pandas/NumPy**: Data manipulation and numerical computing
- **MLflow**: Experiment tracking and model management
- **Poetry**: Dependency management and packaging

## Getting Started

### Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/NKDataConv/FedCast.git
   cd FedCast
   ```

2. **Install dependencies:**
   ```bash
   poetry install
   ```

3. **Run tests:**
   ```bash
   poetry run pytest
   ```

### Running Experiments

```bash
# Basic experiments
poetry run python fedcast/experiments/basic_fedavg.py
poetry run python fedcast/experiments/basic_fedtrend.py

# Grid search experiments
poetry run python fedcast/experiments/grid_all.py
```

### Monitoring Results

View experiment results with MLflow:
```bash
mlflow ui --host 127.0.0.1 --port 5000
```

## Documentation

- [GitHub Repository](https://github.com/NKDataConv/FedCast)
- [PyPI Package](https://pypi.org/project/fedcast/)
- [Flower Framework](https://flower.ai/)

## Supporters

This project is supported by the Bundesministerium für Forschung, Technologie und Raumfahrt (BMFTR).

<div align="center">
  <img src="https://raw.githubusercontent.com/NKDataConv/FedCast/main/assets/logo_bmftr.jpg" alt="BMFTR Logo" width="250">
</div>

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](https://github.com/NKDataConv/FedCast/blob/main/LICENSE) file for details.
