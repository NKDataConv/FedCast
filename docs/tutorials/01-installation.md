---
layout: default
title: Installation & Setup
description: Get started with FedCast installation and basic environment setup
---

# Tutorial 1: Installation & Setup

Welcome to FedCast! This tutorial will guide you through installing and setting up FedCast for federated time series forecasting. By the end of this tutorial, you'll have a working FedCast environment ready for your first federated learning experiment.

## Prerequisites

Before installing FedCast, ensure you have:

- **Python 3.9+** (recommended: Python 3.12)
- **Git** for cloning the repository
- **Poetry** for dependency management (recommended) or **pip** for basic installation

## Installation Methods

FedCast offers multiple installation methods to suit different use cases:

### Method 1: Install from PyPI (Recommended for Users)

The easiest way to get started with FedCast is to install it directly from PyPI:

```bash
pip install fedcast
```

> **Note**: FedCast is currently in **Beta** (v0.1.1b1). While the core functionality is stable, some features may still be under development.

### Method 2: Development Installation with Poetry (Recommended for Contributors)

For development, research, or if you want the latest features, install from source using Poetry:

#### Step 1: Install Poetry

If you don't have Poetry installed:

```bash
# On macOS and Linux
curl -sSL https://install.python-poetry.org | python3 -

# On Windows (PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -

# Or using pip
pip install poetry
```

#### Step 2: Clone the Repository

```bash
git clone https://github.com/NKDataConv/FedCast.git
cd FedCast
```

#### Step 3: Install Dependencies

```bash
poetry install
```

This will:
- Create a virtual environment
- Install all required dependencies
- Install development dependencies (pytest, etc.)
- Set up the project in development mode

#### Step 4: Activate the Virtual Environment

```bash
poetry shell
```

### Method 3: Development Installation with pip

If you prefer using pip instead of Poetry:

```bash
git clone https://github.com/NKDataConv/FedCast.git
cd FedCast
pip install -e .
```

## Verification

Let's verify that FedCast is properly installed:

### Basic Import Test

```python
# Test basic imports
try:
    import fedcast
    from fedcast.datasets import load_sinus_dataset
    from fedcast.cast_models import MLPModel
    from fedcast.federated_learning_strategies import build_fedavg_strategy
    print("✅ FedCast imported successfully!")
    print(f"FedCast version: {fedcast.__version__}")
except ImportError as e:
    print(f"❌ Import failed: {e}")
```

### Check Dependencies

```python
# Verify key dependencies
import torch
import flwr
import pandas as pd
import mlflow

print(f"PyTorch version: {torch.__version__}")
print(f"Flower version: {flwr.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"MLflow version: {mlflow.__version__}")
```

### Run Tests

If you installed from source, run the test suite to ensure everything works:

```bash
# With Poetry
poetry run pytest

# With pip
pytest
```

## Environment Setup

### Virtual Environment Best Practices

FedCast uses Poetry for dependency management, which automatically creates and manages virtual environments. However, if you're using pip, we recommend creating a dedicated virtual environment:

```bash
# Create virtual environment
python -m venv fedcast-env

# Activate (Linux/macOS)
source fedcast-env/bin/activate

# Activate (Windows)
fedcast-env\Scripts\activate

# Install FedCast
pip install fedcast
```

### Environment Variables

FedCast uses MLflow for experiment tracking. You can configure MLflow settings:

```bash
# Optional: Set MLflow tracking URI
export MLFLOW_TRACKING_URI="file:./mlruns"

# Optional: Set experiment name
export MLFLOW_EXPERIMENT_NAME="fedcast-experiments"
```

## Quick Start Verification

Let's run a simple test to ensure everything is working:

```python
from fedcast.datasets import load_sinus_dataset
from fedcast.cast_models import MLPModel

# Load a simple dataset
dataset = load_sinus_dataset(partition_id=0, num_examples=100)
print(f"Dataset loaded: {type(dataset).__name__}")

# Create a simple model
model = MLPModel()  # Uses WINDOW_SIZE = 20 internally
print(f"Model created: {model}")

print("✅ FedCast is ready to use!")
```

## Troubleshooting

### Common Issues

#### 1. Poetry Installation Issues

If Poetry installation fails:

```bash
# Try alternative installation method
pip install --user poetry

# Or use pipx
pip install pipx
pipx install poetry
```

#### 2. PyTorch Installation Issues

If PyTorch installation fails, install it separately:

```bash
# For CPU only
pip install torch torchvision

# For CUDA (if you have NVIDIA GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### 3. Flower Installation Issues

If Flower installation fails:

```bash
pip install flwr[simulation]
```

#### 4. MLflow Issues

If MLflow UI doesn't start:

```bash
# Check if port 5000 is available
mlflow ui --host 127.0.0.1 --port 5001
```

#### 5. Ray/Flower Simulation Issues

If you encounter Ray-related errors (common on macOS):

```bash
# Ray is optional for basic usage, you can skip simulation features
# For development, you can install without simulation extras:
pip install flwr  # instead of flwr[simulation]
```

### Getting Help

If you encounter issues:

1. **Check the [GitHub Issues](https://github.com/NKDataConv/FedCast/issues)** for known problems
2. **Create a new issue** with:
   - Your operating system
   - Python version
   - Installation method used
   - Full error message
3. **Join the community** discussions

## Next Steps

Congratulations! You've successfully installed FedCast. Here's what you can do next:

### For Beginners
- **[Tutorial 2: Your First Federated Experiment](02-first-experiment.md)** - Run your first federated learning experiment

> **Coming Soon**: More tutorials are being developed covering architecture, datasets, strategies, and advanced topics. Check the [Tutorial Index](README.md) for updates!

## Additional Resources

- **[FedCast Documentation](../index.md)** - Complete framework documentation
- **[GitHub Repository](https://github.com/NKDataConv/FedCast)** - Source code and issue tracking
- **[Flower Framework](https://flower.ai/)** - Underlying federated learning framework
- **[MLflow Documentation](https://mlflow.org/docs/latest/)** - Experiment tracking guide

---

**Ready to start your federated learning journey?** Continue to [Tutorial 2: Your First Federated Experiment](02-first-experiment.md) to run your first complete federated learning experiment!
