---
layout: default
title: Examples
---

# FedCast Examples

This section provides practical examples of how to use FedCast for various time series forecasting tasks.

## Basic Examples

### 1. Simple Sinusoidal Time Series Forecasting

This example demonstrates how to use FedCast with synthetic sinusoidal data:

```python
from fedcast.datasets import SinusDataset
from fedcast.cast_models import MLP
from fedcast.federated_learning_strategies import FedAvg
from fedcast.experiments import run_federated_experiment

# Create dataset
dataset = SinusDataset(
    num_clients=5,
    sequence_length=50,
    num_samples_per_client=500
)

# Create model
model = MLP(
    input_size=50,
    hidden_size=32,
    output_size=1,
    num_layers=2
)

# Create strategy
strategy = FedAvg(
    fraction_fit=0.5,
    fraction_evaluate=0.5,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2
)

# Run experiment
results = run_federated_experiment(
    dataset=dataset,
    model=model,
    strategy=strategy,
    num_rounds=20,
    num_clients=5
)

print(f"Final accuracy: {results['final_accuracy']:.4f}")
print(f"Final loss: {results['final_loss']:.4f}")
```

### 2. ECG Signal Classification

This example shows how to work with medical time series data:

```python
from fedcast.datasets import ECGDataset
from fedcast.cast_models import MLP
from fedcast.federated_learning_strategies import FedProx

# Create ECG dataset
dataset = ECGDataset(
    partition_id=0,
    sequence_length=100
)

# Create model for ECG classification
model = MLP(
    input_size=100,
    hidden_size=64,
    output_size=2,  # Normal vs Abnormal
    num_layers=3
)

# Use FedProx for better convergence
strategy = FedProx(
    proximal_mu=0.01,
    fraction_fit=0.3,
    fraction_evaluate=0.3,
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=3
)

# Run experiment
results = run_federated_experiment(
    dataset=dataset,
    model=model,
    strategy=strategy,
    num_rounds=30,
    num_clients=10
)
```

### 3. Energy Load Forecasting

This example demonstrates energy consumption forecasting:

```python
from fedcast.datasets import EnergyLoadDataset
from fedcast.cast_models import MLP
from fedcast.federated_learning_strategies import FedTrend

# Create energy dataset
dataset = EnergyLoadDataset(
    partition_id=0,
    sequence_length=24  # 24-hour forecasting
)

# Create model
model = MLP(
    input_size=24,
    hidden_size=48,
    output_size=1,
    num_layers=3
)

# Use FedTrend for time series-specific optimization
strategy = FedTrend(
    trend_weight=0.1,
    fraction_fit=0.2,
    fraction_evaluate=0.2,
    min_fit_clients=5,
    min_evaluate_clients=5,
    min_available_clients=5
)

# Run experiment
results = run_federated_experiment(
    dataset=dataset,
    model=model,
    strategy=strategy,
    num_rounds=50,
    num_clients=20
)
```

## Advanced Examples

### 4. Communication-Efficient Federated Learning with FedLAMA

This example shows how to use FedLAMA for communication-efficient training:

```python
from fedcast.datasets import StockDataset
from fedcast.cast_models import MLP
from fedcast.federated_learning_strategies import FedLAMA

# Create stock dataset
dataset = StockDataset(
    symbol="AAPL",
    partition_id=0,
    sequence_length=30
)

# Create model
model = MLP(
    input_size=30,
    hidden_size=64,
    output_size=1,
    num_layers=4
)

# Use FedLAMA for communication efficiency
strategy = FedLAMA(
    layer_weights=[1.0, 0.8, 0.6, 0.4],  # Different weights for each layer
    fraction_fit=0.1,
    fraction_evaluate=0.1,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2
)

# Run experiment
results = run_federated_experiment(
    dataset=dataset,
    model=model,
    strategy=strategy,
    num_rounds=100,
    num_clients=50
)

print(f"Communication rounds saved: {results.get('communication_savings', 'N/A')}")
```

### 5. Heterogeneous Federated Learning with FedNova

This example demonstrates handling heterogeneous clients:

```python
from fedcast.datasets import WeatherDataset
from fedcast.cast_models import MLP
from fedcast.federated_learning_strategies import FedNova

# Create weather dataset
dataset = WeatherDataset(
    partition_id=0,
    sequence_length=24
)

# Create model
model = MLP(
    input_size=24,
    hidden_size=32,
    output_size=1,
    num_layers=2
)

# Use FedNova for heterogeneous settings
strategy = FedNova(
    fraction_fit=0.2,
    fraction_evaluate=0.2,
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=3
)

# Run experiment
results = run_federated_experiment(
    dataset=dataset,
    model=model,
    strategy=strategy,
    num_rounds=40,
    num_clients=15
)
```

### 6. Personalized Federated Learning with SCAFFOLD

This example shows how to use SCAFFOLD for personalization:

```python
from fedcast.datasets import IntelIoTDataset
from fedcast.cast_models import MLP
from fedcast.federated_learning_strategies import SCAFFOLD

# Create IoT dataset
dataset = IntelIoTDataset(
    partition_id=0,
    sequence_length=100
)

# Create model
model = MLP(
    input_size=100,
    hidden_size=64,
    output_size=1,
    num_layers=3
)

# Use SCAFFOLD for personalization
strategy = SCAFFOLD(
    fraction_fit=0.3,
    fraction_evaluate=0.3,
    min_fit_clients=4,
    min_evaluate_clients=4,
    min_available_clients=4
)

# Run experiment
results = run_federated_experiment(
    dataset=dataset,
    model=model,
    strategy=strategy,
    num_rounds=60,
    num_clients=12
)
```

## Custom Dataset Example

### 7. Creating a Custom Dataset

Here's how to create your own dataset for FedCast:

```python
import torch
from torch.utils.data import Dataset
from fedcast.datasets.base import BaseTimeSeriesDataset

class CustomTimeSeriesDataset(BaseTimeSeriesDataset):
    def __init__(self, partition_id: int, sequence_length: int = 100):
        super().__init__(partition_id, sequence_length)
        self.data = self._load_data()
        
    def _load_data(self):
        # Load your custom time series data here
        # Return a list of time series sequences
        pass
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        sequence = self.data[idx]
        # Return input sequence and target
        return sequence[:-1], sequence[-1]
```

## Experiment Tracking Example

### 8. Using MLflow for Experiment Tracking

```python
from fedcast.telemetry import MLflowLogger
from fedcast.datasets import SinusDataset
from fedcast.cast_models import MLP
from fedcast.federated_learning_strategies import FedAvg

# Initialize MLflow logger
logger = MLflowLogger(
    experiment_name="time_series_forecasting",
    tracking_uri="http://localhost:5000"
)

# Log parameters
params = {
    "dataset": "sinus",
    "model": "mlp",
    "strategy": "fedavg",
    "num_clients": 10,
    "num_rounds": 50,
    "sequence_length": 100
}
logger.log_parameters(params)

# Run experiment
dataset = SinusDataset(num_clients=10, sequence_length=100)
model = MLP(input_size=100, hidden_size=64, output_size=1)
strategy = FedAvg()

results = run_federated_experiment(
    dataset=dataset,
    model=model,
    strategy=strategy,
    num_rounds=50
)

# Log metrics
metrics = {
    "final_accuracy": results["final_accuracy"],
    "final_loss": results["final_loss"],
    "convergence_round": results["convergence_round"]
}
logger.log_metrics(metrics)

# Log artifacts (plots, models, etc.)
logger.log_artifacts("runs/experiment_001/")
```

## Grid Search Example

### 9. Running Comprehensive Experiments

```python
from fedcast.experiments.grid_all import run_grid_experiment

# Define experiment configurations
datasets = ["sinus", "ecg", "energy"]
models = ["mlp", "linear"]
strategies = ["fedavg", "fedtrend", "fedlama", "fednova"]

# Run grid search
results = run_grid_experiment(
    datasets=datasets,
    models=models,
    strategies=strategies,
    num_rounds=50,
    num_clients=10
)

# Results are automatically logged to MLflow
print("Grid search completed!")
print(f"Total experiments run: {len(results)}")
```

## Tips and Best Practices

1. **Start Simple**: Begin with basic examples and gradually move to more complex scenarios
2. **Monitor Progress**: Use MLflow to track your experiments and compare results
3. **Choose Appropriate Strategies**: Different strategies work better for different scenarios:
   - FedAvg: Good baseline for most cases
   - FedProx: Better for heterogeneous clients
   - FedTrend: Specifically designed for time series
   - FedLAMA: Communication-efficient for large-scale scenarios
   - FedNova: Handles heterogeneous settings well
   - SCAFFOLD: Good for personalization

4. **Tune Hyperparameters**: Experiment with different model architectures and strategy parameters
5. **Validate Results**: Always validate your results with multiple runs and different random seeds
