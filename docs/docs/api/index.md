---
layout: default
title: API Reference
---

# API Reference

This section provides detailed documentation for the FedCast API.

## Datasets

### SinusDataset
```python
from fedcast.datasets import SinusDataset

dataset = SinusDataset(
    num_clients=10,
    sequence_length=100,
    num_samples_per_client=1000
)
```

### ECG Dataset
```python
from fedcast.datasets import ECGDataset

dataset = ECGDataset(
    partition_id=0,
    sequence_length=100
)
```

### Energy Load Dataset
```python
from fedcast.datasets import EnergyLoadDataset

dataset = EnergyLoadDataset(
    partition_id=0,
    sequence_length=24
)
```

### Stock Dataset
```python
from fedcast.datasets import StockDataset

dataset = StockDataset(
    symbol="AAPL",
    partition_id=0,
    sequence_length=30
)
```

### Weather Dataset
```python
from fedcast.datasets import WeatherDataset

dataset = WeatherDataset(
    partition_id=0,
    sequence_length=24
)
```

### Network Traffic Dataset
```python
from fedcast.datasets import NetworkTrafficDataset

dataset = NetworkTrafficDataset(
    partition_id=0,
    sequence_length=100
)
```

### Intel IoT Dataset
```python
from fedcast.datasets import IntelIoTDataset

dataset = IntelIoTDataset(
    partition_id=0,
    sequence_length=100
)
```

## Models

### MLP Model
```python
from fedcast.cast_models import MLP

model = MLP(
    input_size=100,
    hidden_size=64,
    output_size=1,
    num_layers=3
)
```

### Linear Model
```python
from fedcast.cast_models import Linear

model = Linear(
    input_size=100,
    output_size=1
)
```

## Federated Learning Strategies

### FedAvg
```python
from fedcast.federated_learning_strategies import FedAvg

strategy = FedAvg(
    fraction_fit=0.1,
    fraction_evaluate=0.1,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2
)
```

### FedProx
```python
from fedcast.federated_learning_strategies import FedProx

strategy = FedProx(
    proximal_mu=0.01,
    fraction_fit=0.1,
    fraction_evaluate=0.1,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2
)
```

### FedTrend
```python
from fedcast.federated_learning_strategies import FedTrend

strategy = FedTrend(
    trend_weight=0.1,
    fraction_fit=0.1,
    fraction_evaluate=0.1,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2
)
```

### FedLAMA
```python
from fedcast.federated_learning_strategies import FedLAMA

strategy = FedLAMA(
    layer_weights=[1.0, 0.5, 0.3],
    fraction_fit=0.1,
    fraction_evaluate=0.1,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2
)
```

### FedNova
```python
from fedcast.federated_learning_strategies import FedNova

strategy = FedNova(
    fraction_fit=0.1,
    fraction_evaluate=0.1,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2
)
```

### SCAFFOLD
```python
from fedcast.federated_learning_strategies import SCAFFOLD

strategy = SCAFFOLD(
    fraction_fit=0.1,
    fraction_evaluate=0.1,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2
)
```

### FedOpt
```python
from fedcast.federated_learning_strategies import FedOpt

strategy = FedOpt(
    beta_1=0.9,
    beta_2=0.99,
    tau=1e-3,
    fraction_fit=0.1,
    fraction_evaluate=0.1,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2
)
```

### FedDyn
```python
from fedcast.federated_learning_strategies import FedDyn

strategy = FedDyn(
    alpha=0.01,
    fraction_fit=0.1,
    fraction_evaluate=0.1,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2
)
```

### Partial Sampling
```python
from fedcast.federated_learning_strategies import PartialSampling

strategy = PartialSampling(
    sampling_fraction=0.5,
    fraction_fit=0.1,
    fraction_evaluate=0.1,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2
)
```

## Experiments

### Running Basic Experiments
```python
from fedcast.experiments import run_federated_experiment

results = run_federated_experiment(
    dataset=dataset,
    model=model,
    strategy=strategy,
    num_rounds=50,
    num_clients=10
)
```

### Grid Search Experiments
```python
from fedcast.experiments.grid_all import run_grid_experiment

results = run_grid_experiment(
    datasets=["sinus", "ecg", "energy"],
    models=["mlp", "linear"],
    strategies=["fedavg", "fedtrend", "fedlama"],
    num_rounds=50
)
```

## Telemetry

### MLflow Logger
```python
from fedcast.telemetry import MLflowLogger

logger = MLflowLogger(
    experiment_name="my_experiment",
    tracking_uri="http://localhost:5000"
)

logger.log_parameters(params)
logger.log_metrics(metrics)
logger.log_artifacts(artifacts_path)
```

## Configuration

### Model Configuration
```python
model_config = {
    "input_size": 100,
    "hidden_size": 64,
    "output_size": 1,
    "num_layers": 3,
    "activation": "relu",
    "dropout": 0.1
}
```

### Strategy Configuration
```python
strategy_config = {
    "fraction_fit": 0.1,
    "fraction_evaluate": 0.1,
    "min_fit_clients": 2,
    "min_evaluate_clients": 2,
    "min_available_clients": 2,
    "initial_parameters": None,
    "evaluate_fn": None,
    "on_fit_config_fn": None,
    "on_evaluate_config_fn": None,
    "accept_failures": True,
    "fit_metrics_aggregation_fn": None,
    "evaluate_metrics_aggregation_fn": None,
}
```

### Dataset Configuration
```python
dataset_config = {
    "num_clients": 10,
    "sequence_length": 100,
    "num_samples_per_client": 1000,
    "train_split": 0.8,
    "test_split": 0.2,
    "normalize": True,
    "shuffle": True
}
```
