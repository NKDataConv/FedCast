---
layout: default
title: Your First Federated Experiment
description: Hands-on introduction to running a complete federated learning experiment with FedCast
---

# Tutorial 2: Your First Federated Experiment

Welcome to your first hands-on tutorial with FedCast! In this tutorial, you'll learn how to set up and run a complete federated learning experiment from scratch. By the end, you'll understand the core workflow of federated learning with FedCast and see how results are tracked with MLflow.

## Prerequisites

Before starting this tutorial, make sure you have:

- ✅ Completed [Tutorial 1: Installation & Setup](01-installation.md)
- ✅ FedCast installed and verified
- ✅ Basic understanding of Python and machine learning concepts
- ✅ Familiarity with PyTorch (helpful but not required)

## Overview

In this tutorial, you'll learn to:

1. **Set up a Flower client** for federated learning
2. **Load and partition time series data** across multiple clients
3. **Create a neural network model** for time series forecasting
4. **Choose a federated learning strategy** (FedAvg)
5. **Run a federated simulation** with multiple clients
6. **Track results** with MLflow
7. **Understand and interpret** the experiment results

## Understanding the Components

Before we dive into code, let's understand the key components:

### Flower Client

A **Flower client** represents a single participant in the federated learning process. Each client:
- Has its own local dataset (private, never shared)
- Trains a model on its local data
- Sends model updates (not data) to the server
- Receives aggregated updates from the server

### Federated Learning Strategy

A **strategy** defines how the server aggregates updates from clients. FedAvg (Federated Averaging) is the most common approach, simply averaging all client updates.

### MLflow Integration

**MLflow** automatically tracks:
- Experiment parameters (strategy, model, dataset, number of rounds)
- Metrics per round (loss, accuracy)
- Client-specific metrics
- Full experiment history

## Step-by-Step Tutorial

### Step 1: Import Required Libraries

Let's start by importing all the necessary components:

```python
import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import OrderedDict

# FedCast imports
from fedcast.datasets.dataset_sinus import load_dataset, WINDOW_SIZE
from fedcast.cast_models import MLPModel
from fedcast.federated_learning_strategies import build_fedavg_strategy
from fedcast.telemetry.mlflow_logger import (
    MLflowLoggingStrategy,
    MLflowConfig,
    start_run,
    log_params,
    log_history_artifact,
)
from flwr.common import Context
```

### Step 2: Create a Flower Client

The client handles local training and evaluation. Here's a complete client implementation:

```python
class SinusClient(fl.client.NumPyClient):
    def __init__(self, cid: str, model_builder):
        self.cid = int(cid)
        self.net = model_builder()
        self.trainloader = None
        self.valloader = None

    def get_parameters(self, config):
        """Return the current model parameters."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        """Set model parameters from server."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Train the model on local data."""
        # Set parameters received from server
        self.set_parameters(parameters)
        self.net.train()
        
        # Load dataset for this specific client
        dataset = load_dataset(partition_id=self.cid, num_examples=500)
        dataset.set_format("torch", columns=["x", "y"])
        trainloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)

        # Train for 5 epochs
        for epoch in range(5):
            for batch in trainloader:
                inputs, labels = batch["x"].float(), batch["y"].float().view(-1, 1)
                optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Return updated parameters and metadata
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        """Evaluate the model on local validation data."""
        self.set_parameters(parameters)
        self.net.eval()

        # Load validation dataset
        dataset = load_dataset(partition_id=self.cid, num_examples=100)
        dataset.set_format("torch", columns=["x", "y"])
        valloader = DataLoader(dataset, batch_size=32)

        criterion = nn.MSELoss()
        loss = 0
        
        with torch.no_grad():
            for batch in valloader:
                inputs, labels = batch["x"].float(), batch["y"].float().view(-1, 1)
                outputs = self.net(inputs)
                loss += criterion(outputs, labels).item()
        
        return loss / len(valloader), len(valloader.dataset), {"mse": loss}
```

### Step 3: Create a Client Factory Function

Flower needs a function to create clients. This function will be called for each client in the simulation:

```python
def make_client_fn(model_builder):
    def _client_fn(context: Context) -> fl.client.Client:
        # Get client ID from context
        cid = str(getattr(context, "node_id", "0"))
        return SinusClient(cid=cid, model_builder=model_builder).to_client()
    return _client_fn
```

### Step 4: Set Up MLflow and Run the Experiment

Now let's put it all together and run the experiment:

```python
def run_experiment():
    # Build the base federated learning strategy
    base_strategy = build_fedavg_strategy()
    
    # Wrap it with MLflow logging
    strategy = MLflowLoggingStrategy(base_strategy, dataset_name="sinus")

    # Configure MLflow
    mlf_cfg = MLflowConfig(
        experiment_name="FedCast-Tutorial",
        run_name="first_federated_experiment",
        tags={
            "strategy": "FedAvg",
            "dataset": "sinus",
            "model": "MLP",
            "tutorial": "first-experiment"
        },
    )
    
    # Start MLflow run and execute simulation
    with start_run(mlf_cfg):
        # Log experiment parameters
        log_params({
            "strategy": "FedAvg",
            "num_rounds": 10,
            "num_clients": 5,
            "model": "MLP",
            "dataset": "sinus",
            "batch_size": 32,
            "learning_rate": 0.001,
            "local_epochs": 5,
        })
        
        # Run federated learning simulation
        history = fl.simulation.start_simulation(
            client_fn=make_client_fn(MLPModel),
            num_clients=5,
            config=fl.server.ServerConfig(num_rounds=10),
            strategy=strategy,
        )
        
        # Save full history as artifact
        log_history_artifact(history)
        
        print("✅ Experiment completed successfully!")
        print(f"   Rounds: {len(history.losses_distributed)}")
        print(f"   Final loss: {history.losses_distributed[-1][1]:.4f}")
        
        return history

# Run the experiment
if __name__ == "__main__":
    history = run_experiment()
```

### Step 5: Understanding the Results

After running the experiment, you'll see output like this:

```
✅ Experiment completed successfully!
   Rounds: 10
   Final loss: 0.0234
```

#### What Just Happened?

1. **5 clients** were created, each with their own unique sinus dataset
2. **10 federated rounds** were executed:
   - Each round: clients train locally → send updates → server aggregates → distributes global model
3. **Training happened locally** - no data left the clients
4. **Metrics were logged** to MLflow automatically

#### Accessing the Results

The `history` object contains detailed information:

```python
# Distributed training loss (average across clients)
print("Training losses:", history.losses_distributed)

# Evaluation metrics
print("Evaluation metrics:", history.metrics_distributed)

# Per-round information
for round_num, (round_idx, loss) in enumerate(history.losses_distributed):
    print(f"Round {round_idx}: Loss = {loss:.4f}")
```

## Viewing Results in MLflow

### Start MLflow UI

After running the experiment, view the results in MLflow:

```bash
mlflow ui --host 127.0.0.1 --port 5000
```

Then open your browser to `http://127.0.0.1:5000`

### What You'll See in MLflow

1. **Experiment Dashboard**: Overview of all runs
2. **Parameters**: Strategy, model, dataset, hyperparameters
3. **Metrics**: Loss per round, per-client metrics
4. **Artifacts**: Full experiment history JSON
5. **Tags**: Strategy type, dataset, model type

### Key Metrics to Observe

- **Loss Over Rounds**: Should generally decrease as the model improves
- **Per-Client Loss**: Shows how different clients perform
- **Training vs Evaluation**: Compare fit and eval metrics

## Complete Example Script

Here's a complete, runnable script that combines all the steps:

```python
import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import OrderedDict

from fedcast.datasets.dataset_sinus import load_dataset
from fedcast.cast_models import MLPModel
from fedcast.federated_learning_strategies import build_fedavg_strategy
from fedcast.telemetry.mlflow_logger import (
    MLflowLoggingStrategy,
    MLflowConfig,
    start_run,
    log_params,
    log_history_artifact,
)
from flwr.common import Context


class SinusClient(fl.client.NumPyClient):
    def __init__(self, cid: str, model_builder):
        self.cid = int(cid)
        self.net = model_builder()

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.net.train()
        
        dataset = load_dataset(partition_id=self.cid, num_examples=500)
        dataset.set_format("torch", columns=["x", "y"])
        trainloader = DataLoader(dataset, batch_size=32, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)

        for epoch in range(5):
            for batch in trainloader:
                inputs, labels = batch["x"].float(), batch["y"].float().view(-1, 1)
                optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.net.eval()

        dataset = load_dataset(partition_id=self.cid, num_examples=100)
        dataset.set_format("torch", columns=["x", "y"])
        valloader = DataLoader(dataset, batch_size=32)

        criterion = nn.MSELoss()
        loss = 0
        
        with torch.no_grad():
            for batch in valloader:
                inputs, labels = batch["x"].float(), batch["y"].float().view(-1, 1)
                outputs = self.net(inputs)
                loss += criterion(outputs, labels).item()
        
        return loss / len(valloader), len(valloader.dataset), {"mse": loss}


def make_client_fn(model_builder):
    def _client_fn(context: Context) -> fl.client.Client:
        cid = str(getattr(context, "node_id", "0"))
        return SinusClient(cid=cid, model_builder=model_builder).to_client()
    return _client_fn


def run_experiment():
    base_strategy = build_fedavg_strategy()
    strategy = MLflowLoggingStrategy(base_strategy, dataset_name="sinus")

    mlf_cfg = MLflowConfig(
        experiment_name="FedCast-Tutorial",
        run_name="first_federated_experiment",
        tags={"strategy": "FedAvg", "dataset": "sinus", "model": "MLP"},
    )
    
    with start_run(mlf_cfg):
        log_params({
            "strategy": "FedAvg",
            "num_rounds": 10,
            "num_clients": 5,
            "model": "MLP",
            "dataset": "sinus",
        })
        
        history = fl.simulation.start_simulation(
            client_fn=make_client_fn(MLPModel),
            num_clients=5,
            config=fl.server.ServerConfig(num_rounds=10),
            strategy=strategy,
        )
        
        log_history_artifact(history)
        
        print("✅ Experiment completed!")
        print(f"   Rounds: {len(history.losses_distributed)}")
        if history.losses_distributed:
            print(f"   Final loss: {history.losses_distributed[-1][1]:.4f}")
        
        return history


if __name__ == "__main__":
    history = run_experiment()
```

### Running the Script

Save the script as `first_experiment.py` and run:

```bash
# With Poetry
poetry run python first_experiment.py

# With pip/venv
python first_experiment.py
```

## Experiment Variations

Now that you understand the basics, try experimenting with different configurations:

### More Clients

```python
num_clients=10  # Increase from 5 to 10
```

### More Rounds

```python
config=fl.server.ServerConfig(num_rounds=20)  # More training rounds
```

### Different Strategies

```python
from fedcast.federated_learning_strategies import build_fedprox_strategy

base_strategy = build_fedprox_strategy(mu=0.01)  # FedProx with regularization
```

### Different Models

```python
from fedcast.cast_models import LinearModel

client_fn=make_client_fn(LinearModel)  # Try linear model instead of MLP
```

## Understanding Federated Learning Workflow

Let's visualize what happens in each round:

```
Round 1:
  Client 1: Local Training → Update₁
  Client 2: Local Training → Update₂
  Client 3: Local Training → Update₃
  Client 4: Local Training → Update₄
  Client 5: Local Training → Update₅
         ↓
  Server: Aggregate (Average) → Global Model₁
         ↓
  Clients: Receive Global Model₁

Round 2:
  Clients: Train on Global Model₁
         ↓
  Server: Aggregate → Global Model₂
         ↓
  (repeat...)
```

Each round improves the global model without sharing raw data!

## Troubleshooting

### Common Issues

#### 1. Ray/Simulation Errors

If you encounter Ray-related errors (especially on macOS):

```python
# Try with fewer clients
num_clients=2

# Or check if simulation is properly configured
# Some systems may need additional Ray setup
```

#### 2. Memory Issues

If running out of memory:

```python
# Reduce batch size
batch_size=16  # Instead of 32

# Reduce number of examples per client
num_examples=200  # Instead of 500
```

#### 3. MLflow Not Starting

If MLflow UI doesn't start:

```bash
# Try different port
mlflow ui --host 127.0.0.1 --port 5001

# Check if port is already in use
lsof -i :5000
```

#### 4. Import Errors

Make sure all FedCast components are installed:

```python
# Verify imports
from fedcast.datasets import load_sinus_dataset
from fedcast.cast_models import MLPModel
from fedcast.federated_learning_strategies import build_fedavg_strategy
```

## Key Takeaways

✅ **Federated Learning Basics**: Clients train locally, only share model updates  
✅ **Flower Integration**: FedCast builds on Flower's robust framework  
✅ **MLflow Tracking**: Automatic experiment tracking and reproducibility  
✅ **Privacy Preservation**: Data never leaves the client  
✅ **Model Convergence**: Global model improves over federated rounds  

## Next Steps

Congratulations! You've completed your first federated learning experiment. Here's what you can explore next:

### Recommended Next Steps

> **Coming Soon**: More tutorials are being developed! Future tutorials will cover:
> - Understanding FedCast Architecture
> - Working with Different Datasets (ECG, stocks, weather, IoT)
> - Choosing Aggregation Strategies (FedAvg, FedProx, FedTrend, and more)
> 
> Check the [Tutorial Index](README.md) for updates as new tutorials are published.

### Experiment Ideas

1. **Compare Strategies**: Run the same experiment with different strategies
2. **Vary Client Count**: See how performance changes with more/fewer clients
3. **Try Different Models**: Compare MLP vs Linear models
4. **Adjust Hyperparameters**: Experiment with learning rates, batch sizes
5. **Extend Rounds**: See how many rounds are needed for convergence

### Additional Resources

- **[FedCast Documentation](../index.md)** - Complete framework documentation
- **[Flower Framework Docs](https://flower.ai/docs/)** - Underlying framework details
- **[MLflow Documentation](https://mlflow.org/docs/latest/)** - Advanced tracking features

---

**Ready to explore more?** Check the [Tutorial Index](README.md) for updates on new tutorials as they become available!
