#!/usr/bin/env python3
"""
FedCast Tutorial 2: Your First Federated Experiment

This script demonstrates a complete federated learning experiment with FedCast.
It shows how to set up clients, run federated simulation, and track results with MLflow.

Usage:
    python first_experiment.py
"""

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
    """
    Flower client for federated learning with sinus dataset.
    
    Each client trains a model on its local dataset and participates
    in the federated learning process.
    """
    
    def __init__(self, cid: str, model_builder):
        """
        Initialize client.
        
        Args:
            cid: Client identifier (determines which data partition to use)
            model_builder: Function that returns a new model instance
        """
        self.cid = int(cid)
        self.net = model_builder()

    def get_parameters(self, config):
        """Return the current model parameters as NumPy arrays."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        """Set model parameters from server update."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """
        Train the model on local data.
        
        Args:
            parameters: Model parameters from server
            config: Training configuration
            
        Returns:
            Updated parameters, number of training examples, and metrics
        """
        # Set parameters received from server
        self.set_parameters(parameters)
        self.net.train()
        
        # Load dataset for this specific client (partition determined by cid)
        dataset = load_dataset(partition_id=self.cid, num_examples=500)
        dataset.set_format("torch", columns=["x", "y"])
        trainloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)

        # Train for 5 local epochs
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
        """
        Evaluate the model on local validation data.
        
        Args:
            parameters: Model parameters from server
            config: Evaluation configuration
            
        Returns:
            Loss, number of examples, and metrics dictionary
        """
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


def make_client_fn(model_builder):
    """
    Create a client factory function for Flower simulation.
    
    Args:
        model_builder: Function that returns a new model instance
        
    Returns:
        Client factory function
    """
    def _client_fn(context: Context) -> fl.client.Client:
        # Get client ID from Flower context
        cid = str(getattr(context, "node_id", "0"))
        return SinusClient(cid=cid, model_builder=model_builder).to_client()
    return _client_fn


def run_experiment(num_rounds: int = 10, num_clients: int = 5):
    """
    Run a complete federated learning experiment.
    
    Args:
        num_rounds: Number of federated learning rounds
        num_clients: Number of clients to simulate
        
    Returns:
        Flower history object with experiment results
    """
    print(f"🚀 Starting federated learning experiment")
    print(f"   Clients: {num_clients}")
    print(f"   Rounds: {num_rounds}")
    
    # Build the base federated learning strategy (FedAvg)
    base_strategy = build_fedavg_strategy()
    
    # Wrap it with MLflow logging for automatic experiment tracking
    strategy = MLflowLoggingStrategy(base_strategy, dataset_name="sinus")

    # Configure MLflow experiment tracking
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
            "num_rounds": num_rounds,
            "num_clients": num_clients,
            "model": "MLP",
            "dataset": "sinus",
            "batch_size": 32,
            "learning_rate": 0.001,
            "local_epochs": 5,
        })
        
        print("\n📊 Running federated simulation...")
        
        # Run federated learning simulation
        history = fl.simulation.start_simulation(
            client_fn=make_client_fn(MLPModel),
            num_clients=num_clients,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
        )
        
        # Save full history as MLflow artifact
        log_history_artifact(history)
        
        # Print summary
        print("\n✅ Experiment completed successfully!")
        print(f"   Rounds executed: {len(history.losses_distributed)}")
        if history.losses_distributed:
            final_loss = history.losses_distributed[-1][1]
            print(f"   Final loss: {final_loss:.4f}")
        
        print("\n📈 View results in MLflow:")
        print("   mlflow ui --host 127.0.0.1 --port 5000")
        
        return history


if __name__ == "__main__":
    # Run the experiment with default parameters
    history = run_experiment(num_rounds=10, num_clients=5)
