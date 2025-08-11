import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import OrderedDict
from fedcast.datasets.dataset_sinus import load_dataset, WINDOW_SIZE
from fedcast.telemetry.mlflow_logger import (
    MLflowLoggingStrategy,
    MLflowConfig,
    start_run,
    log_params,
    log_history_artifact,
)
from fedcast.cast_models import MLPModel, LinearModel
from flwr.common import Context
from fedcast.federated_learning_strategies import build_fedprox_strategy

# Define the Flower client
class SinusClient(fl.client.NumPyClient):
    def __init__(self, cid: str, model_builder):
        self.cid = int(cid)
        self.net = model_builder()
        self.trainloader = None
        self.valloader = None

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.net.train()
        
        # Load dataset for this client and set the format to torch
        dataset = load_dataset(partition_id=self.cid, num_examples=500)
        dataset.set_format("torch", columns=["x", "y"])
        trainloader = DataLoader(dataset, batch_size=32, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)

        for epoch in range(5):  # 5 epochs
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

        # Load validation dataset and set the format to torch
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
        # New Flower signature expects Context. Use node_id as cid.
        cid = str(getattr(context, "node_id", "0"))
        return SinusClient(cid=cid, model_builder=model_builder).to_client()
    return _client_fn

def run_experiment_for_model(model_name: str, model_builder) -> None:
    # Build strategy for this model run, using model-specific initial parameters
    # Default to a strategy that requires no initial parameters and works with all models
    base_strategy = build_fedprox_strategy()
    strategy = MLflowLoggingStrategy(base_strategy, dataset_name="sinus")

    strategy_name = type(base_strategy).__name__

    mlf_cfg = MLflowConfig(
        experiment_name="FedCast",
        run_name=f"basic_{strategy_name}_{model_name}",
        tags={"strategy": strategy_name, "dataset": "sinus", "model": model_name},
    )
    with start_run(mlf_cfg):
        log_params({
            "strategy": strategy_name,
            "num_rounds": 3,
            "num_clients": 2,
            "model": model_name,
        })
        history = fl.simulation.start_simulation(
            client_fn=make_client_fn(model_builder),
            num_clients=2,
            config=fl.server.ServerConfig(num_rounds=3),
            strategy=strategy,
        )
        log_history_artifact(history)


# Start the simulation
if __name__ == "__main__":
    # Run experiments for both models separately
    run_experiment_for_model("MLP", MLPModel)
    run_experiment_for_model("Linear", LinearModel)
        