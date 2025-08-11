import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import OrderedDict
from fedcast.datasets.dataset_sinus import load_dataset, WINDOW_SIZE
from fedcast.telemetry.mlflow_logger import (
    MLflowConfig,
    start_run,
    log_params,
    log_history_artifact,
    MLflowLoggingStrategy,
)

# Define the neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(WINDOW_SIZE, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the Flower client
class SinusClient(fl.client.NumPyClient):
    def __init__(self, cid: str):
        self.cid = int(cid)
        self.net = Net()
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

def client_fn(cid: str) -> fl.client.Client:
    """Create a Flower client-side client."""
    return SinusClient(cid=cid).to_client()

# Define the strategy
base_strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=1.0,  # Sample 100% of available clients for evaluation
    min_fit_clients=2,  # Never sample less than 2 clients for training
    min_evaluate_clients=2,  # Never sample less than 2 clients for evaluation
    min_available_clients=2,  # Wait until all 2 clients are available
)
strategy = MLflowLoggingStrategy(base_strategy)

# Start the simulation
if __name__ == "__main__":
    mlf_cfg = MLflowConfig(
        experiment_name="FedCast",
        run_name="basic_fedavg",
        tags={"strategy": "FedAvg", "dataset": "sinus"},
    )
    with start_run(mlf_cfg):
        log_params({
            "strategy": "FedAvg",
            "num_rounds": 3,
            "num_clients": 2,
            "model": "MLP",
        })
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=2,
            config=fl.server.ServerConfig(num_rounds=3),
            strategy=strategy,
        )
        log_history_artifact(history)
        