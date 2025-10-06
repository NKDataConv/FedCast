from __future__ import annotations

import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import OrderedDict
from typing import Callable, Dict, Tuple

from flwr.common import Context

# Model registry
from fedcast.cast_models import MLPModel, LinearModel

# Dataset loaders (all supported datasets)
from fedcast.datasets import dataset_sinus as ds_sinus
from fedcast.datasets import dataset_ecg as ds_ecg
from fedcast.datasets import dataset_eld as ds_eld
from fedcast.datasets import dataset_intel_iot as ds_iot
from fedcast.datasets import dataset_network_traffic as ds_net
from fedcast.datasets import dataset_stocks as ds_stocks
from fedcast.datasets import dataset_weather as ds_weather

# Strategy builders and telemetry
from fedcast.federated_learning_strategies import (
    build_fedavg_strategy,
    build_partial_sampling_strategy,
    build_fedprox_strategy,
    build_fedtrend_strategy,
    build_fedlama_strategy,
    build_fednova_strategy,
    build_feddyn_strategy,
    build_scaffold_strategy,
    build_fedopt_strategy,
)
from fedcast.telemetry.mlflow_logger import (
    MLflowLoggingStrategy,
    MLflowConfig,
    start_run,
    log_params,
    log_history_artifact,
)


# Use the shared window size used by models
from fedcast.datasets.dataset_sinus import WINDOW_SIZE


class GenericTimeSeriesClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: str,
        model_builder: Callable[[], torch.nn.Module],
        dataset_loader: Callable[[int, int], any],
        dataset_name: str,
        num_partitions: int,
    ) -> None:
        # Map Flower's potentially large node_id into a valid partition range
        self.cid_int = int(cid)
        self.partition_id = int(self.cid_int % max(1, num_partitions))
        self.model = model_builder()
        self.dataset_loader = dataset_loader
        self.dataset_name = dataset_name

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()

        dataset = self.dataset_loader(partition_id=self.partition_id, num_examples=500)
        dataset.set_format("torch", columns=["x", "y"])  # type: ignore[attr-defined]
        trainloader = DataLoader(dataset, batch_size=32, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        for _ in range(5):
            for batch in trainloader:
                inputs, labels = batch["x"].float(), batch["y"].float().view(-1, 1)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()

        dataset = self.dataset_loader(partition_id=self.partition_id, num_examples=100)
        dataset.set_format("torch", columns=["x", "y"])  # type: ignore[attr-defined]
        valloader = DataLoader(dataset, batch_size=32)

        criterion = nn.MSELoss()
        total_loss = 0.0
        with torch.no_grad():
            for batch in valloader:
                inputs, labels = batch["x"].float(), batch["y"].float().view(-1, 1)
                outputs = self.model(inputs)
                total_loss += criterion(outputs, labels).item()

        avg_loss = total_loss / max(1, len(valloader))
        return float(avg_loss), len(valloader.dataset), {"mse": float(total_loss)}


def make_client_fn(
    model_builder: Callable[[], torch.nn.Module],
    dataset_loader: Callable[[int, int], any],
    dataset_name: str,
    num_partitions: int,
):
    def _client_fn(context: Context) -> fl.client.Client:
        cid = str(getattr(context, "node_id", "0"))
        return GenericTimeSeriesClient(
            cid=cid,
            model_builder=model_builder,
            dataset_loader=dataset_loader,
            dataset_name=dataset_name,
            num_partitions=num_partitions,
        ).to_client()

    return _client_fn


def _get_num_partitions_for_dataset(dataset_module) -> int:
    """Best-effort detection of available partitions for a dataset module.

    Falls back to a small default to keep simulations fast.
    """
    candidate_funcs = [
        "get_patient_ids",
        "get_client_ids",
        "get_sensor_ids",
        "get_traffic_categories",
        "get_stock_symbols",
        "get_weather_stations",
    ]
    for fn_name in candidate_funcs:
        getter = getattr(dataset_module, fn_name, None)
        if callable(getter):
            try:
                items = getter()
                return max(2, min(10, len(items)))  # cap to avoid huge runs
            except Exception:
                continue
    # default when unknown
    return 2


def get_dataset_registry() -> Dict[str, Tuple[Callable[[int, int], any], int]]:
    """Return mapping of dataset name -> (loader_fn, num_partitions)."""
    registry: Dict[str, Tuple[Callable[[int, int], any], int]] = {
        "sinus": (ds_sinus.load_dataset, _get_num_partitions_for_dataset(ds_sinus)),
        "ecg": (ds_ecg.load_dataset, _get_num_partitions_for_dataset(ds_ecg)),
        "eld": (ds_eld.load_dataset, _get_num_partitions_for_dataset(ds_eld)),
        "intel_iot": (ds_iot.load_dataset, _get_num_partitions_for_dataset(ds_iot)),
        "network_traffic": (ds_net.load_dataset, _get_num_partitions_for_dataset(ds_net)),
        "stocks": (ds_stocks.load_dataset, _get_num_partitions_for_dataset(ds_stocks)),
        "weather": (ds_weather.load_dataset, _get_num_partitions_for_dataset(ds_weather)),
    }
    return registry


def get_model_registry() -> Dict[str, Callable[[], torch.nn.Module]]:
    return {
        "MLP": MLPModel,
        "Linear": LinearModel,
    }


def get_strategy_registry() -> Dict[str, Callable[[], any]]:
    return {
        "FedAvg": lambda: build_fedavg_strategy(),
        "PartialSampling": lambda: build_partial_sampling_strategy(),
        "FedProx": lambda: build_fedprox_strategy(),
        "FedTrend": lambda: build_fedtrend_strategy(),
        "FedLAMA": lambda: build_fedlama_strategy(),
        "FedNova": lambda: build_fednova_strategy(),
        "FedDyn": lambda: build_feddyn_strategy(),
        "SCAFFOLD": lambda: build_scaffold_strategy(),
        "FedOpt": lambda: build_fedopt_strategy(),
    }


def run_all_experiments(num_rounds: int = 3) -> None:
    datasets = get_dataset_registry()
    models = get_model_registry()
    strategies = get_strategy_registry()

    for dataset_name, (dataset_loader, num_partitions) in datasets.items():
        for model_name, model_builder in models.items():
            for strategy_name, strategy_builder in strategies.items():
                base_strategy = strategy_builder()
                strategy = MLflowLoggingStrategy(
                    base_strategy, dataset_name=dataset_name
                )

                run_name = f"grid_{dataset_name}_{model_name}_{strategy_name}"
                mlf_cfg = MLflowConfig(
                    experiment_name="FedCast",
                    run_name=run_name,
                    tags={
                        "strategy": strategy_name,
                        "dataset": dataset_name,
                        "model": model_name,
                    },
                )

                # Keep number of clients manageable
                num_clients = max(2, min(5, num_partitions))

                with start_run(mlf_cfg):
                    log_params(
                        {
                            "strategy": strategy_name,
                            "num_rounds": num_rounds,
                            "num_clients": num_clients,
                            "model": model_name,
                            "dataset": dataset_name,
                            "window_size": WINDOW_SIZE,
                        }
                    )

                    history = fl.simulation.start_simulation(
                        client_fn=make_client_fn(
                            model_builder=model_builder,
                            dataset_loader=dataset_loader,
                            dataset_name=dataset_name,
                            num_partitions=num_partitions,
                        ),
                        num_clients=num_clients,
                        config=fl.server.ServerConfig(num_rounds=num_rounds),
                        strategy=strategy,
                    )

                    log_history_artifact(history)


if __name__ == "__main__":
    run_all_experiments(num_rounds=3)
