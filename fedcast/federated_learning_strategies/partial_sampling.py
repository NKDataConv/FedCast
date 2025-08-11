from __future__ import annotations

from typing import Any

import flwr as fl


def build_partial_sampling_strategy(
    *,
    fraction_fit: float = 0.5,
    fraction_evaluate: float = 0.5,
    min_fit_clients: int = 2,
    min_evaluate_clients: int = 2,
    min_available_clients: int = 2,
) -> Any:
    """FedAvg with partial client sampling each round (pure Flower strategy)."""
    return fl.server.strategy.FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
    )


