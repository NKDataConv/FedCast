from __future__ import annotations

from typing import Any

import flwr as fl


def build_fedprox_strategy(
    *,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    min_fit_clients: int = 2,
    min_evaluate_clients: int = 2,
    min_available_clients: int = 2,
    proximal_mu: float = 0.0,
) -> Any:
    """FedProx strategy that works without initial parameters.

    This strategy is compatible with all models and requires no extra setup.
    """
    return fl.server.strategy.FedProx(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        proximal_mu=proximal_mu,
    )


