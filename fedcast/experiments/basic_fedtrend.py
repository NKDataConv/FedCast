"""Example of using FedTrend strategy in a Flower server application.

This script demonstrates how to use the FedTrend strategy with a base strategy
(like FedAdam) in a Flower server application, similar to the pattern shown
in the user's example script.

Reference: Yuan et al. "Tackling Data Heterogeneity in Federated Time Series Forecasting"
Paper: https://arxiv.org/abs/2411.15716
"""

from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.common import Context

from fedcast.federated_learning_strategies import build_fedtrend_strategy


def server_fn(context: Context) -> ServerAppComponents:
    """Server function that creates a FedTrend strategy with FedAdam base."""
    num_rounds = context.run_config.get("num-server-rounds", 50)

    # Create FedTrend strategy with FedAdam as base strategy
    strategy = build_fedtrend_strategy(
        l_ct=5,                # recompute client-consensus direction every 5 rounds
        l_gt=5,                # recompute global-trend direction every 5 rounds
        consistency_threshold=0.6,
        eta_refine=0.05,       # refinement step size on global parameters
        ema_momentum=0.9,
    )

    return ServerAppComponents(
        strategy=strategy,
        config=ServerConfig(num_rounds=num_rounds),
    )


app = ServerApp(server_fn=server_fn)


if __name__ == "__main__":
    # This can be run with: flower-server-app fedcast.experiments.basic_fedtrend:app
    print("FedTrend server app created. Run with:")
    print("flower-server-app fedcast.experiments.basic_fedtrend:app")
