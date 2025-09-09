"""Test suite for FedLAMA strategy implementation."""

import pytest
import numpy as np
import flwr as fl
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from fedcast.federated_learning_strategies.fedlama import (
    FedLAMAStrategy,
    build_fedlama_strategy,
    _compute_layer_discrepancy,
    _compute_layer_importance,
    _adaptive_aggregation_interval,
)


class MockClientProxy(ClientProxy):
    """Mock client proxy for testing."""
    
    def __init__(self, cid: str):
        self.cid = cid
    
    def get_properties(self, ins, timeout=None):
        return {}
    
    def get_parameters(self, ins, timeout=None):
        return Parameters(tensors=[], tensor_type="numpy.ndarray")
    
    def fit(self, ins, timeout=None):
        # Return mock fit result
        params = ndarrays_to_parameters([np.random.randn(10, 5), np.random.randn(5)])
        return fl.common.FitRes(
            status=fl.common.Status(code=fl.common.Code.OK, message="Success"),
            parameters=params,
            num_examples=100,
            metrics={},
        )
    
    def evaluate(self, ins, timeout=None):
        return fl.common.EvaluateRes(
            status=fl.common.Status(code=fl.common.Code.OK, message="Success"),
            loss=0.5,
            num_examples=50,
            metrics={},
        )
    
    def reconnect(self, reconnect_ins, timeout=None):
        return fl.common.ReconnectRes(
            status=fl.common.Status(code=fl.common.Code.OK, message="Success")
        )


class MockClientManager:
    """Mock client manager for testing."""
    
    def __init__(self, num_clients: int = 3):
        self.clients = [MockClientProxy(f"client_{i}") for i in range(num_clients)]
    
    def num_available(self) -> int:
        return len(self.clients)
    
    def wait_for(self, num_clients: int, timeout: int = 1):
        return self.clients[:num_clients]
    
    def sample(self, num_clients: int, min_num_clients: int = 1):
        """Sample clients for training."""
        return self.clients[:num_clients]


def test_compute_layer_discrepancy():
    """Test layer discrepancy computation."""
    # Create test data
    global_layer = np.array([[1.0, 2.0], [3.0, 4.0]])
    client_layers = [
        np.array([[1.1, 2.1], [3.1, 4.1]]),  # Small difference
        np.array([[0.9, 1.9], [2.9, 3.9]]),  # Small difference
    ]
    
    discrepancy = _compute_layer_discrepancy(client_layers, global_layer)
    
    # Should be a positive number
    assert discrepancy > 0
    # Should be relatively small for similar layers
    assert discrepancy < 1.0


def test_compute_layer_importance():
    """Test layer importance computation."""
    global_layer = np.array([[1.0, 2.0], [3.0, 4.0]])
    client_layers = [
        np.array([[1.1, 2.1], [3.1, 4.1]]),
        np.array([[0.9, 1.9], [2.9, 3.9]]),
    ]
    
    importance = _compute_layer_importance(client_layers, global_layer, 0, 3)
    
    # Should be a positive number
    assert importance > 0
    # First layer should have higher importance than deeper layers
    importance_deep = _compute_layer_importance(client_layers, global_layer, 2, 3)
    assert importance > importance_deep


def test_adaptive_aggregation_interval():
    """Test adaptive aggregation interval computation."""
    # High discrepancy should lead to more frequent aggregation
    interval_high_disc = _adaptive_aggregation_interval(
        discrepancy=0.5,  # High discrepancy
        importance=1.0,
        current_interval=5,
        min_interval=1,
        max_interval=10,
        discrepancy_threshold=0.1
    )
    
    # Low discrepancy should lead to less frequent aggregation
    interval_low_disc = _adaptive_aggregation_interval(
        discrepancy=0.05,  # Low discrepancy
        importance=1.0,
        current_interval=5,
        min_interval=1,
        max_interval=10,
        discrepancy_threshold=0.1
    )
    
    assert interval_high_disc < interval_low_disc
    assert interval_high_disc >= 1
    assert interval_low_disc <= 10


def test_fedlama_strategy_initialization():
    """Test FedLAMA strategy initialization."""
    strategy = FedLAMAStrategy(
        min_aggregation_interval=2,
        max_aggregation_interval=8,
        discrepancy_threshold=0.2,
    )
    
    assert strategy.min_interval == 2
    assert strategy.max_interval == 8
    assert strategy.discrepancy_threshold == 0.2
    assert strategy._total_layers == 0  # Not initialized yet


def test_fedlama_strategy_aggregation():
    """Test FedLAMA strategy aggregation process."""
    strategy = FedLAMAStrategy(
        min_aggregation_interval=1,
        max_aggregation_interval=5,
        discrepancy_threshold=0.1,
    )
    
    client_manager = MockClientManager(num_clients=3)
    
    # Initialize parameters - create mock initial parameters
    initial_params = ndarrays_to_parameters([
        np.random.randn(10, 5),
        np.random.randn(5)
    ])
    
    # Configure fit
    fit_config = strategy.configure_fit(1, initial_params, client_manager)
    assert len(fit_config) == 3  # Three clients
    
    # Mock fit results
    mock_results = []
    for client_proxy, _ in fit_config:
        fit_res = fl.common.FitRes(
            status=fl.common.Status(code=fl.common.Code.OK, message="Success"),
            parameters=ndarrays_to_parameters([
                np.random.randn(10, 5),
                np.random.randn(5)
            ]),
            num_examples=100,
            metrics={},
        )
        mock_results.append((client_proxy, fit_res))
    
    # Aggregate fit
    aggregated_params, metrics = strategy.aggregate_fit(1, mock_results, [])
    
    assert aggregated_params is not None
    assert isinstance(metrics, dict)
    assert "fedlama_layers_aggregated" in metrics
    assert "fedlama_avg_discrepancy" in metrics
    assert "fedlama_communication_ratio" in metrics


def test_build_fedlama_strategy():
    """Test FedLAMA strategy builder function."""
    strategy = build_fedlama_strategy(
        min_aggregation_interval=2,
        max_aggregation_interval=6,
        discrepancy_threshold=0.15,
    )
    
    assert isinstance(strategy, FedLAMAStrategy)
    assert strategy.min_interval == 2
    assert strategy.max_interval == 6
    assert strategy.discrepancy_threshold == 0.15


def test_fedlama_layer_tracking():
    """Test that FedLAMA properly tracks layer aggregation intervals."""
    strategy = FedLAMAStrategy(
        min_aggregation_interval=1,
        max_aggregation_interval=3,
        discrepancy_threshold=0.1,
    )
    
    client_manager = MockClientManager(num_clients=2)
    
    # Run multiple rounds to test layer tracking
    initial_params = ndarrays_to_parameters([
        np.random.randn(10, 5),
        np.random.randn(5)
    ])
    
    for round_num in range(1, 4):
        fit_config = strategy.configure_fit(round_num, initial_params, client_manager)
        
        # Create mock results with consistent layer structure
        mock_results = []
        for client_proxy, _ in fit_config:
            fit_res = fl.common.FitRes(
                status=fl.common.Status(code=fl.common.Code.OK, message="Success"),
                parameters=ndarrays_to_parameters([
                    np.random.randn(10, 5),
                    np.random.randn(5)
                ]),
                num_examples=100,
                metrics={},
            )
            mock_results.append((client_proxy, fit_res))
        
        aggregated_params, metrics = strategy.aggregate_fit(round_num, mock_results, [])
        initial_params = aggregated_params
        
        # Check that layer tracking is working
        if round_num > 1:
            assert strategy._total_layers > 0
            assert len(strategy._layer_aggregation_intervals) == strategy._total_layers
            assert len(strategy._layer_last_aggregated) == strategy._total_layers


if __name__ == "__main__":
    pytest.main([__file__])
