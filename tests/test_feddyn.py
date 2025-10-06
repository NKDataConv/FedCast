"""Test suite for FedDyn strategy implementation."""

import pytest
import numpy as np
import flwr as fl
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from fedcast.federated_learning_strategies.feddyn import (
    FedDynStrategy,
    build_feddyn_strategy,
    _compute_dynamic_regularization_term,
    _aggregate_regularization_terms,
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


def test_compute_dynamic_regularization_term():
    """Test dynamic regularization term computation."""
    # Create test data
    client_params = [np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([0.5, 1.5])]
    global_params = [np.array([[1.1, 2.1], [3.1, 4.1]]), np.array([0.6, 1.6])]
    client_regularization_terms = [np.array([[0.1, 0.2], [0.3, 0.4]]), np.array([0.05, 0.15])]
    alpha = 0.01
    
    updated_terms = _compute_dynamic_regularization_term(
        client_params, global_params, client_regularization_terms, alpha
    )
    
    # Check that regularization terms are updated
    assert len(updated_terms) == len(client_regularization_terms)
    
    # Check that the update is applied correctly
    for i, (old_term, new_term) in enumerate(zip(client_regularization_terms, updated_terms)):
        expected_update = alpha * (global_params[i] - client_params[i])
        expected_term = old_term + expected_update
        np.testing.assert_array_almost_equal(new_term, expected_term)


def test_aggregate_regularization_terms():
    """Test regularization terms aggregation."""
    # Create test data
    client_regularization_terms = [
        [np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([0.5, 1.5])],
        [np.array([[2.0, 3.0], [4.0, 5.0]]), np.array([1.0, 2.0])],
        [np.array([[0.5, 1.5], [2.5, 3.5]]), np.array([0.25, 0.75])],
    ]
    client_weights = [100, 200, 50]  # Total: 350
    
    aggregated_terms = _aggregate_regularization_terms(client_regularization_terms, client_weights)
    
    # Check that aggregation produces correct number of layers
    assert len(aggregated_terms) == len(client_regularization_terms[0])
    
    # Check that aggregation is weighted correctly
    # First layer: (100*[1,2;3,4] + 200*[2,3;4,5] + 50*[0.5,1.5;2.5,3.5]) / 350
    expected_first_layer = (
        100 * client_regularization_terms[0][0] +
        200 * client_regularization_terms[1][0] +
        50 * client_regularization_terms[2][0]
    ) / 350
    
    np.testing.assert_array_almost_equal(aggregated_terms[0], expected_first_layer)


def test_aggregate_regularization_terms_empty():
    """Test regularization terms aggregation with empty inputs."""
    # Test with empty regularization terms
    result = _aggregate_regularization_terms([], [])
    assert result == []
    
    # Test with empty weights
    client_terms = [[np.array([1.0, 2.0])]]
    result = _aggregate_regularization_terms(client_terms, [])
    assert result == client_terms[0]


def test_feddyn_strategy_initialization():
    """Test FedDyn strategy initialization."""
    strategy = FedDynStrategy(
        alpha=0.05,
        track_regularization=True,
    )
    
    assert strategy.alpha == 0.05
    assert strategy.track_regularization is True
    assert strategy._round == 0
    assert strategy._last_global_params is None
    assert strategy._global_regularization_terms is None
    assert strategy._regularization_history == []


def test_feddyn_strategy_initialization_with_base():
    """Test FedDyn strategy initialization with custom base strategy."""
    base_strategy = FedAvg()
    strategy = FedDynStrategy(
        base_strategy=base_strategy,
        alpha=0.02,
        track_regularization=False,
    )
    
    assert strategy.base == base_strategy
    assert strategy.alpha == 0.02
    assert strategy.track_regularization is False


def test_feddyn_strategy_configure_fit():
    """Test FedDyn strategy fit configuration."""
    strategy = FedDynStrategy(alpha=0.01)
    client_manager = MockClientManager(num_clients=3)
    
    # Initialize parameters
    initial_params = ndarrays_to_parameters([
        np.random.randn(10, 5),
        np.random.randn(5)
    ])
    
    # Configure fit
    fit_config = strategy.configure_fit(1, initial_params, client_manager)
    
    assert len(fit_config) == 3  # Three clients
    
    # Check that FedDyn-specific configuration is added
    for client_proxy, config in fit_config:
        assert "feddyn_alpha" in config
        assert config["feddyn_alpha"] == 0.01
        assert "feddyn_global_regularization_terms" in config


def test_feddyn_strategy_aggregation():
    """Test FedDyn strategy aggregation process."""
    strategy = FedDynStrategy(
        alpha=0.01,
        track_regularization=True,
    )
    
    client_manager = MockClientManager(num_clients=3)
    
    # Initialize parameters
    initial_params = ndarrays_to_parameters([
        np.random.randn(10, 5),
        np.random.randn(5)
    ])
    
    # Configure fit
    fit_config = strategy.configure_fit(1, initial_params, client_manager)
    
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
    
    # Check FedDyn-specific metrics
    assert "feddyn_alpha" in metrics
    assert "feddyn_regularization_enabled" in metrics
    assert "feddyn_avg_client_drift" in metrics
    assert "feddyn_max_client_drift" in metrics
    assert "feddyn_client_drift_std" in metrics
    
    # Check that regularization history is updated
    assert len(strategy._regularization_history) == 1
    assert strategy._regularization_history[0]["round"] == 1


def test_feddyn_strategy_multiple_rounds():
    """Test FedDyn strategy over multiple rounds."""
    strategy = FedDynStrategy(
        alpha=0.01,
        track_regularization=True,
    )
    
    client_manager = MockClientManager(num_clients=2)
    
    # Run multiple rounds
    initial_params = ndarrays_to_parameters([
        np.random.randn(10, 5),
        np.random.randn(5)
    ])
    
    for round_num in range(1, 4):
        fit_config = strategy.configure_fit(round_num, initial_params, client_manager)
        
        # Create mock results
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
        
        # Check that regularization history grows
        assert len(strategy._regularization_history) == round_num
        assert strategy._regularization_history[-1]["round"] == round_num
        
        # Check that global parameters are updated
        assert strategy._last_global_params is not None


def test_feddyn_strategy_regularization_info():
    """Test FedDyn strategy regularization info retrieval."""
    strategy = FedDynStrategy(alpha=0.01)
    
    # Initially, regularization info should be empty
    info = strategy.get_regularization_info()
    assert info["regularization_history"] == []
    assert info["current_round"] == 0
    assert info["alpha"] == 0.01
    assert info["global_regularization_terms"] is None
    assert info["last_global_params"] is None
    
    # After running a round, info should be populated
    client_manager = MockClientManager(num_clients=2)
    initial_params = ndarrays_to_parameters([
        np.random.randn(10, 5),
        np.random.randn(5)
    ])
    
    fit_config = strategy.configure_fit(1, initial_params, client_manager)
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
    
    strategy.aggregate_fit(1, mock_results, [])
    
    info = strategy.get_regularization_info()
    assert len(info["regularization_history"]) == 1
    assert info["current_round"] == 1
    assert info["last_global_params"] is not None


def test_build_feddyn_strategy():
    """Test FedDyn strategy builder function."""
    strategy = build_feddyn_strategy(
        fraction_fit=0.8,
        fraction_evaluate=0.6,
        min_fit_clients=3,
        min_evaluate_clients=2,
        min_available_clients=4,
        alpha=0.02,
        track_regularization=False,
    )
    
    assert isinstance(strategy, FedDynStrategy)
    assert strategy.alpha == 0.02
    assert strategy.track_regularization is False
    assert isinstance(strategy.base, FedAvg)


def test_feddyn_strategy_edge_cases():
    """Test FedDyn strategy edge cases."""
    strategy = FedDynStrategy(alpha=0.01)
    
    # Test with empty results
    aggregated_params, metrics = strategy.aggregate_fit(1, [], [])
    assert aggregated_params is None
    assert metrics == {}
    
    # Test with None results
    aggregated_params, metrics = strategy.aggregate_fit(1, None, [])
    assert aggregated_params is None
    assert metrics == {}


def test_feddyn_strategy_alpha_values():
    """Test FedDyn strategy with different alpha values."""
    # Test with very small alpha
    strategy_small = FedDynStrategy(alpha=0.001)
    assert strategy_small.alpha == 0.001
    
    # Test with larger alpha
    strategy_large = FedDynStrategy(alpha=0.1)
    assert strategy_large.alpha == 0.1
    
    # Test with zero alpha (should still work)
    strategy_zero = FedDynStrategy(alpha=0.0)
    assert strategy_zero.alpha == 0.0


def test_feddyn_strategy_regularization_metrics():
    """Test FedDyn strategy regularization metrics computation."""
    strategy = FedDynStrategy(alpha=0.01, track_regularization=True)
    
    # Set up some test data
    strategy._last_global_params = [
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([0.5, 1.5])
    ]
    strategy._global_regularization_terms = [
        np.array([[0.1, 0.2], [0.3, 0.4]]),
        np.array([0.05, 0.15])
    ]
    
    client_params_list = [
        [np.array([[1.1, 2.1], [3.1, 4.1]]), np.array([0.6, 1.6])],
        [np.array([[0.9, 1.9], [2.9, 3.9]]), np.array([0.4, 1.4])],
    ]
    client_weights = [100, 150]
    
    metrics = strategy._compute_regularization_metrics(client_params_list, client_weights)
    
    # Check that all expected metrics are present
    expected_metrics = [
        "feddyn_avg_reg_magnitude",
        "feddyn_max_reg_magnitude", 
        "feddyn_reg_magnitude_std",
        "feddyn_avg_client_drift",
        "feddyn_max_client_drift",
        "feddyn_client_drift_std",
        "feddyn_alpha",
        "feddyn_regularization_enabled"
    ]
    
    for metric in expected_metrics:
        assert metric in metrics
        assert isinstance(metrics[metric], (int, float))


if __name__ == "__main__":
    pytest.main([__file__])
