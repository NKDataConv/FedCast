"""Test suite for FedOpt strategy implementation."""

import pytest
import numpy as np
import flwr as fl
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from fedcast.federated_learning_strategies.fedopt import (
    FedOptStrategy,
    build_fedopt_strategy,
    _compute_fedadam_update,
    _compute_fedadagrad_update,
    _compute_fedyogi_update,
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


def test_compute_fedadam_update():
    """Test FedAdam update computation."""
    # Create test data
    gradients = [np.array([[0.1, 0.2], [0.3, 0.4]]), np.array([0.05, 0.15])]
    server_momentum = [np.array([[0.01, 0.02], [0.03, 0.04]]), np.array([0.005, 0.015])]
    server_variance = [np.array([[0.001, 0.002], [0.003, 0.004]]), np.array([0.0005, 0.0015])]
    server_learning_rate = 0.01
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    
    param_updates, updated_momentum, updated_variance = _compute_fedadam_update(
        gradients, server_momentum, server_variance, server_learning_rate, beta1, beta2, epsilon
    )
    
    # Check that all outputs have correct length
    assert len(param_updates) == len(gradients)
    assert len(updated_momentum) == len(gradients)
    assert len(updated_variance) == len(gradients)
    
    # Check that momentum and variance are updated correctly
    for i, (grad, old_momentum, old_variance) in enumerate(zip(gradients, server_momentum, server_variance)):
        # Expected momentum: m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        expected_momentum = beta1 * old_momentum + (1 - beta1) * grad
        np.testing.assert_array_almost_equal(updated_momentum[i], expected_momentum)
        
        # Expected variance: v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
        expected_variance = beta2 * old_variance + (1 - beta2) * (grad ** 2)
        np.testing.assert_array_almost_equal(updated_variance[i], expected_variance)


def test_compute_fedadagrad_update():
    """Test FedAdaGrad update computation."""
    # Create test data
    gradients = [np.array([[0.1, 0.2], [0.3, 0.4]]), np.array([0.05, 0.15])]
    server_variance = [np.array([[0.001, 0.002], [0.003, 0.004]]), np.array([0.0005, 0.0015])]
    server_learning_rate = 0.01
    epsilon = 1e-8
    
    param_updates, updated_variance = _compute_fedadagrad_update(
        gradients, server_variance, server_learning_rate, epsilon
    )
    
    # Check that all outputs have correct length
    assert len(param_updates) == len(gradients)
    assert len(updated_variance) == len(gradients)
    
    # Check that variance is updated correctly
    for i, (grad, old_variance) in enumerate(zip(gradients, server_variance)):
        # Expected variance: v_t = v_{t-1} + g_t^2
        expected_variance = old_variance + (grad ** 2)
        np.testing.assert_array_almost_equal(updated_variance[i], expected_variance)


def test_compute_fedyogi_update():
    """Test FedYogi update computation."""
    # Create test data
    gradients = [np.array([[0.1, 0.2], [0.3, 0.4]]), np.array([0.05, 0.15])]
    server_momentum = [np.array([[0.01, 0.02], [0.03, 0.04]]), np.array([0.005, 0.015])]
    server_variance = [np.array([[0.001, 0.002], [0.003, 0.004]]), np.array([0.0005, 0.0015])]
    server_learning_rate = 0.01
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-3
    
    param_updates, updated_momentum, updated_variance = _compute_fedyogi_update(
        gradients, server_momentum, server_variance, server_learning_rate, beta1, beta2, epsilon
    )
    
    # Check that all outputs have correct length
    assert len(param_updates) == len(gradients)
    assert len(updated_momentum) == len(gradients)
    assert len(updated_variance) == len(gradients)
    
    # Check that momentum is updated correctly
    for i, (grad, old_momentum) in enumerate(zip(gradients, server_momentum)):
        # Expected momentum: m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        expected_momentum = beta1 * old_momentum + (1 - beta1) * grad
        np.testing.assert_array_almost_equal(updated_momentum[i], expected_momentum)


def test_fedopt_strategy_initialization():
    """Test FedOpt strategy initialization."""
    strategy = FedOptStrategy(
        optimizer_type="adam",
        server_learning_rate=0.01,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        track_optimizer_state=True,
    )
    
    assert strategy.optimizer_type == "adam"
    assert strategy.server_learning_rate == 0.01
    assert strategy.beta1 == 0.9
    assert strategy.beta2 == 0.999
    assert strategy.epsilon == 1e-8
    assert strategy.track_optimizer_state is True
    assert strategy._round == 0
    assert strategy._last_global_params is None
    assert strategy._server_momentum is None
    assert strategy._server_variance is None
    assert strategy._optimizer_history == []


def test_fedopt_strategy_initialization_with_base():
    """Test FedOpt strategy initialization with custom base strategy."""
    base_strategy = FedAvg()
    strategy = FedOptStrategy(
        base_strategy=base_strategy,
        optimizer_type="adagrad",
        server_learning_rate=0.02,
        track_optimizer_state=False,
    )
    
    assert strategy.base == base_strategy
    assert strategy.optimizer_type == "adagrad"
    assert strategy.server_learning_rate == 0.02
    assert strategy.track_optimizer_state is False


def test_fedopt_strategy_invalid_optimizer():
    """Test FedOpt strategy with invalid optimizer type."""
    with pytest.raises(ValueError, match="Unsupported optimizer type"):
        FedOptStrategy(optimizer_type="invalid")


def test_fedopt_strategy_aggregation_adam():
    """Test FedOpt strategy aggregation process with Adam optimizer."""
    strategy = FedOptStrategy(
        optimizer_type="adam",
        server_learning_rate=0.01,
        track_optimizer_state=True,
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
    
    # Check FedOpt-specific metrics
    assert "fedopt_optimizer_type" in metrics
    assert "fedopt_server_learning_rate" in metrics
    assert "fedopt_adaptive_optimization_enabled" in metrics
    assert "fedopt_avg_gradient_norm" in metrics
    assert "fedopt_max_gradient_norm" in metrics
    assert "fedopt_gradient_norm_std" in metrics
    assert "fedopt_avg_momentum_norm" in metrics
    assert "fedopt_avg_variance_norm" in metrics
    
    # Check that optimizer history is updated
    assert len(strategy._optimizer_history) == 1
    assert strategy._optimizer_history[0]["round"] == 1


def test_fedopt_strategy_aggregation_adagrad():
    """Test FedOpt strategy aggregation process with AdaGrad optimizer."""
    strategy = FedOptStrategy(
        optimizer_type="adagrad",
        server_learning_rate=0.01,
        track_optimizer_state=True,
    )
    
    client_manager = MockClientManager(num_clients=2)
    
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
    
    # Check that AdaGrad-specific metrics are present
    assert "fedopt_avg_variance_norm" in metrics
    # AdaGrad doesn't use momentum, so momentum metrics should be 0 or not present
    assert "fedopt_avg_momentum_norm" in metrics


def test_fedopt_strategy_aggregation_yogi():
    """Test FedOpt strategy aggregation process with Yogi optimizer."""
    strategy = FedOptStrategy(
        optimizer_type="yogi",
        server_learning_rate=0.01,
        track_optimizer_state=True,
    )
    
    client_manager = MockClientManager(num_clients=2)
    
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
    
    # Check that Yogi-specific metrics are present
    assert "fedopt_avg_momentum_norm" in metrics
    assert "fedopt_avg_variance_norm" in metrics


def test_fedopt_strategy_multiple_rounds():
    """Test FedOpt strategy over multiple rounds."""
    strategy = FedOptStrategy(
        optimizer_type="adam",
        server_learning_rate=0.01,
        track_optimizer_state=True,
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
        
        # Check that optimizer history grows
        assert len(strategy._optimizer_history) == round_num
        assert strategy._optimizer_history[-1]["round"] == round_num
        
        # Check that global parameters are updated
        assert strategy._last_global_params is not None


def test_fedopt_strategy_optimizer_info():
    """Test FedOpt strategy optimizer info retrieval."""
    strategy = FedOptStrategy(optimizer_type="adam", server_learning_rate=0.01)
    
    # Initially, optimizer info should be empty
    info = strategy.get_optimizer_info()
    assert info["optimizer_history"] == []
    assert info["current_round"] == 0
    assert info["optimizer_type"] == "adam"
    assert info["server_learning_rate"] == 0.01
    assert info["server_momentum"] is None
    assert info["server_variance"] is None
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
    
    info = strategy.get_optimizer_info()
    assert len(info["optimizer_history"]) == 1
    assert info["current_round"] == 1
    assert info["last_global_params"] is not None
    assert info["server_momentum"] is not None
    assert info["server_variance"] is not None


def test_build_fedopt_strategy():
    """Test FedOpt strategy builder function."""
    strategy = build_fedopt_strategy(
        fraction_fit=0.8,
        fraction_evaluate=0.6,
        min_fit_clients=3,
        min_evaluate_clients=2,
        min_available_clients=4,
        optimizer_type="yogi",
        server_learning_rate=0.02,
        beta1=0.8,
        beta2=0.99,
        epsilon=1e-6,
        track_optimizer_state=False,
    )
    
    assert isinstance(strategy, FedOptStrategy)
    assert strategy.optimizer_type == "yogi"
    assert strategy.server_learning_rate == 0.02
    assert strategy.beta1 == 0.8
    assert strategy.beta2 == 0.99
    assert strategy.epsilon == 1e-6
    assert strategy.track_optimizer_state is False
    assert isinstance(strategy.base, FedAvg)


def test_fedopt_strategy_edge_cases():
    """Test FedOpt strategy edge cases."""
    strategy = FedOptStrategy(optimizer_type="adam", server_learning_rate=0.01)
    
    # Test with empty results
    aggregated_params, metrics = strategy.aggregate_fit(1, [], [])
    assert aggregated_params is None
    assert metrics == {}
    
    # Test with None results
    aggregated_params, metrics = strategy.aggregate_fit(1, None, [])
    assert aggregated_params is None
    assert metrics == {}


def test_fedopt_strategy_optimizer_types():
    """Test FedOpt strategy with different optimizer types."""
    # Test Adam
    strategy_adam = FedOptStrategy(optimizer_type="adam")
    assert strategy_adam.optimizer_type == "adam"
    
    # Test AdaGrad
    strategy_adagrad = FedOptStrategy(optimizer_type="adagrad")
    assert strategy_adagrad.optimizer_type == "adagrad"
    
    # Test Yogi
    strategy_yogi = FedOptStrategy(optimizer_type="yogi")
    assert strategy_yogi.optimizer_type == "yogi"
    
    # Test case insensitive
    strategy_adam_upper = FedOptStrategy(optimizer_type="ADAM")
    assert strategy_adam_upper.optimizer_type == "adam"


def test_fedopt_strategy_learning_rate_values():
    """Test FedOpt strategy with different learning rate values."""
    # Test with very small learning rate
    strategy_small = FedOptStrategy(server_learning_rate=0.001)
    assert strategy_small.server_learning_rate == 0.001
    
    # Test with larger learning rate
    strategy_large = FedOptStrategy(server_learning_rate=0.1)
    assert strategy_large.server_learning_rate == 0.1
    
    # Test with zero learning rate (should still work)
    strategy_zero = FedOptStrategy(server_learning_rate=0.0)
    assert strategy_zero.server_learning_rate == 0.0


def test_fedopt_strategy_optimizer_metrics():
    """Test FedOpt strategy optimizer metrics computation."""
    strategy = FedOptStrategy(optimizer_type="adam", track_optimizer_state=True)
    
    # Set up some test data
    strategy._last_global_params = [
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([0.5, 1.5])
    ]
    strategy._server_momentum = [
        np.array([[0.1, 0.2], [0.3, 0.4]]),
        np.array([0.05, 0.15])
    ]
    strategy._server_variance = [
        np.array([[0.01, 0.02], [0.03, 0.04]]),
        np.array([0.005, 0.015])
    ]
    
    gradients = [
        np.array([[0.1, 0.2], [0.3, 0.4]]),
        np.array([0.05, 0.15])
    ]
    
    metrics = strategy._compute_optimizer_metrics(gradients)
    
    # Check that all expected metrics are present
    expected_metrics = [
        "fedopt_avg_gradient_norm",
        "fedopt_max_gradient_norm", 
        "fedopt_gradient_norm_std",
        "fedopt_avg_momentum_norm",
        "fedopt_max_momentum_norm",
        "fedopt_momentum_norm_std",
        "fedopt_avg_variance_norm",
        "fedopt_max_variance_norm",
        "fedopt_variance_norm_std",
        "fedopt_optimizer_type",
        "fedopt_server_learning_rate",
        "fedopt_beta1",
        "fedopt_beta2",
        "fedopt_epsilon",
        "fedopt_adaptive_optimization_enabled"
    ]
    
    for metric in expected_metrics:
        assert metric in metrics
        assert isinstance(metrics[metric], (int, float))


if __name__ == "__main__":
    pytest.main([__file__])
