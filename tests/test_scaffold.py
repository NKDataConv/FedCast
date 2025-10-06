"""Test suite for SCAFFOLD strategy implementation."""

import pytest
import numpy as np
import flwr as fl
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from fedcast.federated_learning_strategies.scaffold import (
    SCAFFOLDStrategy,
    build_scaffold_strategy,
    _compute_control_variate_update,
    _aggregate_control_variates,
    _apply_control_variate_correction,
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


def test_compute_control_variate_update():
    """Test control variate update computation."""
    # Create test data
    client_params = [np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([0.5, 1.5])]
    global_params = [np.array([[1.1, 2.1], [3.1, 4.1]]), np.array([0.6, 1.6])]
    client_control_variate = [np.array([[0.1, 0.2], [0.3, 0.4]]), np.array([0.05, 0.15])]
    global_control_variate = [np.array([[0.2, 0.3], [0.4, 0.5]]), np.array([0.1, 0.2])]
    learning_rate = 0.01
    
    updated_cv = _compute_control_variate_update(
        client_params, global_params, client_control_variate, global_control_variate, learning_rate
    )
    
    # Check that control variates are updated
    assert len(updated_cv) == len(client_control_variate)
    
    # Check that the update is applied correctly
    for i, (old_cv, new_cv) in enumerate(zip(client_control_variate, updated_cv)):
        expected_update = (1.0 / learning_rate) * (global_params[i] - client_params[i])
        expected_cv = old_cv + expected_update
        np.testing.assert_array_almost_equal(new_cv, expected_cv)


def test_aggregate_control_variates():
    """Test control variates aggregation."""
    # Create test data
    client_control_variates = [
        [np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([0.5, 1.5])],
        [np.array([[2.0, 3.0], [4.0, 5.0]]), np.array([1.0, 2.0])],
        [np.array([[0.5, 1.5], [2.5, 3.5]]), np.array([0.25, 0.75])],
    ]
    client_weights = [100, 200, 50]  # Total: 350
    
    aggregated_cv = _aggregate_control_variates(client_control_variates, client_weights)
    
    # Check that aggregation produces correct number of layers
    assert len(aggregated_cv) == len(client_control_variates[0])
    
    # Check that aggregation is weighted correctly
    # First layer: (100*[1,2;3,4] + 200*[2,3;4,5] + 50*[0.5,1.5;2.5,3.5]) / 350
    expected_first_layer = (
        100 * client_control_variates[0][0] +
        200 * client_control_variates[1][0] +
        50 * client_control_variates[2][0]
    ) / 350
    
    np.testing.assert_array_almost_equal(aggregated_cv[0], expected_first_layer)


def test_aggregate_control_variates_empty():
    """Test control variates aggregation with empty inputs."""
    # Test with empty control variates
    result = _aggregate_control_variates([], [])
    assert result == []
    
    # Test with empty weights
    client_cv = [[np.array([1.0, 2.0])]]
    result = _aggregate_control_variates(client_cv, [])
    assert result == client_cv[0]


def test_apply_control_variate_correction():
    """Test control variate correction application."""
    # Create test data
    client_params = [np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([0.5, 1.5])]
    global_params = [np.array([[1.1, 2.1], [3.1, 4.1]]), np.array([0.6, 1.6])]
    client_control_variate = [np.array([[0.1, 0.2], [0.3, 0.4]]), np.array([0.05, 0.15])]
    global_control_variate = [np.array([[0.2, 0.3], [0.4, 0.5]]), np.array([0.1, 0.2])]
    correction_strength = 1.0
    
    corrected_params = _apply_control_variate_correction(
        client_params, global_params, client_control_variate, 
        global_control_variate, correction_strength
    )
    
    # Check that correction is applied correctly
    assert len(corrected_params) == len(client_params)
    
    for i, (client_param, client_cv, global_cv) in enumerate(zip(client_params, client_control_variate, global_control_variate)):
        expected_correction = correction_strength * (global_cv - client_cv)
        expected_param = client_param + expected_correction
        np.testing.assert_array_almost_equal(corrected_params[i], expected_param)


def test_scaffold_strategy_initialization():
    """Test SCAFFOLD strategy initialization."""
    strategy = SCAFFOLDStrategy(
        learning_rate=0.01,
        correction_strength=1.0,
        track_control_variates=True,
    )
    
    assert strategy.learning_rate == 0.01
    assert strategy.correction_strength == 1.0
    assert strategy.track_control_variates is True
    assert strategy._round == 0
    assert strategy._last_global_params is None
    assert strategy._global_control_variate is None
    assert strategy._client_control_variates == {}
    assert strategy._control_variate_history == []


def test_scaffold_strategy_initialization_with_base():
    """Test SCAFFOLD strategy initialization with custom base strategy."""
    base_strategy = FedAvg()
    strategy = SCAFFOLDStrategy(
        base_strategy=base_strategy,
        learning_rate=0.02,
        correction_strength=0.5,
        track_control_variates=False,
    )
    
    assert strategy.base == base_strategy
    assert strategy.learning_rate == 0.02
    assert strategy.correction_strength == 0.5
    assert strategy.track_control_variates is False


def test_scaffold_strategy_configure_fit():
    """Test SCAFFOLD strategy fit configuration."""
    strategy = SCAFFOLDStrategy(learning_rate=0.01)
    client_manager = MockClientManager(num_clients=3)
    
    # Initialize parameters
    initial_params = ndarrays_to_parameters([
        np.random.randn(10, 5),
        np.random.randn(5)
    ])
    
    # Configure fit
    fit_config = strategy.configure_fit(1, initial_params, client_manager)
    
    assert len(fit_config) == 3  # Three clients
    
    # Check that SCAFFOLD-specific configuration is added
    for client_proxy, config in fit_config:
        assert "scaffold_learning_rate" in config
        assert config["scaffold_learning_rate"] == 0.01
        assert "scaffold_global_control_variate" in config
        assert "scaffold_correction_strength" in config


def test_scaffold_strategy_aggregation():
    """Test SCAFFOLD strategy aggregation process."""
    strategy = SCAFFOLDStrategy(
        learning_rate=0.01,
        correction_strength=1.0,
        track_control_variates=True,
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
    
    # Check SCAFFOLD-specific metrics
    assert "scaffold_learning_rate" in metrics
    assert "scaffold_correction_strength" in metrics
    assert "scaffold_control_variates_enabled" in metrics
    assert "scaffold_avg_cv_magnitude" in metrics
    assert "scaffold_max_cv_magnitude" in metrics
    assert "scaffold_cv_magnitude_std" in metrics
    assert "scaffold_cv_variance" in metrics
    
    # Check that control variate history is updated
    assert len(strategy._control_variate_history) == 1
    assert strategy._control_variate_history[0]["round"] == 1


def test_scaffold_strategy_multiple_rounds():
    """Test SCAFFOLD strategy over multiple rounds."""
    strategy = SCAFFOLDStrategy(
        learning_rate=0.01,
        correction_strength=1.0,
        track_control_variates=True,
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
        
        # Check that control variate history grows
        assert len(strategy._control_variate_history) == round_num
        assert strategy._control_variate_history[-1]["round"] == round_num
        
        # Check that global parameters are updated
        assert strategy._last_global_params is not None


def test_scaffold_strategy_control_variate_info():
    """Test SCAFFOLD strategy control variate info retrieval."""
    strategy = SCAFFOLDStrategy(learning_rate=0.01)
    
    # Initially, control variate info should be empty
    info = strategy.get_control_variate_info()
    assert info["control_variate_history"] == []
    assert info["current_round"] == 0
    assert info["learning_rate"] == 0.01
    assert info["correction_strength"] == 1.0
    assert info["global_control_variate"] is None
    assert info["client_control_variates"] == {}
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
    
    info = strategy.get_control_variate_info()
    assert len(info["control_variate_history"]) == 1
    assert info["current_round"] == 1
    assert info["last_global_params"] is not None


def test_build_scaffold_strategy():
    """Test SCAFFOLD strategy builder function."""
    strategy = build_scaffold_strategy(
        fraction_fit=0.8,
        fraction_evaluate=0.6,
        min_fit_clients=3,
        min_evaluate_clients=2,
        min_available_clients=4,
        learning_rate=0.02,
        correction_strength=0.5,
        track_control_variates=False,
    )
    
    assert isinstance(strategy, SCAFFOLDStrategy)
    assert strategy.learning_rate == 0.02
    assert strategy.correction_strength == 0.5
    assert strategy.track_control_variates is False
    assert isinstance(strategy.base, FedAvg)


def test_scaffold_strategy_edge_cases():
    """Test SCAFFOLD strategy edge cases."""
    strategy = SCAFFOLDStrategy(learning_rate=0.01)
    
    # Test with empty results
    aggregated_params, metrics = strategy.aggregate_fit(1, [], [])
    assert aggregated_params is None
    assert metrics == {}
    
    # Test with None results
    aggregated_params, metrics = strategy.aggregate_fit(1, None, [])
    assert aggregated_params is None
    assert metrics == {}


def test_scaffold_strategy_learning_rate_values():
    """Test SCAFFOLD strategy with different learning rate values."""
    # Test with very small learning rate
    strategy_small = SCAFFOLDStrategy(learning_rate=0.001)
    assert strategy_small.learning_rate == 0.001
    
    # Test with larger learning rate
    strategy_large = SCAFFOLDStrategy(learning_rate=0.1)
    assert strategy_large.learning_rate == 0.1
    
    # Test with zero learning rate (should still work)
    strategy_zero = SCAFFOLDStrategy(learning_rate=0.0)
    assert strategy_zero.learning_rate == 0.0


def test_scaffold_strategy_correction_strength_values():
    """Test SCAFFOLD strategy with different correction strength values."""
    # Test with small correction strength
    strategy_small = SCAFFOLDStrategy(correction_strength=0.1)
    assert strategy_small.correction_strength == 0.1
    
    # Test with larger correction strength
    strategy_large = SCAFFOLDStrategy(correction_strength=2.0)
    assert strategy_large.correction_strength == 2.0
    
    # Test with zero correction strength (no correction)
    strategy_zero = SCAFFOLDStrategy(correction_strength=0.0)
    assert strategy_zero.correction_strength == 0.0


def test_scaffold_strategy_control_variate_metrics():
    """Test SCAFFOLD strategy control variate metrics computation."""
    strategy = SCAFFOLDStrategy(learning_rate=0.01, track_control_variates=True)
    
    # Set up some test data
    strategy._last_global_params = [
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([0.5, 1.5])
    ]
    strategy._global_control_variate = [
        np.array([[0.1, 0.2], [0.3, 0.4]]),
        np.array([0.05, 0.15])
    ]
    
    client_control_variates = [
        [np.array([[0.1, 0.2], [0.3, 0.4]]), np.array([0.05, 0.15])],
        [np.array([[0.2, 0.3], [0.4, 0.5]]), np.array([0.1, 0.2])],
    ]
    client_weights = [100, 150]
    
    metrics = strategy._compute_control_variate_metrics(client_control_variates, client_weights)
    
    # Check that all expected metrics are present
    expected_metrics = [
        "scaffold_global_cv_magnitude",
        "scaffold_avg_cv_magnitude",
        "scaffold_max_cv_magnitude", 
        "scaffold_cv_magnitude_std",
        "scaffold_cv_variance",
        "scaffold_learning_rate",
        "scaffold_correction_strength",
        "scaffold_control_variates_enabled"
    ]
    
    for metric in expected_metrics:
        assert metric in metrics
        assert isinstance(metrics[metric], (int, float))


if __name__ == "__main__":
    pytest.main([__file__])
