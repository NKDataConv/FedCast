"""Tests for FedNova strategy implementation."""

import numpy as np
import pytest
from unittest.mock import Mock, MagicMock

from fedcast.federated_learning_strategies.fednova import (
    FedNovaStrategy,
    build_fednova_strategy,
    _normalize_updates,
    _compute_effective_steps,
)


class TestFedNovaNormalization:
    """Test FedNova normalization functions."""

    def test_normalize_updates_basic(self):
        """Test basic normalization of client updates."""
        # Create test data
        global_params = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        client_params = [
            [np.array([2.0, 3.0]), np.array([4.0, 5.0])],  # Client 1: 2 steps
            [np.array([1.5, 2.5]), np.array([3.5, 4.5])],  # Client 2: 1 step
        ]
        client_steps = [2, 1]
        
        normalized = _normalize_updates(client_params, client_steps, global_params)
        
        # Client 1: normalization factor = 2/2 = 1.0 (no change)
        # Client 2: normalization factor = 2/1 = 2.0 (double the update)
        
        # Check that normalization was applied
        assert len(normalized) == 2
        assert len(normalized[0]) == 2
        assert len(normalized[1]) == 2
        
        # Client 1 should be unchanged (normalization factor = 1.0)
        np.testing.assert_array_almost_equal(normalized[0][0], client_params[0][0])
        np.testing.assert_array_almost_equal(normalized[0][1], client_params[0][1])
        
        # Client 2 should have doubled update
        expected_client2_param1 = global_params[0] + (client_params[1][0] - global_params[0]) * 2.0
        expected_client2_param2 = global_params[1] + (client_params[1][1] - global_params[1]) * 2.0
        np.testing.assert_array_almost_equal(normalized[1][0], expected_client2_param1)
        np.testing.assert_array_almost_equal(normalized[1][1], expected_client2_param2)

    def test_normalize_updates_zero_steps(self):
        """Test normalization when client has zero steps."""
        global_params = [np.array([1.0, 2.0])]
        client_params = [[np.array([2.0, 3.0])]]
        client_steps = [0]
        
        normalized = _normalize_updates(client_params, client_steps, global_params)
        
        # Should return zero update (same as global params)
        np.testing.assert_array_almost_equal(normalized[0][0], global_params[0])

    def test_normalize_updates_empty_input(self):
        """Test normalization with empty inputs."""
        result = _normalize_updates([], [], [])
        assert result == []

    def test_compute_effective_steps(self):
        """Test computation of effective steps."""
        client_steps = [2, 4, 1]
        client_weights = [0.3, 0.5, 0.2]
        
        effective_steps = _compute_effective_steps(client_steps, client_weights)
        
        # Expected: (2*0.3 + 4*0.5 + 1*0.2) / (0.3 + 0.5 + 0.2) = 2.8
        expected = (2 * 0.3 + 4 * 0.5 + 1 * 0.2) / 1.0
        assert abs(effective_steps - expected) < 1e-6

    def test_compute_effective_steps_empty(self):
        """Test effective steps computation with empty inputs."""
        result = _compute_effective_steps([], [])
        assert result == 1.0


class TestFedNovaStrategy:
    """Test FedNova strategy implementation."""

    def test_fednova_strategy_initialization(self):
        """Test FedNova strategy initialization."""
        strategy = FedNovaStrategy()
        
        assert strategy.normalize_updates is True
        assert strategy.track_client_steps is True
        assert strategy._round == 0
        assert strategy._last_global_params is None
        assert strategy._client_steps_history == []

    def test_fednova_strategy_with_custom_base(self):
        """Test FedNova strategy with custom base strategy."""
        base_strategy = Mock()
        strategy = FedNovaStrategy(
            base_strategy=base_strategy,
            normalize_updates=False,
            track_client_steps=False
        )
        
        assert strategy.base == base_strategy
        assert strategy.normalize_updates is False
        assert strategy.track_client_steps is False

    def test_aggregate_fit_basic(self):
        """Test basic aggregate_fit functionality."""
        # Create mock base strategy
        base_strategy = Mock()
        base_strategy.aggregate_fit.return_value = (
            Mock(),  # aggregated_params
            {"base_metric": 1.0}  # metrics
        )
        
        strategy = FedNovaStrategy(base_strategy=base_strategy)
        
        # Create mock results
        mock_client = Mock()
        mock_fit_res = Mock()
        mock_fit_res.parameters = Mock()
        mock_fit_res.num_examples = 10
        mock_fit_res.metrics = {"num_steps": 2}
        
        # Mock parameters_to_ndarrays
        with pytest.MonkeyPatch().context() as m:
            m.setattr("fedcast.federated_learning_strategies.fednova.parameters_to_ndarrays", 
                     lambda x: [np.array([1.0, 2.0])])
            m.setattr("fedcast.federated_learning_strategies.fednova.ndarrays_to_parameters", 
                     lambda x: Mock())
            
            results = [(mock_client, mock_fit_res)]
            failures = []
            
            # Set initial global parameters
            strategy._last_global_params = [np.array([0.5, 1.5])]
            
            aggregated_params, metrics = strategy.aggregate_fit(1, results, failures)
            
            # Check that base strategy was called
            base_strategy.aggregate_fit.assert_called_once()
            
            # Check that FedNova metrics were added
            assert "fednova_avg_client_steps" in metrics
            assert "fednova_max_client_steps" in metrics
            assert "fednova_min_client_steps" in metrics
            assert "fednova_effective_steps" in metrics
            assert "fednova_steps_std" in metrics
            assert "fednova_normalization_enabled" in metrics
            assert "fednova_step_variance" in metrics

    def test_aggregate_fit_without_normalization(self):
        """Test aggregate_fit without normalization."""
        base_strategy = Mock()
        base_strategy.aggregate_fit.return_value = (Mock(), {"base_metric": 1.0})
        
        strategy = FedNovaStrategy(
            base_strategy=base_strategy,
            normalize_updates=False
        )
        
        mock_client = Mock()
        mock_fit_res = Mock()
        mock_fit_res.parameters = Mock()
        mock_fit_res.num_examples = 10
        mock_fit_res.metrics = {"num_steps": 2}
        
        with pytest.MonkeyPatch().context() as m:
            m.setattr("fedcast.federated_learning_strategies.fednova.parameters_to_ndarrays", 
                     lambda x: [np.array([1.0, 2.0])])
            m.setattr("fedcast.federated_learning_strategies.fednova.ndarrays_to_parameters", 
                     lambda x: Mock())
            
            results = [(mock_client, mock_fit_res)]
            failures = []
            
            aggregated_params, metrics = strategy.aggregate_fit(1, results, failures)
            
            # Check that normalization is disabled
            assert metrics["fednova_normalization_enabled"] == 0.0

    def test_get_client_steps_info(self):
        """Test getting client steps information."""
        strategy = FedNovaStrategy()
        strategy._round = 5
        strategy._client_steps_history = [[2, 4], [3, 1], [2, 3]]
        
        info = strategy.get_client_steps_info()
        
        assert info["current_round"] == 5
        assert len(info["client_steps_history"]) == 3
        assert info["latest_client_steps"] == [2, 3]
        assert info["avg_steps"] == 2.5
        assert info["steps_variance"] == 0.25

    def test_get_client_steps_info_empty(self):
        """Test getting client steps info when no history exists."""
        strategy = FedNovaStrategy()
        
        info = strategy.get_client_steps_info()
        
        assert info["client_steps_history"] == []
        assert info["current_round"] == 0


class TestBuildFedNovaStrategy:
    """Test the build_fednova_strategy function."""

    def test_build_fednova_strategy_default(self):
        """Test building FedNova strategy with default parameters."""
        strategy = build_fednova_strategy()
        
        assert isinstance(strategy, FedNovaStrategy)
        assert strategy.normalize_updates is True
        assert strategy.track_client_steps is True

    def test_build_fednova_strategy_custom(self):
        """Test building FedNova strategy with custom parameters."""
        strategy = build_fednova_strategy(
            fraction_fit=0.5,
            min_fit_clients=3,
            normalize_updates=False,
            track_client_steps=False
        )
        
        assert isinstance(strategy, FedNovaStrategy)
        assert strategy.normalize_updates is False
        assert strategy.track_client_steps is False
        
        # Check that base strategy was configured correctly
        base_strategy = strategy.base
        assert base_strategy.fraction_fit == 0.5
        assert base_strategy.min_fit_clients == 3

    def test_build_fednova_strategy_parameters(self):
        """Test that all parameters are passed correctly."""
        strategy = build_fednova_strategy(
            fraction_fit=0.8,
            fraction_evaluate=0.6,
            min_fit_clients=4,
            min_evaluate_clients=3,
            min_available_clients=5,
            normalize_updates=True,
            track_client_steps=True
        )
        
        base_strategy = strategy.base
        assert base_strategy.fraction_fit == 0.8
        assert base_strategy.fraction_evaluate == 0.6
        assert base_strategy.min_fit_clients == 4
        assert base_strategy.min_evaluate_clients == 3
        assert base_strategy.min_available_clients == 5


if __name__ == "__main__":
    pytest.main([__file__])
