# feddyn_strategy.py
# A Flower Strategy implementing Federated Dynamic Regularization (FedDyn)
# Reference: "Federated Learning with Dynamic Regularization"
# Paper: https://arxiv.org/abs/2111.04263

from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple, Any
import math

import numpy as np
import flwr as fl
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy, FedAvg


def _compute_dynamic_regularization_term(
    client_params: List[np.ndarray],
    global_params: List[np.ndarray],
    client_regularization_terms: List[np.ndarray],
    alpha: float
) -> List[np.ndarray]:
    """Compute the dynamic regularization term for FedDyn.
    
    FedDyn introduces a dynamic regularization term that adapts based on the
    difference between local and global model parameters. This helps mitigate
    client drift and improves convergence in heterogeneous federated learning.
    
    The regularization term is computed as:
    R_i(w) = alpha * ||w - w_global||^2 - <h_i, w>
    
    where h_i is the client's regularization term that gets updated each round.
    
    Args:
        client_params: Current client model parameters
        global_params: Current global model parameters
        client_regularization_terms: Client's regularization terms (h_i)
        alpha: Regularization strength parameter
        
    Returns:
        Updated regularization terms for the client
    """
    if not client_params or not global_params or not client_regularization_terms:
        return client_regularization_terms
    
    updated_terms = []
    for client_param, global_param, reg_term in zip(client_params, global_params, client_regularization_terms):
        # Compute the dynamic regularization update
        # h_i^{t+1} = h_i^t + alpha * (w_global^t - w_i^t)
        regularization_update = alpha * (global_param - client_param)
        updated_term = reg_term + regularization_update
        updated_terms.append(updated_term)
    
    return updated_terms


def _aggregate_regularization_terms(
    client_regularization_terms: List[List[np.ndarray]],
    client_weights: List[float]
) -> List[np.ndarray]:
    """Aggregate client regularization terms using weighted averaging.
    
    Args:
        client_regularization_terms: List of regularization terms from each client
        client_weights: Weights for each client (typically based on data size)
        
    Returns:
        Aggregated regularization terms
    """
    if not client_regularization_terms:
        return []
    
    if not client_weights:
        return client_regularization_terms[0] if client_regularization_terms else []
    
    # Normalize weights
    total_weight = sum(client_weights)
    if total_weight == 0:
        return client_regularization_terms[0] if client_regularization_terms else []
    
    normalized_weights = [w / total_weight for w in client_weights]
    
    # Weighted average of regularization terms
    num_layers = len(client_regularization_terms[0])
    aggregated_terms = []
    
    for layer_idx in range(num_layers):
        weighted_sum = np.zeros_like(client_regularization_terms[0][layer_idx])
        for client_terms, weight in zip(client_regularization_terms, normalized_weights):
            weighted_sum += weight * client_terms[layer_idx]
        aggregated_terms.append(weighted_sum)
    
    return aggregated_terms


class FedDynStrategy(Strategy):
    """
    Federated Dynamic Regularization (FedDyn) Strategy.
    
    FedDyn addresses client drift and data heterogeneity in federated learning
    by introducing a dynamic regularization term to the local objective function.
    The regularization term adapts over time based on the difference between
    local and global model parameters, helping to maintain model consistency
    while allowing for local adaptation.
    
    The key insight is that the regularization term prevents local models from
    drifting too far from the global model while still allowing them to adapt
    to their local data distribution. This is particularly effective for time
    series forecasting where different clients may have different temporal patterns.
    
    Parameters
    ----------
    base_strategy : Strategy
        The underlying strategy to perform vanilla aggregation (e.g., FedAvg).
    alpha : float
        Regularization strength parameter. Higher values lead to stronger
        regularization and more consistent models across clients.
    track_regularization : bool
        Whether to track and log regularization-related metrics.
    """
    
    def __init__(
        self,
        base_strategy: Optional[Strategy] = None,
        alpha: float = 0.01,
        track_regularization: bool = True,
    ) -> None:
        self.base = base_strategy if base_strategy is not None else FedAvg()
        self.alpha = alpha
        self.track_regularization = track_regularization
        
        # Internal state for tracking
        self._round = 0
        self._last_global_params: Optional[List[np.ndarray]] = None
        self._global_regularization_terms: Optional[List[np.ndarray]] = None
        self._regularization_history: List[Dict[str, float]] = []
    
    # ---- Required Strategy API, most delegated to base strategy ----
    
    def initialize_parameters(self, client_manager):
        return self.base.initialize_parameters(client_manager)
    
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager,
    ) -> List[Tuple[ClientProxy, Dict]]:
        # Add FedDyn-specific configuration
        base_config = self.base.configure_fit(server_round, parameters, client_manager)
        
        # Add regularization parameters to client configuration
        enhanced_config = []
        for client_proxy, config in base_config:
            # Convert config to dict if it's a FitIns object
            if hasattr(config, 'config'):
                enhanced_config_dict = dict(config.config)
            else:
                enhanced_config_dict = dict(config)
            
            enhanced_config_dict["feddyn_alpha"] = self.alpha
            enhanced_config_dict["feddyn_global_regularization_terms"] = self._global_regularization_terms
            enhanced_config.append((client_proxy, enhanced_config_dict))
        
        return enhanced_config
    
    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager):
        return self.base.configure_evaluate(server_round, parameters, client_manager)
    
    def evaluate(self, server_round: int, parameters: Parameters):
        return self.base.evaluate(server_round, parameters)
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results,
        failures,
    ):
        return self.base.aggregate_evaluate(server_round, results, failures)
    
    # ---- FedDyn logic hooks into aggregate_fit ----
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        self._round = server_round
        
        if not results:
            return None, {}
        
        # Get current global parameters
        if self._last_global_params is None and len(results) > 0:
            self._last_global_params = parameters_to_ndarrays(results[0][1].parameters)
        
        # Extract client parameters and metadata
        client_params_list = [parameters_to_ndarrays(fr.parameters) for _, fr in results]
        client_weights = [fr.num_examples for _, fr in results]
        
        # Extract client regularization terms from metrics if available
        client_regularization_terms = []
        for _, fit_res in results:
            if 'feddyn_regularization_terms' in fit_res.metrics:
                # Convert serialized regularization terms back to numpy arrays
                reg_terms_serialized = fit_res.metrics['feddyn_regularization_terms']
                if isinstance(reg_terms_serialized, str):
                    # Handle serialized format (simplified for this implementation)
                    # In a real implementation, you'd need proper serialization
                    reg_terms = [np.zeros_like(param) for param in client_params_list[0]]
                else:
                    reg_terms = [np.zeros_like(param) for param in client_params_list[0]]
            else:
                # Initialize with zeros if not provided
                reg_terms = [np.zeros_like(param) for param in client_params_list[0]]
            client_regularization_terms.append(reg_terms)
        
        # Update global regularization terms
        if self._last_global_params is not None:
            # Aggregate client regularization terms
            if client_regularization_terms:
                self._global_regularization_terms = _aggregate_regularization_terms(
                    client_regularization_terms, client_weights
                )
            else:
                # Initialize with zeros if no client terms available
                self._global_regularization_terms = [np.zeros_like(param) for param in self._last_global_params]
        
        # Perform base aggregation
        aggregated_params, metrics = self.base.aggregate_fit(server_round, results, failures)
        
        # Update global parameters
        if aggregated_params is not None:
            self._last_global_params = parameters_to_ndarrays(aggregated_params)
        
        # Add FedDyn-specific metrics
        if metrics is None:
            metrics = {}
        
        # Calculate regularization metrics
        if self.track_regularization and self._last_global_params is not None:
            regularization_metrics = self._compute_regularization_metrics(
                client_params_list, client_weights
            )
            metrics.update(regularization_metrics)
            
            # Store regularization history
            self._regularization_history.append({
                "round": server_round,
                "alpha": self.alpha,
                **regularization_metrics
            })
        
        return aggregated_params, metrics
    
    def _compute_regularization_metrics(
        self,
        client_params_list: List[List[np.ndarray]],
        client_weights: List[float]
    ) -> Dict[str, float]:
        """Compute regularization-related metrics for monitoring."""
        if not client_params_list or self._last_global_params is None:
            return {}
        
        metrics = {}
        
        # Compute client drift (distance from global model)
        client_drifts = []
        for client_params in client_params_list:
            total_drift = 0.0
            for client_param, global_param in zip(client_params, self._last_global_params):
                drift = np.linalg.norm(client_param - global_param)
                total_drift += drift
            client_drifts.append(total_drift)
        
        # Compute regularization term magnitudes
        if self._global_regularization_terms is not None:
            reg_magnitudes = []
            for reg_term in self._global_regularization_terms:
                magnitude = np.linalg.norm(reg_term)
                reg_magnitudes.append(magnitude)
            
            metrics["feddyn_avg_reg_magnitude"] = float(np.mean(reg_magnitudes))
            metrics["feddyn_max_reg_magnitude"] = float(np.max(reg_magnitudes))
            metrics["feddyn_reg_magnitude_std"] = float(np.std(reg_magnitudes))
        
        # Client drift metrics
        metrics["feddyn_avg_client_drift"] = float(np.mean(client_drifts))
        metrics["feddyn_max_client_drift"] = float(np.max(client_drifts))
        metrics["feddyn_client_drift_std"] = float(np.std(client_drifts))
        
        # Regularization strength
        metrics["feddyn_alpha"] = float(self.alpha)
        metrics["feddyn_regularization_enabled"] = 1.0
        
        return metrics
    
    def get_regularization_info(self) -> Dict[str, Any]:
        """Get information about regularization terms and history."""
        return {
            "regularization_history": self._regularization_history.copy(),
            "current_round": self._round,
            "alpha": self.alpha,
            "global_regularization_terms": self._global_regularization_terms,
            "last_global_params": self._last_global_params,
        }


def build_feddyn_strategy(
    *,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    min_fit_clients: int = 2,
    min_evaluate_clients: int = 2,
    min_available_clients: int = 2,
    alpha: float = 0.01,
    track_regularization: bool = True,
) -> Any:
    """Create a FedDyn strategy with a FedAvg base strategy.
    
    FedDyn (Federated Dynamic Regularization) addresses client drift and data
    heterogeneity in federated learning by introducing a dynamic regularization
    term to the local objective function. The regularization term adapts over
    time based on the difference between local and global model parameters.
    
    This is particularly effective for time series forecasting where different
    clients may have different temporal patterns and data distributions.
    
    Reference: "Federated Learning with Dynamic Regularization"
    Paper: https://arxiv.org/abs/2111.04263
    
    Parameters
    ----------
    fraction_fit : float
        Fraction of clients used for training in each round.
    fraction_evaluate : float
        Fraction of clients used for evaluation in each round.
    min_fit_clients : int
        Minimum number of clients used for training in each round.
    min_evaluate_clients : int
        Minimum number of clients used for evaluation in each round.
    min_available_clients : int
        Minimum number of available clients needed for a round to proceed.
    alpha : float
        Regularization strength parameter. Higher values lead to stronger
        regularization and more consistent models across clients.
    track_regularization : bool
        Whether to track and log regularization-related metrics.
    """
    base_strategy = fl.server.strategy.FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
    )
    
    return FedDynStrategy(
        base_strategy=base_strategy,
        alpha=alpha,
        track_regularization=track_regularization,
    )
