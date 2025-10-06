# fednova_strategy.py
# A Flower Strategy implementing Normalized Averaging (FedNova)
# Reference: "Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization"
# Paper: https://arxiv.org/abs/2007.07481

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


def _normalize_updates(
    client_params: List[List[np.ndarray]], 
    client_steps: List[int],
    global_params: List[np.ndarray]
) -> List[List[np.ndarray]]:
    """Normalize client updates by their local training steps.
    
    FedNova normalizes local model updates before aggregation to address
    objective inconsistency caused by varying local update steps among clients.
    
    Args:
        client_params: List of client parameter updates
        client_steps: List of local training steps for each client
        global_params: Current global model parameters
        
    Returns:
        List of normalized client parameter updates
    """
    if not client_params or not client_steps:
        return client_params
    
    # Calculate the maximum number of steps across all clients
    max_steps = max(client_steps)
    
    # Normalize each client's updates
    normalized_params = []
    for i, (params, steps) in enumerate(zip(client_params, client_steps)):
        if steps == 0:
            # If client didn't train, return global parameters (no update)
            normalized_params.append(global_params.copy())
            continue
            
        # Calculate normalization factor
        # For FedNova, we normalize by the ratio of max_steps to client_steps
        normalization_factor = max_steps / steps
        
        # Apply normalization to each parameter
        normalized_client_params = []
        for param, global_param in zip(params, global_params):
            # Calculate the update (difference from global)
            update = param - global_param
            # Normalize the update
            normalized_update = update * normalization_factor
            # Add normalized update to global parameters
            normalized_param = global_param + normalized_update
            normalized_client_params.append(normalized_param)
        
        normalized_params.append(normalized_client_params)
    
    return normalized_params


def _compute_effective_steps(
    client_steps: List[int],
    client_weights: List[float]
) -> float:
    """Compute the effective number of steps for aggregation.
    
    Args:
        client_steps: List of local training steps for each client
        client_weights: List of aggregation weights for each client
        
    Returns:
        Effective number of steps
    """
    if not client_steps or not client_weights:
        return 1.0
    
    # Weighted average of client steps
    total_weighted_steps = sum(steps * weight for steps, weight in zip(client_steps, client_weights))
    total_weight = sum(client_weights)
    
    if total_weight == 0:
        return 1.0
    
    return total_weighted_steps / total_weight


class FedNovaStrategy(Strategy):
    """
    Normalized Averaging (FedNova) Strategy.
    
    FedNova addresses the objective inconsistency problem in heterogeneous federated
    optimization by normalizing local model updates before aggregation. This ensures
    that each client's contribution is appropriately scaled based on their local
    training effort, leading to more stable and fair convergence.
    
    The key insight is that clients performing different numbers of local updates
    should have their contributions normalized to ensure consistent global model
    updates. This is particularly important in time series forecasting where
    different clients may have varying amounts of training data.
    
    Parameters
    ----------
    base_strategy : Strategy
        The underlying strategy to perform vanilla aggregation (e.g., FedAvg).
    normalize_updates : bool
        Whether to apply FedNova normalization to client updates.
    track_client_steps : bool
        Whether to track and use client training steps for normalization.
    """

    def __init__(
        self,
        base_strategy: Optional[Strategy] = None,
        normalize_updates: bool = True,
        track_client_steps: bool = True,
    ) -> None:
        self.base = base_strategy if base_strategy is not None else FedAvg()
        self.normalize_updates = normalize_updates
        self.track_client_steps = track_client_steps

        # Internal state for tracking
        self._round = 0
        self._last_global_params: Optional[List[np.ndarray]] = None
        self._client_steps_history: List[List[int]] = []

    # ---- Required Strategy API, most delegated to base strategy ----

    def initialize_parameters(self, client_manager):
        return self.base.initialize_parameters(client_manager)

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager,
    ) -> List[Tuple[ClientProxy, Dict]]:
        return self.base.configure_fit(server_round, parameters, client_manager)

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

    # ---- FedNova logic hooks into aggregate_fit ----

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
        
        # Extract client training steps from metrics if available
        client_steps = []
        for _, fit_res in results:
            if self.track_client_steps and 'num_steps' in fit_res.metrics:
                client_steps.append(int(fit_res.metrics['num_steps']))
            else:
                # Default to 1 step if not tracked
                client_steps.append(1)

        # Apply FedNova normalization if enabled
        if self.normalize_updates and self._last_global_params is not None:
            normalized_params = _normalize_updates(
                client_params_list, 
                client_steps, 
                self._last_global_params
            )
        else:
            normalized_params = client_params_list

        # Create modified results with normalized parameters
        modified_results = []
        for i, ((client_proxy, fit_res), normalized_params) in enumerate(zip(results, normalized_params)):
            # Convert normalized parameters back to Parameters object
            normalized_parameters = ndarrays_to_parameters(normalized_params)
            
            # Create modified FitRes with normalized parameters
            modified_fit_res = FitRes(
                status=fit_res.status,
                parameters=normalized_parameters,
                num_examples=fit_res.num_examples,
                metrics=fit_res.metrics,
            )
            modified_results.append((client_proxy, modified_fit_res))

        # Perform base aggregation with normalized parameters
        aggregated_params, metrics = self.base.aggregate_fit(server_round, modified_results, failures)

        # Update global parameters
        if aggregated_params is not None:
            self._last_global_params = parameters_to_ndarrays(aggregated_params)

        # Store client steps history
        self._client_steps_history.append(client_steps.copy())

        # Add FedNova-specific metrics
        if metrics is None:
            metrics = {}
        
        # Calculate effective steps
        effective_steps = _compute_effective_steps(client_steps, client_weights)
        
        # Add metrics
        metrics["fednova_avg_client_steps"] = float(np.mean(client_steps))
        metrics["fednova_max_client_steps"] = float(np.max(client_steps))
        metrics["fednova_min_client_steps"] = float(np.min(client_steps))
        metrics["fednova_effective_steps"] = float(effective_steps)
        metrics["fednova_steps_std"] = float(np.std(client_steps))
        
        # Normalization metrics
        if self.normalize_updates:
            metrics["fednova_normalization_enabled"] = 1.0
            # Calculate step variance (higher variance means more benefit from normalization)
            step_variance = np.var(client_steps)
            metrics["fednova_step_variance"] = float(step_variance)
        else:
            metrics["fednova_normalization_enabled"] = 0.0

        return aggregated_params, metrics

    def get_client_steps_info(self) -> Dict[str, Any]:
        """Get information about client training steps."""
        if not self._client_steps_history:
            return {"client_steps_history": [], "current_round": self._round}
        
        latest_steps = self._client_steps_history[-1]
        return {
            "client_steps_history": self._client_steps_history.copy(),
            "latest_client_steps": latest_steps,
            "current_round": self._round,
            "avg_steps": float(np.mean(latest_steps)),
            "steps_variance": float(np.var(latest_steps)),
        }


def build_fednova_strategy(
    *,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    min_fit_clients: int = 2,
    min_evaluate_clients: int = 2,
    min_available_clients: int = 2,
    normalize_updates: bool = True,
    track_client_steps: bool = True,
) -> Any:
    """Create a FedNova strategy with a FedAvg base strategy.
    
    FedNova (Normalized Averaging) addresses the objective inconsistency problem
    in heterogeneous federated optimization by normalizing local model updates
    before aggregation. This ensures that each client's contribution is
    appropriately scaled based on their local training effort.
    
    This is particularly effective for time series forecasting where different
    clients may have varying amounts of training data and perform different
    numbers of local updates.
    
    Reference: "Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization"
    Paper: https://arxiv.org/abs/2007.07481
    
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
    normalize_updates : bool
        Whether to apply FedNova normalization to client updates.
    track_client_steps : bool
        Whether to track and use client training steps for normalization.
    """
    base_strategy = fl.server.strategy.FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
    )
    
    return FedNovaStrategy(
        base_strategy=base_strategy,
        normalize_updates=normalize_updates,
        track_client_steps=track_client_steps,
    )
