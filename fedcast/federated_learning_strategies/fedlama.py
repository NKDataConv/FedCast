# fedlama_strategy.py
# A Flower Strategy implementing Layer-wise Adaptive Model Aggregation (FedLAMA)
# Reference: "Layer-wise Adaptive Model Aggregation for Scalable Federated Learning"
# Paper: https://arxiv.org/abs/2110.10302

from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple, Any
from collections import defaultdict, deque
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


def _compute_layer_discrepancy(
    layer_params: List[np.ndarray], 
    global_layer: np.ndarray
) -> float:
    """Compute discrepancy between client layer parameters and global layer.
    
    Args:
        layer_params: List of layer parameters from different clients
        global_layer: Global layer parameters
        
    Returns:
        Average L2 distance between client layers and global layer
    """
    if not layer_params:
        return 0.0
    
    total_discrepancy = 0.0
    for client_layer in layer_params:
        # Compute L2 distance between client and global layer
        diff = client_layer - global_layer
        discrepancy = np.sqrt(np.sum(diff * diff))
        total_discrepancy += discrepancy
    
    return total_discrepancy / len(layer_params)


def _compute_layer_importance(
    layer_params: List[np.ndarray],
    global_layer: np.ndarray,
    layer_index: int,
    total_layers: int
) -> float:
    """Compute importance weight for a layer based on its contribution to model performance.
    
    Args:
        layer_params: List of layer parameters from different clients
        global_layer: Global layer parameters
        layer_index: Index of the current layer
        total_layers: Total number of layers in the model
        
    Returns:
        Importance weight for the layer
    """
    # Base importance decreases for deeper layers (common in neural networks)
    depth_factor = 1.0 / (1.0 + layer_index * 0.1)
    
    # Discrepancy factor - higher discrepancy means higher importance
    discrepancy = _compute_layer_discrepancy(layer_params, global_layer)
    discrepancy_factor = 1.0 + discrepancy
    
    # Combine factors
    importance = depth_factor * discrepancy_factor
    
    return importance


def _adaptive_aggregation_interval(
    discrepancy: float,
    importance: float,
    current_interval: int,
    min_interval: int = 1,
    max_interval: int = 10,
    discrepancy_threshold: float = 0.1
) -> int:
    """Determine adaptive aggregation interval for a layer.
    
    Args:
        discrepancy: Current layer discrepancy
        importance: Layer importance weight
        current_interval: Current aggregation interval
        min_interval: Minimum aggregation interval
        max_interval: Maximum aggregation interval
        discrepancy_threshold: Threshold for high discrepancy
        
    Returns:
        New aggregation interval for the layer
    """
    # If discrepancy is high, aggregate more frequently
    if discrepancy > discrepancy_threshold:
        new_interval = max(min_interval, current_interval - 1)
    else:
        # If discrepancy is low, can aggregate less frequently
        new_interval = min(max_interval, current_interval + 1)
    
    # Adjust based on importance
    if importance > 1.5:  # High importance layer
        new_interval = max(min_interval, new_interval - 1)
    
    return new_interval


class FedLAMAStrategy(Strategy):
    """
    Layer-wise Adaptive Model Aggregation (FedLAMA) Strategy.
    
    FedLAMA adaptively adjusts the aggregation interval for each layer based on:
    - Model discrepancy between clients
    - Layer importance in the model
    - Communication cost constraints
    
    This strategy is particularly effective for time series forecasting where
    different layers may have different convergence patterns and importance.
    
    Parameters
    ----------
    base_strategy : Strategy
        The underlying strategy to perform vanilla aggregation (e.g., FedAvg).
    min_aggregation_interval : int
        Minimum rounds between layer aggregations.
    max_aggregation_interval : int
        Maximum rounds between layer aggregations.
    discrepancy_threshold : float
        Threshold for determining high vs low discrepancy.
    communication_budget : float
        Maximum communication cost per round (0.0 = no limit).
    layer_importance_decay : float
        Decay factor for layer importance based on depth.
    """

    def __init__(
        self,
        base_strategy: Optional[Strategy] = None,
        min_aggregation_interval: int = 1,
        max_aggregation_interval: int = 10,
        discrepancy_threshold: float = 0.1,
        communication_budget: float = 0.0,
        layer_importance_decay: float = 0.1,
    ) -> None:
        self.base = base_strategy if base_strategy is not None else FedAvg()
        self.min_interval = max(1, min_aggregation_interval)
        self.max_interval = max(min_aggregation_interval, max_aggregation_interval)
        self.discrepancy_threshold = float(discrepancy_threshold)
        self.communication_budget = float(communication_budget)
        self.layer_importance_decay = float(layer_importance_decay)

        # Internal state for layer-wise tracking
        self._round = 0
        self._layer_aggregation_intervals: List[int] = []
        self._layer_last_aggregated: List[int] = []
        self._layer_discrepancy_history: List[deque] = []
        self._layer_importance_history: List[deque] = []
        self._last_global_params: Optional[List[np.ndarray]] = None
        self._total_layers = 0

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

    # ---- FedLAMA logic hooks into aggregate_fit ----

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        self._round = server_round

        # Initialize layer tracking on first round
        if self._total_layers == 0 and len(results) > 0:
            first_client_params = parameters_to_ndarrays(results[0][1].parameters)
            self._total_layers = len(first_client_params)
            self._layer_aggregation_intervals = [self.min_interval] * self._total_layers
            self._layer_last_aggregated = [0] * self._total_layers
            self._layer_discrepancy_history = [deque(maxlen=10) for _ in range(self._total_layers)]
            self._layer_importance_history = [deque(maxlen=10) for _ in range(self._total_layers)]

        # Get current global parameters
        if self._last_global_params is None and len(results) > 0:
            self._last_global_params = parameters_to_ndarrays(results[0][1].parameters)

        # Collect client parameters
        client_params_list = [parameters_to_ndarrays(fr.parameters) for _, fr in results]

        # Compute layer-wise discrepancies and importance
        layer_discrepancies = []
        layer_importances = []
        
        for layer_idx in range(self._total_layers):
            client_layers = [params[layer_idx] for params in client_params_list]
            global_layer = self._last_global_params[layer_idx]
            
            # Compute discrepancy and importance
            discrepancy = _compute_layer_discrepancy(client_layers, global_layer)
            importance = _compute_layer_importance(
                client_layers, global_layer, layer_idx, self._total_layers
            )
            
            layer_discrepancies.append(discrepancy)
            layer_importances.append(importance)
            
            # Update history
            self._layer_discrepancy_history[layer_idx].append(discrepancy)
            self._layer_importance_history[layer_idx].append(importance)

        # Determine which layers to aggregate
        layers_to_aggregate = []
        for layer_idx in range(self._total_layers):
            rounds_since_last = server_round - self._layer_last_aggregated[layer_idx]
            current_interval = self._layer_aggregation_intervals[layer_idx]
            
            # Check if it's time to aggregate this layer
            if rounds_since_last >= current_interval:
                layers_to_aggregate.append(layer_idx)
                self._layer_last_aggregated[layer_idx] = server_round
                
                # Update aggregation interval based on current discrepancy
                new_interval = _adaptive_aggregation_interval(
                    layer_discrepancies[layer_idx],
                    layer_importances[layer_idx],
                    current_interval,
                    self.min_interval,
                    self.max_interval,
                    self.discrepancy_threshold
                )
                self._layer_aggregation_intervals[layer_idx] = new_interval

        # Perform selective aggregation
        if layers_to_aggregate:
            # Create modified client results with only selected layers
            modified_results = []
            for client_proxy, fit_res in results:
                client_params = parameters_to_ndarrays(fit_res.parameters)
                
                # Create new parameters with only aggregated layers updated
                new_params = []
                for layer_idx in range(self._total_layers):
                    if layer_idx in layers_to_aggregate:
                        # Use client's updated parameters
                        new_params.append(client_params[layer_idx])
                    else:
                        # Keep global parameters for non-aggregated layers
                        new_params.append(self._last_global_params[layer_idx])
                
                # Convert back to Parameters object
                new_parameters = ndarrays_to_parameters(new_params)
                
                # Create modified FitRes
                modified_fit_res = FitRes(
                    status=fit_res.status,
                    parameters=new_parameters,
                    num_examples=fit_res.num_examples,
                    metrics=fit_res.metrics,
                )
                modified_results.append((client_proxy, modified_fit_res))
            
            # Perform base aggregation
            base_agg = self.base.aggregate_fit(server_round, modified_results, failures)
        else:
            # No layers to aggregate, return current global parameters
            base_agg = (ndarrays_to_parameters(self._last_global_params), {})

        aggregated_params, metrics = base_agg

        if aggregated_params is not None:
            self._last_global_params = parameters_to_ndarrays(aggregated_params)

        # Add FedLAMA-specific metrics
        if metrics is None:
            metrics = {}
        
        metrics["fedlama_layers_aggregated"] = len(layers_to_aggregate)
        metrics["fedlama_avg_discrepancy"] = float(np.mean(layer_discrepancies))
        metrics["fedlama_avg_importance"] = float(np.mean(layer_importances))
        metrics["fedlama_avg_interval"] = float(np.mean(self._layer_aggregation_intervals))
        
        # Communication efficiency metric
        if self._total_layers > 0:
            communication_ratio = len(layers_to_aggregate) / self._total_layers
            metrics["fedlama_communication_ratio"] = communication_ratio

        return aggregated_params, metrics

    def get_layer_aggregation_info(self) -> Dict[str, Any]:
        """Get information about current layer aggregation state."""
        return {
            "layer_intervals": self._layer_aggregation_intervals.copy(),
            "layer_last_aggregated": self._layer_last_aggregated.copy(),
            "total_layers": self._total_layers,
            "current_round": self._round,
        }


def build_fedlama_strategy(
    *,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    min_fit_clients: int = 2,
    min_evaluate_clients: int = 2,
    min_available_clients: int = 2,
    min_aggregation_interval: int = 1,
    max_aggregation_interval: int = 10,
    discrepancy_threshold: float = 0.1,
    communication_budget: float = 0.0,
    layer_importance_decay: float = 0.1,
) -> Any:
    """Create a FedLAMA strategy with a FedAvg base strategy.
    
    FedLAMA (Layer-wise Adaptive Model Aggregation) is a communication-efficient
    federated learning strategy that adaptively adjusts aggregation intervals
    for each layer based on model discrepancy and importance. This is particularly
    effective for time series forecasting where different layers may converge
    at different rates.
    
    Reference: "Layer-wise Adaptive Model Aggregation for Scalable Federated Learning"
    Paper: https://arxiv.org/abs/2110.10302
    
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
    min_aggregation_interval : int
        Minimum rounds between layer aggregations.
    max_aggregation_interval : int
        Maximum rounds between layer aggregations.
    discrepancy_threshold : float
        Threshold for determining high vs low discrepancy.
    communication_budget : float
        Maximum communication cost per round (0.0 = no limit).
    layer_importance_decay : float
        Decay factor for layer importance based on depth.
    """
    base_strategy = fl.server.strategy.FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
    )
    
    return FedLAMAStrategy(
        base_strategy=base_strategy,
        min_aggregation_interval=min_aggregation_interval,
        max_aggregation_interval=max_aggregation_interval,
        discrepancy_threshold=discrepancy_threshold,
        communication_budget=communication_budget,
        layer_importance_decay=layer_importance_decay,
    )
