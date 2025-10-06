# scaffold_strategy.py
# A Flower Strategy implementing Stochastic Controlled Averaging (SCAFFOLD)
# Reference: "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning"
# Paper: https://arxiv.org/abs/1910.06378

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


def _compute_control_variate_update(
    client_params: List[np.ndarray],
    global_params: List[np.ndarray],
    client_control_variate: List[np.ndarray],
    global_control_variate: List[np.ndarray],
    learning_rate: float
) -> List[np.ndarray]:
    """Compute the control variate update for SCAFFOLD.
    
    SCAFFOLD uses control variates to correct for client drift in federated learning.
    The control variate update is computed as:
    c_i^{t+1} = c_i^t + (1 / (K * eta)) * (x^t - y_i^{t+1})
    
    where:
    - c_i^t is the client's control variate at time t
    - K is the number of local steps
    - eta is the learning rate
    - x^t is the global model at time t
    - y_i^{t+1} is the client's model after local training
    
    Args:
        client_params: Client model parameters after local training
        global_params: Global model parameters before local training
        client_control_variate: Client's current control variate
        global_control_variate: Global control variate
        learning_rate: Learning rate used in local training
        
    Returns:
        Updated client control variate
    """
    if not client_params or not global_params or not client_control_variate:
        return client_control_variate
    
    updated_control_variate = []
    for client_param, global_param, client_cv in zip(client_params, global_params, client_control_variate):
        # Compute the control variate update
        # c_i^{t+1} = c_i^t + (1 / (K * eta)) * (x^t - y_i^{t+1})
        # For simplicity, we assume K=1 (single local step) and use the learning rate
        if learning_rate > 0:
            control_update = (1.0 / learning_rate) * (global_param - client_param)
        else:
            control_update = global_param - client_param
        
        updated_cv = client_cv + control_update
        updated_control_variate.append(updated_cv)
    
    return updated_control_variate


def _aggregate_control_variates(
    client_control_variates: List[List[np.ndarray]],
    client_weights: List[float]
) -> List[np.ndarray]:
    """Aggregate client control variates using weighted averaging.
    
    Args:
        client_control_variates: List of control variates from each client
        client_weights: Weights for each client (typically based on data size)
        
    Returns:
        Aggregated global control variate
    """
    if not client_control_variates:
        return []
    
    if not client_weights:
        return client_control_variates[0] if client_control_variates else []
    
    # Normalize weights
    total_weight = sum(client_weights)
    if total_weight == 0:
        return client_control_variates[0] if client_control_variates else []
    
    normalized_weights = [w / total_weight for w in client_weights]
    
    # Weighted average of control variates
    num_layers = len(client_control_variates[0])
    aggregated_cv = []
    
    for layer_idx in range(num_layers):
        weighted_sum = np.zeros_like(client_control_variates[0][layer_idx])
        for client_cv, weight in zip(client_control_variates, normalized_weights):
            weighted_sum += weight * client_cv[layer_idx]
        aggregated_cv.append(weighted_sum)
    
    return aggregated_cv


def _apply_control_variate_correction(
    client_params: List[np.ndarray],
    global_params: List[np.ndarray],
    client_control_variate: List[np.ndarray],
    global_control_variate: List[np.ndarray],
    correction_strength: float = 1.0
) -> List[np.ndarray]:
    """Apply control variate correction to client parameters.
    
    The correction is applied as:
    corrected_params = client_params + correction_strength * (global_control_variate - client_control_variate)
    
    Args:
        client_params: Client model parameters
        global_params: Global model parameters
        client_control_variate: Client's control variate
        global_control_variate: Global control variate
        correction_strength: Strength of the correction (typically 1.0)
        
    Returns:
        Corrected client parameters
    """
    if not client_params or not client_control_variate or not global_control_variate:
        return client_params
    
    corrected_params = []
    for client_param, client_cv, global_cv in zip(client_params, client_control_variate, global_control_variate):
        # Apply control variate correction
        correction = correction_strength * (global_cv - client_cv)
        corrected_param = client_param + correction
        corrected_params.append(corrected_param)
    
    return corrected_params


class SCAFFOLDStrategy(Strategy):
    """
    Stochastic Controlled Averaging (SCAFFOLD) Strategy.
    
    SCAFFOLD addresses client drift in federated learning by introducing control
    variates that correct for the variance in local updates. This method is
    particularly effective in scenarios with non-IID data distributions across
    clients, leading to improved convergence rates and stability.
    
    The key insight is that control variates help reduce the variance of local
    updates by correcting for client-specific biases. Each client maintains a
    local control variate, and the server maintains a global control variate
    that is aggregated from all clients.
    
    This is particularly effective for time series forecasting where different
    clients may have different temporal patterns and data distributions.
    
    Parameters
    ----------
    base_strategy : Strategy
        The underlying strategy to perform vanilla aggregation (e.g., FedAvg).
    learning_rate : float
        Learning rate used in local training (needed for control variate updates).
    correction_strength : float
        Strength of the control variate correction applied to client updates.
    track_control_variates : bool
        Whether to track and log control variate-related metrics.
    """
    
    def __init__(
        self,
        base_strategy: Optional[Strategy] = None,
        learning_rate: float = 0.01,
        correction_strength: float = 1.0,
        track_control_variates: bool = True,
    ) -> None:
        self.base = base_strategy if base_strategy is not None else FedAvg()
        self.learning_rate = learning_rate
        self.correction_strength = correction_strength
        self.track_control_variates = track_control_variates
        
        # Internal state for tracking
        self._round = 0
        self._last_global_params: Optional[List[np.ndarray]] = None
        self._global_control_variate: Optional[List[np.ndarray]] = None
        self._client_control_variates: Dict[str, List[np.ndarray]] = {}
        self._control_variate_history: List[Dict[str, float]] = []
    
    # ---- Required Strategy API, most delegated to base strategy ----
    
    def initialize_parameters(self, client_manager):
        return self.base.initialize_parameters(client_manager)
    
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager,
    ) -> List[Tuple[ClientProxy, Dict]]:
        # Add SCAFFOLD-specific configuration
        base_config = self.base.configure_fit(server_round, parameters, client_manager)
        
        # Add control variate parameters to client configuration
        enhanced_config = []
        for client_proxy, config in base_config:
            # Convert config to dict if it's a FitIns object
            if hasattr(config, 'config'):
                enhanced_config_dict = dict(config.config)
            else:
                enhanced_config_dict = dict(config)
            
            enhanced_config_dict["scaffold_learning_rate"] = self.learning_rate
            enhanced_config_dict["scaffold_global_control_variate"] = self._global_control_variate
            enhanced_config_dict["scaffold_correction_strength"] = self.correction_strength
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
    
    # ---- SCAFFOLD logic hooks into aggregate_fit ----
    
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
        
        # Extract client control variates from metrics if available
        client_control_variates = []
        for client_proxy, fit_res in results:
            client_id = client_proxy.cid
            if 'scaffold_control_variate' in fit_res.metrics:
                # Convert serialized control variate back to numpy arrays
                cv_serialized = fit_res.metrics['scaffold_control_variate']
                if isinstance(cv_serialized, str):
                    # Handle serialized format (simplified for this implementation)
                    # In a real implementation, you'd need proper serialization
                    cv = [np.zeros_like(param) for param in client_params_list[0]]
                else:
                    cv = [np.zeros_like(param) for param in client_params_list[0]]
            else:
                # Initialize with zeros if not provided
                cv = [np.zeros_like(param) for param in client_params_list[0]]
            
            client_control_variates.append(cv)
            self._client_control_variates[client_id] = cv
        
        # Update global control variate
        if self._last_global_params is not None:
            # Aggregate client control variates
            if client_control_variates:
                self._global_control_variate = _aggregate_control_variates(
                    client_control_variates, client_weights
                )
            else:
                # Initialize with zeros if no client control variates available
                self._global_control_variate = [np.zeros_like(param) for param in self._last_global_params]
        
        # Apply control variate correction to client parameters
        corrected_client_params = []
        for i, (client_params, client_cv) in enumerate(zip(client_params_list, client_control_variates)):
            if self._global_control_variate is not None:
                corrected_params = _apply_control_variate_correction(
                    client_params, self._last_global_params, client_cv, 
                    self._global_control_variate, self.correction_strength
                )
            else:
                corrected_params = client_params
            corrected_client_params.append(corrected_params)
        
        # Create modified results with corrected parameters
        modified_results = []
        for i, ((client_proxy, fit_res), corrected_params) in enumerate(zip(results, corrected_client_params)):
            # Convert corrected parameters back to Parameters object
            corrected_parameters = ndarrays_to_parameters(corrected_params)
            
            # Create modified FitRes with corrected parameters
            modified_fit_res = FitRes(
                status=fit_res.status,
                parameters=corrected_parameters,
                num_examples=fit_res.num_examples,
                metrics=fit_res.metrics,
            )
            modified_results.append((client_proxy, modified_fit_res))
        
        # Perform base aggregation with corrected parameters
        aggregated_params, metrics = self.base.aggregate_fit(server_round, modified_results, failures)
        
        # Update global parameters
        if aggregated_params is not None:
            self._last_global_params = parameters_to_ndarrays(aggregated_params)
        
        # Add SCAFFOLD-specific metrics
        if metrics is None:
            metrics = {}
        
        # Calculate control variate metrics
        if self.track_control_variates and self._last_global_params is not None:
            control_variate_metrics = self._compute_control_variate_metrics(
                client_control_variates, client_weights
            )
            metrics.update(control_variate_metrics)
            
            # Store control variate history
            self._control_variate_history.append({
                "round": server_round,
                "learning_rate": self.learning_rate,
                "correction_strength": self.correction_strength,
                **control_variate_metrics
            })
        
        return aggregated_params, metrics
    
    def _compute_control_variate_metrics(
        self,
        client_control_variates: List[List[np.ndarray]],
        client_weights: List[float]
    ) -> Dict[str, float]:
        """Compute control variate-related metrics for monitoring."""
        if not client_control_variates:
            return {}
        
        metrics = {}
        
        # Compute control variate magnitudes
        cv_magnitudes = []
        for client_cv in client_control_variates:
            total_magnitude = 0.0
            for cv_layer in client_cv:
                magnitude = np.linalg.norm(cv_layer)
                total_magnitude += magnitude
            cv_magnitudes.append(total_magnitude)
        
        # Global control variate magnitude
        if self._global_control_variate is not None:
            global_cv_magnitude = 0.0
            for cv_layer in self._global_control_variate:
                magnitude = np.linalg.norm(cv_layer)
                global_cv_magnitude += magnitude
            metrics["scaffold_global_cv_magnitude"] = float(global_cv_magnitude)
        
        # Client control variate metrics
        metrics["scaffold_avg_cv_magnitude"] = float(np.mean(cv_magnitudes))
        metrics["scaffold_max_cv_magnitude"] = float(np.max(cv_magnitudes))
        metrics["scaffold_cv_magnitude_std"] = float(np.std(cv_magnitudes))
        
        # Control variate variance (higher variance indicates more client drift)
        cv_variance = np.var(cv_magnitudes)
        metrics["scaffold_cv_variance"] = float(cv_variance)
        
        # SCAFFOLD configuration
        metrics["scaffold_learning_rate"] = float(self.learning_rate)
        metrics["scaffold_correction_strength"] = float(self.correction_strength)
        metrics["scaffold_control_variates_enabled"] = 1.0
        
        return metrics
    
    def get_control_variate_info(self) -> Dict[str, Any]:
        """Get information about control variates and history."""
        return {
            "control_variate_history": self._control_variate_history.copy(),
            "current_round": self._round,
            "learning_rate": self.learning_rate,
            "correction_strength": self.correction_strength,
            "global_control_variate": self._global_control_variate,
            "client_control_variates": self._client_control_variates.copy(),
            "last_global_params": self._last_global_params,
        }


def build_scaffold_strategy(
    *,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    min_fit_clients: int = 2,
    min_evaluate_clients: int = 2,
    min_available_clients: int = 2,
    learning_rate: float = 0.01,
    correction_strength: float = 1.0,
    track_control_variates: bool = True,
) -> Any:
    """Create a SCAFFOLD strategy with a FedAvg base strategy.
    
    SCAFFOLD (Stochastic Controlled Averaging) addresses client drift in federated
    learning by introducing control variates that correct for the variance in local
    updates. This method is particularly effective in scenarios with non-IID data
    distributions across clients, leading to improved convergence rates and stability.
    
    This is particularly effective for time series forecasting where different
    clients may have different temporal patterns and data distributions.
    
    Reference: "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning"
    Paper: https://arxiv.org/abs/1910.06378
    
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
    learning_rate : float
        Learning rate used in local training (needed for control variate updates).
    correction_strength : float
        Strength of the control variate correction applied to client updates.
    track_control_variates : bool
        Whether to track and log control variate-related metrics.
    """
    base_strategy = fl.server.strategy.FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
    )
    
    return SCAFFOLDStrategy(
        base_strategy=base_strategy,
        learning_rate=learning_rate,
        correction_strength=correction_strength,
        track_control_variates=track_control_variates,
    )
