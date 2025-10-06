# fedopt_strategy.py
# A Flower Strategy implementing Federated Optimization (FedOpt)
# Reference: "Adaptive Federated Optimization"
# Paper: https://arxiv.org/abs/2003.00295

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


def _compute_fedadam_update(
    aggregated_gradients: List[np.ndarray],
    server_momentum: List[np.ndarray],
    server_variance: List[np.ndarray],
    server_learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    tau: float = 1e-3
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Compute FedAdam server-side update.
    
    FedAdam applies Adam-like adaptive optimization at the server level:
    m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
    theta_t = theta_{t-1} - server_lr * m_t / (sqrt(v_t) + epsilon)
    
    Args:
        aggregated_gradients: Aggregated gradients from clients
        server_momentum: Server momentum terms
        server_variance: Server variance terms
        server_learning_rate: Server learning rate
        beta1: First moment decay rate
        beta2: Second moment decay rate
        epsilon: Small constant for numerical stability
        tau: Server momentum parameter
        
    Returns:
        Tuple of (updated_parameters, updated_momentum, updated_variance)
    """
    updated_momentum = []
    updated_variance = []
    updated_params = []
    
    for grad, momentum, variance in zip(aggregated_gradients, server_momentum, server_variance):
        # Update momentum: m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        new_momentum = beta1 * momentum + (1 - beta1) * grad
        
        # Update variance: v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
        new_variance = beta2 * variance + (1 - beta2) * (grad ** 2)
        
        # Compute bias-corrected estimates
        momentum_hat = new_momentum / (1 - beta1)
        variance_hat = new_variance / (1 - beta2)
        
        # Apply update: theta_t = theta_{t-1} - server_lr * m_t / (sqrt(v_t) + epsilon)
        param_update = server_learning_rate * momentum_hat / (np.sqrt(variance_hat) + epsilon)
        
        updated_momentum.append(new_momentum)
        updated_variance.append(new_variance)
        updated_params.append(param_update)
    
    return updated_params, updated_momentum, updated_variance


def _compute_fedadagrad_update(
    aggregated_gradients: List[np.ndarray],
    server_variance: List[np.ndarray],
    server_learning_rate: float,
    epsilon: float = 1e-8
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Compute FedAdaGrad server-side update.
    
    FedAdaGrad applies AdaGrad-like adaptive optimization at the server level:
    v_t = v_{t-1} + g_t^2
    theta_t = theta_{t-1} - server_lr * g_t / (sqrt(v_t) + epsilon)
    
    Args:
        aggregated_gradients: Aggregated gradients from clients
        server_variance: Server variance terms
        server_learning_rate: Server learning rate
        epsilon: Small constant for numerical stability
        
    Returns:
        Tuple of (updated_parameters, updated_variance)
    """
    updated_variance = []
    updated_params = []
    
    for grad, variance in zip(aggregated_gradients, server_variance):
        # Update variance: v_t = v_{t-1} + g_t^2
        new_variance = variance + (grad ** 2)
        
        # Apply update: theta_t = theta_{t-1} - server_lr * g_t / (sqrt(v_t) + epsilon)
        param_update = server_learning_rate * grad / (np.sqrt(new_variance) + epsilon)
        
        updated_variance.append(new_variance)
        updated_params.append(param_update)
    
    return updated_params, updated_variance


def _compute_fedyogi_update(
    aggregated_gradients: List[np.ndarray],
    server_momentum: List[np.ndarray],
    server_variance: List[np.ndarray],
    server_learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-3
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Compute FedYogi server-side update.
    
    FedYogi applies Yogi-like adaptive optimization at the server level:
    m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    v_t = v_{t-1} - (1 - beta2) * sign(v_{t-1} - g_t^2) * g_t^2
    theta_t = theta_{t-1} - server_lr * m_t / (sqrt(v_t) + epsilon)
    
    Args:
        aggregated_gradients: Aggregated gradients from clients
        server_momentum: Server momentum terms
        server_variance: Server variance terms
        server_learning_rate: Server learning rate
        beta1: First moment decay rate
        beta2: Second moment decay rate
        epsilon: Small constant for numerical stability
        
    Returns:
        Tuple of (updated_parameters, updated_momentum, updated_variance)
    """
    updated_momentum = []
    updated_variance = []
    updated_params = []
    
    for grad, momentum, variance in zip(aggregated_gradients, server_momentum, server_variance):
        # Update momentum: m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        new_momentum = beta1 * momentum + (1 - beta1) * grad
        
        # Update variance: v_t = v_{t-1} - (1 - beta2) * sign(v_{t-1} - g_t^2) * g_t^2
        grad_squared = grad ** 2
        variance_diff = variance - grad_squared
        sign_variance_diff = np.sign(variance_diff)
        new_variance = variance - (1 - beta2) * sign_variance_diff * grad_squared
        
        # Compute bias-corrected momentum
        momentum_hat = new_momentum / (1 - beta1)
        
        # Apply update: theta_t = theta_{t-1} - server_lr * m_t / (sqrt(v_t) + epsilon)
        param_update = server_learning_rate * momentum_hat / (np.sqrt(new_variance) + epsilon)
        
        updated_momentum.append(new_momentum)
        updated_variance.append(new_variance)
        updated_params.append(param_update)
    
    return updated_params, updated_momentum, updated_variance


class FedOptStrategy(Strategy):
    """
    Federated Optimization (FedOpt) Strategy.
    
    FedOpt enhances the standard federated averaging by incorporating adaptive
    optimization techniques at the server level, such as Adam, AdaGrad, and Yogi.
    This approach improves convergence rates and performance, especially in
    heterogeneous data environments where different clients may have varying
    data distributions and training dynamics.
    
    The key insight is that server-side adaptive optimization can help the
    global model converge faster and more stably by adjusting the learning
    rate and momentum based on the aggregated gradients from clients.
    
    This is particularly effective for time series forecasting where different
    clients may have different temporal patterns and convergence requirements.
    
    Parameters
    ----------
    base_strategy : Strategy
        The underlying strategy to perform vanilla aggregation (e.g., FedAvg).
    optimizer_type : str
        Type of adaptive optimizer to use ('adam', 'adagrad', 'yogi').
    server_learning_rate : float
        Learning rate for server-side adaptive optimization.
    beta1 : float
        First moment decay rate (for Adam and Yogi).
    beta2 : float
        Second moment decay rate (for Adam and Yogi).
    epsilon : float
        Small constant for numerical stability.
    track_optimizer_state : bool
        Whether to track and log optimizer state metrics.
    """
    
    def __init__(
        self,
        base_strategy: Optional[Strategy] = None,
        optimizer_type: str = "adam",
        server_learning_rate: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        track_optimizer_state: bool = True,
    ) -> None:
        self.base = base_strategy if base_strategy is not None else FedAvg()
        self.optimizer_type = optimizer_type.lower()
        self.server_learning_rate = server_learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.track_optimizer_state = track_optimizer_state
        
        # Validate optimizer type
        if self.optimizer_type not in ["adam", "adagrad", "yogi"]:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}. Must be one of: adam, adagrad, yogi")
        
        # Internal state for tracking
        self._round = 0
        self._last_global_params: Optional[List[np.ndarray]] = None
        self._server_momentum: Optional[List[np.ndarray]] = None
        self._server_variance: Optional[List[np.ndarray]] = None
        self._optimizer_history: List[Dict[str, float]] = []
    
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
    
    # ---- FedOpt logic hooks into aggregate_fit ----
    
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
        
        # Perform base aggregation to get aggregated parameters
        aggregated_params, metrics = self.base.aggregate_fit(server_round, results, failures)
        
        if aggregated_params is None:
            return None, {}
        
        # Convert aggregated parameters to numpy arrays
        aggregated_params_ndarrays = parameters_to_ndarrays(aggregated_params)
        
        # Compute gradients (difference between aggregated and previous global parameters)
        if self._last_global_params is not None:
            gradients = []
            for agg_param, prev_param in zip(aggregated_params_ndarrays, self._last_global_params):
                gradient = agg_param - prev_param
                gradients.append(gradient)
        else:
            # First round: no gradients available
            gradients = [np.zeros_like(param) for param in aggregated_params_ndarrays]
        
        # Initialize optimizer state if needed
        if self._server_momentum is None:
            self._server_momentum = [np.zeros_like(grad) for grad in gradients]
        if self._server_variance is None:
            self._server_variance = [np.zeros_like(grad) for grad in gradients]
        
        # Apply adaptive optimization
        if self.optimizer_type == "adam":
            param_updates, updated_momentum, updated_variance = _compute_fedadam_update(
                gradients, self._server_momentum, self._server_variance,
                self.server_learning_rate, self.beta1, self.beta2, self.epsilon
            )
        elif self.optimizer_type == "adagrad":
            param_updates, updated_variance = _compute_fedadagrad_update(
                gradients, self._server_variance, self.server_learning_rate, self.epsilon
            )
            updated_momentum = self._server_momentum  # AdaGrad doesn't use momentum
        elif self.optimizer_type == "yogi":
            param_updates, updated_momentum, updated_variance = _compute_fedyogi_update(
                gradients, self._server_momentum, self._server_variance,
                self.server_learning_rate, self.beta1, self.beta2, self.epsilon
            )
        
        # Update optimizer state
        self._server_momentum = updated_momentum
        self._server_variance = updated_variance
        
        # Apply parameter updates
        if self._last_global_params is not None:
            updated_params = []
            for prev_param, param_update in zip(self._last_global_params, param_updates):
                updated_param = prev_param + param_update
                updated_params.append(updated_param)
        else:
            # First round: use aggregated parameters directly
            updated_params = aggregated_params_ndarrays
        
        # Update global parameters
        self._last_global_params = updated_params
        
        # Convert back to Parameters object
        final_params = ndarrays_to_parameters(updated_params)
        
        # Add FedOpt-specific metrics
        if metrics is None:
            metrics = {}
        
        # Calculate optimizer metrics
        if self.track_optimizer_state:
            optimizer_metrics = self._compute_optimizer_metrics(gradients)
            metrics.update(optimizer_metrics)
            
            # Store optimizer history
            self._optimizer_history.append({
                "round": server_round,
                "optimizer_type": self.optimizer_type,
                "server_learning_rate": self.server_learning_rate,
                **optimizer_metrics
            })
        
        return final_params, metrics
    
    def _compute_optimizer_metrics(
        self,
        gradients: List[np.ndarray]
    ) -> Dict[str, float]:
        """Compute optimizer-related metrics for monitoring."""
        if not gradients:
            return {}
        
        metrics = {}
        
        # Compute gradient statistics
        gradient_norms = []
        for grad in gradients:
            norm = np.linalg.norm(grad)
            gradient_norms.append(norm)
        
        metrics["fedopt_avg_gradient_norm"] = float(np.mean(gradient_norms))
        metrics["fedopt_max_gradient_norm"] = float(np.max(gradient_norms))
        metrics["fedopt_gradient_norm_std"] = float(np.std(gradient_norms))
        
        # Compute momentum statistics (if available)
        if self._server_momentum is not None:
            momentum_norms = []
            for momentum in self._server_momentum:
                norm = np.linalg.norm(momentum)
                momentum_norms.append(norm)
            
            metrics["fedopt_avg_momentum_norm"] = float(np.mean(momentum_norms))
            metrics["fedopt_max_momentum_norm"] = float(np.max(momentum_norms))
            metrics["fedopt_momentum_norm_std"] = float(np.std(momentum_norms))
        
        # Compute variance statistics (if available)
        if self._server_variance is not None:
            variance_norms = []
            for variance in self._server_variance:
                norm = np.linalg.norm(variance)
                variance_norms.append(norm)
            
            metrics["fedopt_avg_variance_norm"] = float(np.mean(variance_norms))
            metrics["fedopt_max_variance_norm"] = float(np.max(variance_norms))
            metrics["fedopt_variance_norm_std"] = float(np.std(variance_norms))
        
        # FedOpt configuration
        metrics["fedopt_optimizer_type"] = float(hash(self.optimizer_type) % 1000)  # Convert to numeric
        metrics["fedopt_server_learning_rate"] = float(self.server_learning_rate)
        metrics["fedopt_beta1"] = float(self.beta1)
        metrics["fedopt_beta2"] = float(self.beta2)
        metrics["fedopt_epsilon"] = float(self.epsilon)
        metrics["fedopt_adaptive_optimization_enabled"] = 1.0
        
        return metrics
    
    def get_optimizer_info(self) -> Dict[str, Any]:
        """Get information about optimizer state and history."""
        return {
            "optimizer_history": self._optimizer_history.copy(),
            "current_round": self._round,
            "optimizer_type": self.optimizer_type,
            "server_learning_rate": self.server_learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "server_momentum": self._server_momentum,
            "server_variance": self._server_variance,
            "last_global_params": self._last_global_params,
        }


def build_fedopt_strategy(
    *,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    min_fit_clients: int = 2,
    min_evaluate_clients: int = 2,
    min_available_clients: int = 2,
    optimizer_type: str = "adam",
    server_learning_rate: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    track_optimizer_state: bool = True,
) -> Any:
    """Create a FedOpt strategy with a FedAvg base strategy.
    
    FedOpt (Federated Optimization) enhances the standard federated averaging
    by incorporating adaptive optimization techniques at the server level, such
    as Adam, AdaGrad, and Yogi. This approach improves convergence rates and
    performance, especially in heterogeneous data environments.
    
    This is particularly effective for time series forecasting where different
    clients may have different temporal patterns and convergence requirements.
    
    Reference: "Adaptive Federated Optimization"
    Paper: https://arxiv.org/abs/2003.00295
    
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
    optimizer_type : str
        Type of adaptive optimizer to use ('adam', 'adagrad', 'yogi').
    server_learning_rate : float
        Learning rate for server-side adaptive optimization.
    beta1 : float
        First moment decay rate (for Adam and Yogi).
    beta2 : float
        Second moment decay rate (for Adam and Yogi).
    epsilon : float
        Small constant for numerical stability.
    track_optimizer_state : bool
        Whether to track and log optimizer state metrics.
    """
    base_strategy = fl.server.strategy.FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
    )
    
    return FedOptStrategy(
        base_strategy=base_strategy,
        optimizer_type=optimizer_type,
        server_learning_rate=server_learning_rate,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        track_optimizer_state=track_optimizer_state,
    )
