# fed_trend_strategy.py
# A Flower Strategy inspired by Fed-TREND for time-series forecasting.
# Reference: "Tackling Data Heterogeneity in Federated Time Series Forecasting" (Fed-TREND)
# Wei Yuan et al., arXiv:2411.15716 (2024). Not included in flwr built-ins.
# Paper: https://arxiv.org/abs/2411.15716

from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple, Any
from collections import deque

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


def _stack_like(ref: Sequence[np.ndarray], xs: List[Sequence[np.ndarray]]) -> List[np.ndarray]:
    """Stack per-layer arrays across clients: xs: list over clients -> list over layers -> np.ndarray."""
    # result[i] has shape (num_clients, *layer_shape)
    num_layers = len(ref)
    out: List[np.ndarray] = []
    for i in range(num_layers):
        out.append(np.stack([x[i] for x in xs], axis=0))
    return out


def _sign_consistency_mask(stacked: List[np.ndarray], threshold: float) -> List[np.ndarray]:
    """Return mask per layer (1 where most clients agree on sign, else 0)."""
    masks: List[np.ndarray] = []
    for layer in stacked:
        # layer shape: (C, *shape)
        signs = np.sign(layer)
        # vote on majority sign ignoring zeros
        pos = (signs > 0).sum(axis=0)
        neg = (signs < 0).sum(axis=0)
        tot = np.maximum(pos + neg, 1)  # avoid div0 when all zeros
        agree = np.maximum(pos, neg) / tot  # fraction agreeing with majority sign
        masks.append((agree >= threshold).astype(layer.dtype))
    return masks


def _masked_mean(stacked: List[np.ndarray], masks: List[np.ndarray]) -> List[np.ndarray]:
    """Masked mean across clients (broadcast masks to client axis)."""
    means: List[np.ndarray] = []
    for layer, mask in zip(stacked, masks):
        # layer: (C, *shape), mask: (*shape)
        m = mask  # broadcast on client axis
        # avoid 0-division: if m is zero everywhere, fall back to vanilla mean
        denom = np.clip(m, 0, 1).astype(layer.dtype)
        denom_sum = denom  # same shape as elements
        # weighted sum across clients with equal client weights but masked elements
        # implement as elementwise mean over nonzero-mask positions
        # sum over client axis:
        sum_layer = (layer * 1.0).sum(axis=0)
        C = layer.shape[0]
        # If mask is zero, use plain mean (keep some signal)
        safe_denom = np.where(denom_sum > 0, denom_sum, 1.0)
        mean_all = sum_layer / C
        mean_masked = np.where(denom_sum > 0, sum_layer / C, mean_all)
        # Now enforce zeros where mask=0 to avoid pushing in noisy dims
        means.append(mean_masked * m + mean_all * (1 - m))
    return means


def _add(a: Sequence[np.ndarray], b: Sequence[np.ndarray], alpha: float = 1.0) -> List[np.ndarray]:
    return [ai + alpha * bi for ai, bi in zip(a, b)]


def _sub(a: Sequence[np.ndarray], b: Sequence[np.ndarray]) -> List[np.ndarray]:
    return [ai - bi for ai, bi in zip(a, b)]


def _scale(a: Sequence[np.ndarray], s: float) -> List[np.ndarray]:
    return [ai * s for ai in a]


class FedTRENDStrategy(Strategy):
    """
    A Flower Strategy that injects *time-series-aware* trajectory information into aggregation.

    Core ideas adapted from Fed-TREND (Yuan et al., 2024):
      • Build a consensus update from *client* model-trajectory deltas (proxy for D_ct).
      • Refine the aggregated global model using a trend update from *global* deltas (proxy for D_gt).

    This class composes a base strategy (defaults to FedAvg) and augments aggregate_fit.

    Parameters
    ----------
    base_strategy : Strategy
        The underlying strategy to perform vanilla aggregation (e.g., FedAvg, FedAdam).
    l_ct : int
        How often (in rounds) to recompute the client-consensus update.
    l_gt : int
        How often (in rounds) to recompute the global-trend update.
    consistency_threshold : float in [0.5, 1.0]
        Fraction of clients that must agree on sign per-parameter to treat that element as "consistent".
    eta_refine : float
        Step size applied to the post-aggregation refinement along the global-trend update.
    ema_momentum : float
        Momentum for EMA smoothing of global trend.
    """

    def __init__(
        self,
        base_strategy: Optional[Strategy] = None,
        l_ct: int = 5,
        l_gt: int = 5,
        consistency_threshold: float = 0.6,
        eta_refine: float = 0.05,
        ema_momentum: float = 0.9,
    ) -> None:
        self.base = base_strategy if base_strategy is not None else FedAvg()
        self.l_ct = max(1, l_ct)
        self.l_gt = max(1, l_gt)
        self.consistency_threshold = float(consistency_threshold)
        self.eta_refine = float(eta_refine)
        self.ema_momentum = float(ema_momentum)

        # Internal state
        self._round = 0
        self._last_global_params: Optional[List[np.ndarray]] = None
        self._ct_delta_buffer: List[List[np.ndarray]] = []  # collect per-client deltas over rounds
        self._consensus_delta: Optional[List[np.ndarray]] = None

        self._global_delta_hist: deque[List[np.ndarray]] = deque(maxlen=50)
        self._global_trend_ema: Optional[List[np.ndarray]] = None

    # ---- Required Strategy API, most delegated to base strategy ----

    def initialize_parameters(self, client_manager):
        return self.base.initialize_parameters(client_manager)

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager,
    ) -> List[Tuple[ClientProxy, Dict]]:
        # You can inject hints for clients via config if your client supports it
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

    # ---- FedTREND logic hooks into aggregate_fit ----

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        self._round = server_round

        # 1) Get the *pre-aggregated* global params for this round (the broadcasted ones)
        # Flower provides them via FitRes.metrics? Not guaranteed. Keep our own snapshot.
        if self._last_global_params is None and len(results) > 0:
            # First round: infer shape from first client return
            self._last_global_params = parameters_to_ndarrays(results[0][1].parameters)

        prev_global = self._last_global_params

        # 2) Collect client end params and accumulate deltas for CT consensus
        client_end_params: List[List[np.ndarray]] = [parameters_to_ndarrays(fr.parameters) for _, fr in results]
        if prev_global is not None:
            for end in client_end_params:
                self._ct_delta_buffer.append(_sub(end, prev_global))

        # 3) Do the base aggregation (FedAvg/FedAdam/etc.)
        base_agg = self.base.aggregate_fit(server_round, results, failures)
        aggregated_params, metrics = base_agg

        if aggregated_params is None:
            return base_agg  # Nothing to do

        W_aggr = parameters_to_ndarrays(aggregated_params)

        # 4) Possibly recompute CLIENT consensus delta (proxy for D_ct)
        if server_round % self.l_ct == 0 and len(self._ct_delta_buffer) > 0 and prev_global is not None:
            # Stack deltas across all buffered clients (could be multiple rounds worth)
            stacked = _stack_like(prev_global, self._ct_delta_buffer)  # list over layers, each (C, *shape)
            masks = _sign_consistency_mask(stacked, self.consistency_threshold)
            consensus = _masked_mean(stacked, masks)  # per-layer masked mean
            self._consensus_delta = consensus
            self._ct_delta_buffer.clear()
            # NOTE: In the original Fed-TREND, D_ct is mixed into local training on clients.
            # If your clients support it, you can ship `consensus` via an out-of-band channel
            # and add a small proximal or direction term during local training.

        # 5) Track GLOBAL deltas and compute a trend direction (proxy for D_gt)
        if prev_global is not None:
            gdelta = _sub(W_aggr, prev_global)
            self._global_delta_hist.append(gdelta)

            # Recompute trend every l_gt rounds
            if server_round % self.l_gt == 0:
                # Use simple EMA over the historical global deltas
                if self._global_trend_ema is None:
                    self._global_trend_ema = [d.copy() for d in gdelta]
                else:
                    self._global_trend_ema = [
                        self.ema_momentum * ema + (1.0 - self.ema_momentum) * d
                        for ema, d in zip(self._global_trend_ema, gdelta)
                    ]

        # 6) Refine the aggregated global model along the global trend (Fed-TREND refinement)
        if self._global_trend_ema is not None and self.eta_refine > 0.0:
            W_refined = _add(W_aggr, self._global_trend_ema, alpha=self.eta_refine)
        else:
            W_refined = W_aggr

        # 7) Bookkeeping and return
        self._last_global_params = [w.copy() for w in W_refined]
        refined_params = ndarrays_to_parameters(W_refined)

        # Expose a couple of metrics so you can track behaviour
        if self._consensus_delta is not None:
            # average absolute magnitude of consensus
            mag = float(np.mean([np.mean(np.abs(x)) for x in self._consensus_delta]))
            metrics = dict(metrics or {})
            metrics["fedtrend_consensus_mag"] = mag
        if self._global_trend_ema is not None:
            gmag = float(np.mean([np.mean(np.abs(x)) for x in self._global_trend_ema]))
            metrics = dict(metrics or {})
            metrics["fedtrend_trend_mag"] = gmag

        return refined_params, metrics

    # Optional: expose the latest consensus delta if you want clients to request it
    def get_latest_consensus_delta(self) -> Optional[List[np.ndarray]]:
        return self._consensus_delta


def build_fedtrend_strategy(
    *,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    min_fit_clients: int = 2,
    min_evaluate_clients: int = 2,
    min_available_clients: int = 2,
    l_ct: int = 5,
    l_gt: int = 5,
    consistency_threshold: float = 0.6,
    eta_refine: float = 0.05,
    ema_momentum: float = 0.9,
) -> Any:
    """Create a FedTrend strategy with a FedAvg base strategy.
    
    FedTrend is a time-series-aware federated learning strategy that injects
    trajectory information into aggregation. It builds consensus updates from
    client model-trajectory deltas and refines the global model using trend
    updates from global deltas.
    
    Reference: Yuan et al. "Tackling Data Heterogeneity in Federated Time Series Forecasting"
    Paper: https://arxiv.org/abs/2411.15716
    
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
    l_ct : int
        How often (in rounds) to recompute the client-consensus update.
    l_gt : int
        How often (in rounds) to recompute the global-trend update.
    consistency_threshold : float
        Fraction of clients that must agree on sign per-parameter to treat
        that element as "consistent" (range: [0.5, 1.0]).
    eta_refine : float
        Step size applied to the post-aggregation refinement along the global-trend update.
    ema_momentum : float
        Momentum for EMA smoothing of global trend.
    """
    base_strategy = fl.server.strategy.FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
    )
    
    return FedTRENDStrategy(
        base_strategy=base_strategy,
        l_ct=l_ct,
        l_gt=l_gt,
        consistency_threshold=consistency_threshold,
        eta_refine=eta_refine,
        ema_momentum=ema_momentum,
    )