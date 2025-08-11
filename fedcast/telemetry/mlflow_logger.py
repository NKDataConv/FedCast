from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import json
import mlflow


@dataclass
class MLflowConfig:
    experiment_name: str = "FedCast"
    tracking_uri: Optional[str] = None  # If None, use default local ./mlruns
    run_name: Optional[str] = None
    tags: Optional[Dict[str, str]] = None


@contextmanager
def start_run(config: MLflowConfig):
    if config.tracking_uri:
        mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment(config.experiment_name)
    with mlflow.start_run(run_name=config.run_name, tags=config.tags):
        yield


def log_params(params: Dict[str, Any]) -> None:
    flat_params: Dict[str, Any] = {}
    for key, value in params.items():
        # Keep param values simple for MLflow
        if isinstance(value, (str, int, float, bool)) or value is None:
            flat_params[key] = value
        else:
            flat_params[key] = str(value)
    if flat_params:
        mlflow.log_params(flat_params)


def log_round_metrics(round_index: int, metrics: Dict[str, float], prefix: str = "") -> None:
    for name, value in metrics.items():
        metric_key = f"{prefix}{name}" if prefix else name
        try:
            mlflow.log_metric(metric_key, float(value), step=round_index)
        except Exception:
            # Skip non-numeric values silently
            continue


def _log_client_results(server_round: int, results: Any, phase: str) -> None:
    """Log per-client metrics for a phase ("fit" or "eval").

    Parameters
    ----------
    server_round: int
        Current federated round index.
    results: Any
        Iterable of (client_proxy, result) pairs coming from Flower.
    phase: str
        Either "fit" or "eval".
    """
    try:
        iterable = list(results)  # ensure single pass
    except Exception:
        return

    for item in iterable:
        try:
            client, res = item
        except Exception:
            continue

        cid = getattr(client, "cid", None) or getattr(client, "client_id", "unknown")
        prefix = f"client.{cid}.{phase}."

        # num_examples
        num_examples = getattr(res, "num_examples", None)
        if isinstance(num_examples, (int, float)):
            log_round_metrics(server_round, {"num_examples": float(num_examples)}, prefix=prefix)

        # loss (for evaluate)
        if phase == "eval":
            loss_val = getattr(res, "loss", None)
            if isinstance(loss_val, (int, float)):
                log_round_metrics(server_round, {"loss": float(loss_val)}, prefix=prefix)

        # metrics dict
        metrics_dict = getattr(res, "metrics", None)
        if isinstance(metrics_dict, dict):
            log_round_metrics(server_round, metrics_dict, prefix=prefix)


def log_history_artifact(history: Any, artifact_path: str = "history.json") -> None:
    try:
        serializable = _serialize_history(history)
        mlflow.log_text(json.dumps(serializable, indent=2), artifact_file=artifact_path)
    except Exception:
        # Ignore serialization errors to not block training
        pass


def _serialize_history(history: Any) -> Dict[str, Any]:
    def convert_series(series: Iterable[Tuple[int, float]]) -> List[Tuple[int, float]]:
        return [(int(r), float(v)) for r, v in series]

    payload: Dict[str, Any] = {}
    for attr in [
        "losses_distributed",
        "losses_centralized",
    ]:
        if getattr(history, attr, None):
            payload[attr] = convert_series(getattr(history, attr))

    metric_attr_candidates = [
        "metrics_distributed",
        "metrics_centralized",
        "metrics_distributed_fit",
        "metrics_distributed_evaluate",
        "metrics_centralized_fit",
        "metrics_centralized_evaluate",
    ]
    for attr in metric_attr_candidates:
        metric_dict = getattr(history, attr, None)
        if isinstance(metric_dict, dict):
            payload[attr] = {k: convert_series(v) for k, v in metric_dict.items()}
    return payload


class MLflowLoggingStrategy:
    """
    A wrapper around a Flower strategy to log aggregated metrics to MLflow.

    Works with any strategy implementing `aggregate_fit` and `aggregate_evaluate`.
    """

    def __init__(self, base_strategy: Any, metric_prefix: str = "server.", dataset_name: Optional[str] = None) -> None:
        self.base_strategy = base_strategy
        self.metric_prefix = metric_prefix
        self.dataset_name = dataset_name
        self._metadata_logged = False

    def _log_metadata_once(self) -> None:
        if self._metadata_logged:
            return
        try:
            params = {"strategy_name": type(self.base_strategy).__name__}
            if self.dataset_name:
                params["dataset"] = self.dataset_name
            log_params(params)
        except Exception:
            pass
        self._metadata_logged = True

    # Delegate all unknown attributes/methods to base strategy
    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_strategy, name)

    def aggregate_fit(self, server_round: int, results: Any, failures: Any):
        # Ensure run-level metadata is logged once
        self._log_metadata_once()
        aggregated_result = self.base_strategy.aggregate_fit(server_round, results, failures)
        # aggregated_result is (parameters, metrics)
        try:
            _, metrics = aggregated_result
        except Exception:
            metrics = None
        if isinstance(metrics, dict) and metrics:
            log_round_metrics(server_round, metrics, prefix=f"{self.metric_prefix}fit.")
        # Per-client logging for training
        _log_client_results(server_round, results, phase="fit")
        return aggregated_result

    def aggregate_evaluate(self, server_round: int, results: Any, failures: Any):
        # Ensure run-level metadata is logged once
        self._log_metadata_once()
        aggregated_result = self.base_strategy.aggregate_evaluate(server_round, results, failures)
        # aggregated_result is (loss, metrics)
        try:
            loss, metrics = aggregated_result
            if loss is not None:
                log_round_metrics(server_round, {"loss": float(loss)}, prefix=f"{self.metric_prefix}eval.")
        except Exception:
            metrics = None
        if isinstance(metrics, dict) and metrics:
            log_round_metrics(server_round, metrics, prefix=f"{self.metric_prefix}eval.")
        # Per-client logging for evaluation
        _log_client_results(server_round, results, phase="eval")
        return aggregated_result


