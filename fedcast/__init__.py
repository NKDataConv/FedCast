"""
FedCast: Federated Learning for Time Series Forecasting

A modular framework for time series forecasting using federated learning,
built on top of the Flower framework.
"""

__version__ = "0.1.1b1"
__author__ = "FedCast Team"
__email__ = "nk@data-convolution.de"

# Import main modules for easy access
from . import cast_models
from . import datasets
from . import federated_learning_strategies
from . import experiments
from . import telemetry

__all__ = [
    "cast_models",
    "datasets", 
    "federated_learning_strategies",
    "experiments",
    "telemetry",
]
