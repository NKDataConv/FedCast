"""
Experiment modules for FedCast.

This module provides experiment implementations for testing and evaluating
federated learning strategies on time series data.
"""

from . import basic_fedavg
from . import basic_fedtrend
from . import grid_all

__all__ = [
    "basic_fedavg",
    "basic_fedtrend", 
    "grid_all",
]
