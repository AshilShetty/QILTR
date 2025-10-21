"""
QILTR Source Package
Core implementations for Quantum-Inspired Local Tensor Regression
"""

from .encodings import QuantumEncoder
from .distances import bures_distance, bures_distance_batch, compute_weights
from .als_solver import WeightedTuckerALS, mode_product, unfold, fold
from .qiltr import QILTR
from .baselines import EuclideanLTR, GlobalTuckerRegression
from .synthetic_data import generate_synthetic_tensor_regression, generate_low_rank_tensor
from .metrics import (
    tensor_mse,
    tensor_mae,
    tensor_relative_error,
    tensor_r2_score,
    compute_all_metrics
)

__all__ = [
    'QuantumEncoder',
    'bures_distance',
    'bures_distance_batch',
    'compute_weights',
    'WeightedTuckerALS',
    'mode_product',
    'unfold',
    'fold',
    'QILTR',
    'EuclideanLTR',
    'GlobalTuckerRegression',
    'generate_synthetic_tensor_regression',
    'generate_low_rank_tensor',
    'tensor_mse',
    'tensor_mae',
    'tensor_relative_error',
    'tensor_r2_score',
    'compute_all_metrics'
]
