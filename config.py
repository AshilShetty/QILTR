"""
Configuration file for QILTR experiments
Modify parameters here to control all experiments
"""

import os
import numpy as np

# Random seed for reproducibility
RANDOM_SEED = 42

# Results directory
RESULTS_DIR = 'results'

# Synthetic Data Parameters
SYNTHETIC_CONFIG = {
    'sample_sizes': [200, 500, 1000],
    'input_dims': [10, 20, 50],
    'tensor_shapes': [(5, 5, 5), (10, 10, 5)],
    'tucker_ranks': [(2, 2, 2), (3, 3, 3)],
    'noise_levels': [0.1, 0.5, 1.0],
    'n_centroids': [5, 10],
    'non_stationary_regions': 3
}

# QILTR Parameters
QILTR_PARAMS = {
    'quantum_dims': [2, 4, 8],
    'bandwidths': [0.5, 1.0, 2.0],
    'default_quantum_dim': 4,
    'default_bandwidth': 1.0,
    'max_als_iter': 100,
    'als_tol': 1e-6,
    'reg_lambda': 0.01
}

# Baseline Parameters
BASELINE_PARAMS = {
    'euclidean_bandwidths': [0.5, 1.0, 2.0],
    'global_reg_lambda': 0.01
}

# Experiment Settings
EXPERIMENT_CONFIG = {
    'n_trials': 5,  # Number of random trials for statistical testing
    'test_split': 0.15,
    'val_split': 0.15,
    'cv_folds': 5
}

# ==============================================================================
# ENHANCED: Challenging Data Scenarios Configuration
# ==============================================================================
# Toggle these to test QILTR under challenging conditions

CHALLENGING_SCENARIOS = {
    'run_all_scenarios': True,  # Set to False to run only standard scenario
    
    # Scenario 1: Standard (baseline)
    'standard': {
        'enabled': True,
        'name': 'Standard',
        'n_regions': 3,
        'ill_conditioned': False,
        'high_nonstationarity': False,
        'outlier_fraction': 0.0,
        'outlier_magnitude': 0.0
    },
    
    # Scenario 2: High Nonstationarity (many diverse regions)
    'high_nonstationarity': {
        'enabled': True,
        'name': 'High Nonstationarity',
        'n_regions': 10,  # More regions
        'ill_conditioned': False,
        'high_nonstationarity': True,
        'region_diversity_scale': 2.0,  # More pronounced differences
        'outlier_fraction': 0.0,
        'outlier_magnitude': 0.0
    },
    
    # Scenario 3: Ill-Conditioned Features
    'ill_conditioned': {
        'enabled': True,
        'name': 'Ill-Conditioned Features',
        'n_regions': 3,
        'ill_conditioned': True,
        'condition_number': 100.0,  # High correlation
        'high_nonstationarity': False,
        'outlier_fraction': 0.0,
        'outlier_magnitude': 0.0
    },
    
    # Scenario 4: Strong Outlier Contamination (20%)
    'outliers': {
        'enabled': True,
        'name': 'Strong Outliers (20%)',
        'n_regions': 3,
        'ill_conditioned': False,
        'high_nonstationarity': False,
        'outlier_fraction': 0.2,  # 20% outliers
        'outlier_magnitude': 10.0  # Large magnitude
    },
    
    # Scenario 5: Combined Challenge (worst case)
    'combined': {
        'enabled': True,
        'name': 'Combined Challenges',
        'n_regions': 10,
        'ill_conditioned': True,
        'condition_number': 100.0,
        'high_nonstationarity': True,
        'region_diversity_scale': 2.0,
        'outlier_fraction': 0.2,
        'outlier_magnitude': 10.0
    }
}

# Computational Resources
COMPUTE_CONFIG = {
    'n_jobs': -1,  # -1 means use all available cores
    'verbose': True
}

# Output Directories
OUTPUT_DIRS = {
    'results': 'results/',
    'figures': 'results/figures/',
    'tables': 'results/tables/',
    'logs': 'results/logs/',
    'models': 'results/models/'
}

# Create output directories
for dir_path in OUTPUT_DIRS.values():
    os.makedirs(dir_path, exist_ok=True)
