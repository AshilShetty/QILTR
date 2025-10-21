"""
Evaluation metrics for tensor regression
"""

import numpy as np
from scipy import stats

def tensor_mse(Y_true, Y_pred):
    """
    Mean Squared Error for tensors

    Parameters:
    -----------
    Y_true : ndarray, shape (N, P1, P2, ...)
        True tensors
    Y_pred : ndarray, shape (N, P1, P2, ...)
        Predicted tensors

    Returns:
    --------
    mse : float
        Mean squared error
    """
    return np.mean((Y_true - Y_pred) ** 2)


def tensor_mae(Y_true, Y_pred):
    """Mean Absolute Error for tensors"""
    return np.mean(np.abs(Y_true - Y_pred))


def tensor_relative_error(Y_true, Y_pred):
    """
    Relative error for tensors

    RE = ||Y_true - Y_pred||_F / ||Y_true||_F
    """
    diff_norm = np.linalg.norm((Y_true - Y_pred).reshape(Y_true.shape[0], -1), axis=1)
    true_norm = np.linalg.norm(Y_true.reshape(Y_true.shape[0], -1), axis=1)

    # Avoid division by zero
    relative_errors = diff_norm / (true_norm + 1e-10)

    return np.mean(relative_errors)


def tensor_r2_score(Y_true, Y_pred):
    """
    R^2 score for tensors

    R^2 = 1 - SS_res / SS_tot
    """
    # Flatten tensors
    Y_true_flat = Y_true.reshape(Y_true.shape[0], -1)
    Y_pred_flat = Y_pred.reshape(Y_pred.shape[0], -1)

    # Mean of true values
    Y_mean = np.mean(Y_true_flat, axis=0)

    # Sum of squared residuals
    ss_res = np.sum((Y_true_flat - Y_pred_flat) ** 2)

    # Total sum of squares
    ss_tot = np.sum((Y_true_flat - Y_mean) ** 2)

    # R^2 score
    r2 = 1 - ss_res / (ss_tot + 1e-10)

    return r2


def compute_all_metrics(Y_true, Y_pred):
    """
    Compute all evaluation metrics

    Parameters:
    -----------
    Y_true : ndarray
        True tensors
    Y_pred : ndarray
        Predicted tensors

    Returns:
    --------
    metrics : dict
        Dictionary of metric values
    """
    metrics = {
        'mse': tensor_mse(Y_true, Y_pred),
        'mae': tensor_mae(Y_true, Y_pred),
        'relative_error': tensor_relative_error(Y_true, Y_pred),
        'r2': tensor_r2_score(Y_true, Y_pred),  # Use 'r2' key instead of 'r2_score'
        'r2_score': tensor_r2_score(Y_true, Y_pred)  # Keep both for compatibility
    }

    return metrics


def compare_methods_statistical(results_dict, metric='mse', alpha=0.05):
    """
    Statistical comparison of multiple methods using paired t-test

    Parameters:
    -----------
    results_dict : dict
        Dictionary mapping method names to arrays of metric values
        Each array has shape (n_trials,)
    metric : str
        Metric name
    alpha : float
        Significance level

    Returns:
    --------
    comparison : dict
        Statistical comparison results
    """
    methods = list(results_dict.keys())
    n_methods = len(methods)

    # Pairwise comparisons
    pairwise_pvals = {}

    for i in range(n_methods):
        for j in range(i+1, n_methods):
            method1 = methods[i]
            method2 = methods[j]

            values1 = results_dict[method1]
            values2 = results_dict[method2]

            # Paired t-test
            t_stat, p_val = stats.ttest_rel(values1, values2)

            pairwise_pvals[f"{method1}_vs_{method2}"] = {
                't_statistic': t_stat,
                'p_value': p_val,
                'significant': p_val < alpha,
                'mean_diff': np.mean(values1 - values2)
            }

    # Compute means and stds
    summary = {}
    for method in methods:
        values = results_dict[method]
        summary[method] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'median': np.median(values),
            'min': np.min(values),
            'max': np.max(values)
        }

    comparison = {
        'metric': metric,
        'summary': summary,
        'pairwise_tests': pairwise_pvals
    }

    return comparison


def convergence_rate_analysis(convergence_history):
    """
    Analyze convergence rate from objective history

    Parameters:
    -----------
    convergence_history : list
        List of objective values per iteration

    Returns:
    --------
    analysis : dict
        Convergence analysis results
    """
    if len(convergence_history) == 0:
        return {'rate': None, 'converged': False}

    history = np.array(convergence_history)

    # Compute log of objective (for linear convergence)
    log_obj = np.log(history + 1e-10)

    # Fit linear trend to estimate convergence rate
    iterations = np.arange(len(history))

    if len(iterations) > 1:
        slope, intercept = np.polyfit(iterations, log_obj, 1)
        convergence_rate = -slope  # Negative of slope (decrease rate)
    else:
        convergence_rate = None

    # Check if converged (relative change < threshold)
    converged = False
    if len(history) > 1:
        final_change = abs(history[-1] - history[-2]) / (abs(history[-2]) + 1e-10)
        converged = final_change < 1e-6

    analysis = {
        'n_iterations': len(history),
        'initial_objective': history[0],
        'final_objective': history[-1],
        'convergence_rate': convergence_rate,
        'converged': converged,
        'objective_reduction': (history[0] - history[-1]) / (history[0] + 1e-10)
    }

    return analysis
