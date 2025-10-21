"""
Weighted Tucker-ALS solver for tensor regression
"""

import numpy as np
from scipy.linalg import svd

def tucker_decomposition(tensor, ranks):
    """
    Tucker decomposition via Higher-Order SVD (HOSVD)

    Parameters:
    -----------
    tensor : ndarray
        Input tensor
    ranks : tuple
        Tucker ranks for each mode

    Returns:
    --------
    core : ndarray
        Core tensor
    factors : list of ndarray
        Factor matrices for each mode
    """
    factors = []
    shape = tensor.shape

    for mode in range(len(shape)):
        # Mode-k unfolding
        unfolded = unfold(tensor, mode)

        # Compute SVD
        U, s, Vt = svd(unfolded, full_matrices=False)

        # Truncate to rank
        U_truncated = U[:, :ranks[mode]]
        factors.append(U_truncated)

    # Compute core tensor
    core = tensor.copy()
    for mode, factor in enumerate(factors):
        core = mode_product(core, factor.T, mode)

    return core, factors

def unfold(tensor, mode):
    """Mode-k unfolding of a tensor"""
    shape = tensor.shape
    n_dims = len(shape)

    # Move mode to front, then reshape
    perm = [mode] + [i for i in range(n_dims) if i != mode]
    unfolded = np.transpose(tensor, perm)
    unfolded = unfolded.reshape(shape[mode], -1)

    return unfolded

def mode_product(tensor, matrix, mode):
    """Mode-k product of tensor with matrix"""
    # Unfold tensor along mode
    unfolded = unfold(tensor, mode)

    # Matrix multiply
    result_unfolded = matrix @ unfolded

    # Fold back
    new_shape = list(tensor.shape)
    new_shape[mode] = matrix.shape[0]
    result = fold(result_unfolded, mode, tuple(new_shape))

    return result

def fold(unfolded, mode, shape):
    """Fold unfolded matrix back to tensor"""
    n_dims = len(shape)
    perm = [mode] + [i for i in range(n_dims) if i != mode]

    # Reshape to permuted shape
    permuted_shape = [shape[i] for i in perm]
    tensor = unfolded.reshape(permuted_shape)

    # Inverse permutation
    inv_perm = [perm.index(i) for i in range(n_dims)]
    tensor = np.transpose(tensor, inv_perm)

    return tensor

class WeightedTuckerALS:
    """Weighted Tucker-ALS solver for tensor regression"""

    def __init__(self, ranks, max_iter=100, tol=1e-6, reg_lambda=0.01):
        self.ranks = ranks
        self.max_iter = max_iter
        self.tol = tol
        self.reg_lambda = reg_lambda
        self.convergence_history = []

    def fit(self, Y, weights):
        """
        Fit weighted Tucker decomposition

        Parameters:
        -----------
        Y : ndarray, shape (N, P1, P2, P3)
            Tensor responses
        weights : ndarray, shape (N,)
            Sample weights

        Returns:
        --------
        core : ndarray
            Core tensor
        factors : list of ndarray
            Factor matrices
        """
        N = Y.shape[0]
        tensor_shape = Y.shape[1:]
        n_modes = len(tensor_shape)

        # Initialize weighted mean tensor
        weighted_mean_Y = np.average(Y, axis=0, weights=weights)

        # Initialize factors using SVD
        factors = []
        for mode in range(n_modes):
            unfolded = unfold(weighted_mean_Y, mode)
            U, s, Vt = svd(unfolded, full_matrices=False)
            factors.append(U[:, :self.ranks[mode]])

        # Initialize core
        core = weighted_mean_Y.copy()
        for mode, factor in enumerate(factors):
            core = mode_product(core, factor.T, mode)

        # ALS iterations
        prev_obj = float('inf')

        for iteration in range(self.max_iter):
            # Update each factor
            for mode in range(n_modes):
                # Compute weighted unfolding
                Y_mode_unfold = np.array([unfold(Y[i], mode) for i in range(N)])
                sqrt_weights = np.sqrt(weights).reshape(-1, 1, 1)
                weighted_Y_mode = Y_mode_unfold * sqrt_weights
                avg_Y_mode = np.sum(weighted_Y_mode, axis=0)

                # SVD update
                U, s, Vt = svd(avg_Y_mode, full_matrices=False)
                factors[mode] = U[:, :self.ranks[mode]]

            # Update core
            core = weighted_mean_Y.copy()
            for mode, factor in enumerate(factors):
                core = mode_product(core, factor.T, mode)

            # Compute objective
            obj = self._compute_objective(Y, core, factors, weights)
            self.convergence_history.append(obj)

            # Check convergence
            rel_change = abs(prev_obj - obj) / (abs(prev_obj) + 1e-10)
            if rel_change < self.tol:
                break

            prev_obj = obj

        self.core = core
        self.factors = factors

        return core, factors

    def _compute_objective(self, Y, core, factors, weights):
        """Compute weighted reconstruction error"""
        N = Y.shape[0]
        error = 0.0

        Y_reconstructed = self.reconstruct(core, factors)

        for i in range(N):
            diff = Y[i] - Y_reconstructed
            error += weights[i] * np.sum(diff**2)

        return error

    def reconstruct(self, core, factors):
        """Reconstruct tensor from Tucker decomposition"""
        tensor = core.copy()
        for mode, factor in enumerate(factors):
            tensor = mode_product(tensor, factor, mode)
        return tensor
