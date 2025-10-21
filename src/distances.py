"""
Distance metrics including Bures distance for quantum states
"""

import numpy as np
from scipy.linalg import sqrtm

def matrix_sqrt(A):
    """
    Compute matrix square root efficiently

    Parameters:
    -----------
    A : ndarray, shape (d, d)
        Positive semidefinite matrix

    Returns:
    --------
    sqrtA : ndarray, shape (d, d)
        Matrix square root
    """
    # Use eigendecomposition for Hermitian matrices
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, 0)  # Ensure non-negative
    sqrtA = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.conj().T
    return sqrtA

def bures_distance(rho1, rho2):
    """
    Compute Bures distance between two density matrices

    D_B(ρ₁, ρ₂) = √(2 - 2√F)
    where F = Tr[√(√ρ₁ ρ₂ √ρ₁)]²

    Parameters:
    -----------
    rho1, rho2 : ndarray, shape (d, d)
        Density matrices

    Returns:
    --------
    distance : float
        Bures distance
    """
    # Add small regularization for numerical stability
    eps = 1e-10
    d = rho1.shape[0]
    rho1_reg = rho1 + eps * np.eye(d)
    rho2_reg = rho2 + eps * np.eye(d)

    # Compute fidelity
    sqrt_rho1 = matrix_sqrt(rho1_reg)
    M = sqrt_rho1 @ rho2_reg @ sqrt_rho1
    sqrt_M = matrix_sqrt(M)
    fidelity = np.real(np.trace(sqrt_M))**2

    # Bures distance from fidelity
    distance = np.sqrt(2 - 2 * np.sqrt(np.clip(fidelity, 0, 1)))

    return distance

def bures_distance_batch(rho_ref, rhos):
    """
    Compute Bures distances from reference to batch of states

    Parameters:
    -----------
    rho_ref : ndarray, shape (d, d)
        Reference density matrix
    rhos : ndarray, shape (N, d, d)
        Batch of density matrices

    Returns:
    --------
    distances : ndarray, shape (N,)
        Bures distances
    """
    N = rhos.shape[0]
    distances = np.zeros(N)

    for i in range(N):
        distances[i] = bures_distance(rho_ref, rhos[i])

    return distances

def euclidean_distance(x1, x2):
    """Standard Euclidean distance"""
    return np.linalg.norm(x1 - x2)

def euclidean_distance_batch(x_ref, X):
    """
    Compute Euclidean distances from reference to batch

    Parameters:
    -----------
    x_ref : ndarray, shape (D,)
        Reference vector
    X : ndarray, shape (N, D)
        Batch of vectors

    Returns:
    --------
    distances : ndarray, shape (N,)
        Euclidean distances
    """
    return np.linalg.norm(X - x_ref, axis=1)

def compute_weights(distances, bandwidth):
    """
    Compute Gaussian kernel weights from distances

    Parameters:
    -----------
    distances : ndarray, shape (N,)
        Distances
    bandwidth : float
        Kernel bandwidth

    Returns:
    --------
    weights : ndarray, shape (N,)
        Normalized weights
    """
    weights = np.exp(-distances**2 / (2 * bandwidth**2))
    # Normalize
    weights = weights / (np.sum(weights) + 1e-10)
    return weights
