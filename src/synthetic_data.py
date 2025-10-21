"""
Synthetic data generation for QILTR experiments
"""

import numpy as np
from .als_solver import mode_product

def generate_low_rank_tensor(shape, ranks, random_state=None):
    """
    Generate random low-rank tensor via Tucker decomposition

    Parameters:
    -----------
    shape : tuple
        Tensor shape
    ranks : tuple
        Tucker ranks
    random_state : int
        Random seed

    Returns:
    --------
    tensor : ndarray
        Generated tensor
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Generate core tensor
    core = np.random.randn(*ranks)

    # Generate factor matrices
    factors = []
    for mode, (dim, rank) in enumerate(zip(shape, ranks)):
        factor = np.random.randn(dim, rank)
        factor, _ = np.linalg.qr(factor)  # Orthonormalize
        factors.append(factor)

    # Reconstruct tensor
    tensor = core.copy()
    for mode, factor in enumerate(factors):
        tensor = mode_product(tensor, factor, mode)

    return tensor


def generate_synthetic_tensor_regression(
    n_samples=500,
    input_dim=20,
    tensor_shape=(5, 5, 5),
    tucker_ranks=(3, 3, 3),
    n_regions=3,
    noise_level=0.5,
    random_state=42,
    ill_conditioned=False,
    condition_number=100.0,
    high_nonstationarity=False,
    region_diversity_scale=1.0
):
    """
    Generate synthetic tensor regression data with non-stationary structure
    
    ENHANCED: Now supports challenging data conditions for robust testing

    Parameters:
    -----------
    n_samples : int
        Number of samples
    input_dim : int
        Input feature dimension
    tensor_shape : tuple
        Shape of response tensors
    tucker_ranks : tuple
        True Tucker ranks
    n_regions : int
        Number of non-stationary regions
    noise_level : float
        Noise standard deviation
    random_state : int
        Random seed
    ill_conditioned : bool
        If True, generate highly correlated (ill-conditioned) features
    condition_number : float
        Target condition number for ill-conditioned features
    high_nonstationarity : bool
        If True, make regional differences more pronounced
    region_diversity_scale : float
        Multiplier for regional tensor differences (default 1.0)

    Returns:
    --------
    X : ndarray, shape (n_samples, input_dim)
        Input features
    Y : ndarray, shape (n_samples, *tensor_shape)
        Tensor responses
    region_labels : ndarray, shape (n_samples,)
        Region assignment for each sample
    true_tensors : list
        True coefficient tensors for each region
    """
    np.random.seed(random_state)

    # Generate input features (standard or ill-conditioned)
    if ill_conditioned:
        # Generate highly correlated features with controlled condition number
        # Start with one base feature and add small perturbations
        base = np.random.randn(n_samples, 1)
        X = base @ np.ones((1, input_dim)) + 0.01 * np.random.randn(n_samples, input_dim)
        
        # Add controlled structure using SVD to achieve target condition number
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        # Set singular values to create desired condition number
        s_new = np.linspace(1.0, 1.0/condition_number, min(n_samples, input_dim))
        X = U[:, :len(s_new)] @ np.diag(s_new) @ Vt[:len(s_new), :]
    else:
        # Standard uncorrelated features
        X = np.random.randn(n_samples, input_dim)

    # Generate region centers in feature space
    region_centers = []
    for r in range(n_regions):
        center = np.random.randn(input_dim) * 3
        region_centers.append(center)
    region_centers = np.array(region_centers)

    # Assign samples to regions based on nearest center
    distances_to_centers = np.array([
        np.linalg.norm(X - center, axis=1) 
        for center in region_centers
    ]).T
    region_labels = np.argmin(distances_to_centers, axis=1)

    # Generate true coefficient tensor for each region
    # ENHANCED: Add diversity scale for high nonstationarity
    true_tensors = []
    for r in range(n_regions):
        base_tensor = generate_low_rank_tensor(
            tensor_shape, 
            tucker_ranks, 
            random_state=random_state + r
        )
        
        # Add region-specific offset for high nonstationarity
        if high_nonstationarity:
            # Make each region's tensor significantly different
            region_offset = r * region_diversity_scale
            base_tensor = base_tensor + region_offset
        
        true_tensors.append(base_tensor)

    # Generate responses
    Y = np.zeros((n_samples,) + tensor_shape)

    for i in range(n_samples):
        region = region_labels[i]

        # FIXED: Create proper tensor regression relationship
        # Y = W ⊗ X where W is the coefficient tensor for this region
        # We'll use tensor contraction along a subset of dimensions
        
        # Get regional coefficient tensor
        W_region = true_tensors[region]
        
        # Create tensor response that actually depends on X[i]
        # Method: Each element Y[i, j, k, l] depends on a linear combination of X[i]
        # Y_jkl = sum_d(W_jkl * X_d) - proper tensor regression
        
        # Simple approach: use outer product structure
        # Y[i] = W_region * (scalar_function_of_X)
        scalar_feature = np.tanh(np.sum(X[i] * np.random.randn(input_dim)))
        
        # More complex: each tensor slice depends on different features
        # Use bounded scaling to avoid outlier explosion
        Y[i] = W_region.copy()
        for j in range(tensor_shape[0]):
            for k in range(tensor_shape[1]):
                # Each tensor element has a different feature dependency
                feature_idx = (j * tensor_shape[1] + k) % input_dim
                # Use tanh to bound the effect: scale in [0.5, 1.5]
                scale = 1.0 + 0.5 * np.tanh(X[i, feature_idx])
                Y[i, j, k, :] *= scale
        
        # Add noise
        noise = np.random.randn(*tensor_shape) * noise_level
        Y[i] = Y[i] + noise

    return X, Y, region_labels, true_tensors


def generate_multiple_configs(configs, base_seed=42):
    """
    Generate multiple synthetic datasets with different configurations

    Parameters:
    -----------
    configs : list of dict
        List of configuration dictionaries
    base_seed : int
        Base random seed

    Returns:
    --------
    datasets : list
        List of (X, Y, metadata) tuples
    """
    datasets = []

    for i, config in enumerate(configs):
        seed = base_seed + i * 100

        X, Y, region_labels, true_tensors = generate_synthetic_tensor_regression(
            n_samples=config.get('n_samples', 500),
            input_dim=config.get('input_dim', 20),
            tensor_shape=config.get('tensor_shape', (5, 5, 5)),
            tucker_ranks=config.get('tucker_ranks', (3, 3, 3)),
            n_regions=config.get('n_regions', 3),
            noise_level=config.get('noise_level', 0.5),
            random_state=seed
        )

        metadata = {
            'config': config,
            'seed': seed,
            'region_labels': region_labels,
            'true_tensors': true_tensors
        }

        datasets.append((X, Y, metadata))

    return datasets


def add_outliers(X, Y, outlier_fraction=0.1, outlier_magnitude=5.0, random_state=42):
    """
    Add outliers to dataset
    
    ENHANCED: Adds large-magnitude contamination to both features and responses

    Parameters:
    -----------
    X : ndarray
        Input features
    Y : ndarray
        Tensor responses
    outlier_fraction : float
        Fraction of outliers to add (e.g., 0.2 for 20%)
    outlier_magnitude : float
        Magnitude of outlier noise (e.g., 10.0 for strong outliers)
    random_state : int
        Random seed

    Returns:
    --------
    X_out : ndarray
        Input features with outliers
    Y_out : ndarray
        Responses with outliers
    outlier_indices : ndarray
        Indices of samples that were contaminated
    """
    np.random.seed(random_state)

    N = X.shape[0]
    n_outliers = int(N * outlier_fraction)

    X_out = X.copy()
    Y_out = Y.copy()

    # Select random samples to be outliers
    outlier_indices = np.random.choice(N, n_outliers, replace=False)

    for idx in outlier_indices:
        # Add large noise to features
        X_out[idx] += np.random.randn(*X.shape[1:]) * outlier_magnitude
        
        # Add large noise to response
        Y_out[idx] += np.random.randn(*Y.shape[1:]) * outlier_magnitude

    return X_out, Y_out, outlier_indices
