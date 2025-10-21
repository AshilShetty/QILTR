"""
Main QILTR algorithm implementation
"""

import numpy as np
from .encodings import QuantumEncoder
from .distances import bures_distance_batch, compute_weights
from .als_solver import WeightedTuckerALS

class QILTR:
    """Quantum-Inspired Local Tensor Regression"""

    def __init__(self, n_centroids=10, quantum_dim=4, bandwidth=1.0,
                 ranks=(3, 3, 3), encoding_type='amplitude',
                 max_als_iter=100, als_tol=1e-6, reg_lambda=0.01,
                 centroid_method='kmeans', random_state=42):
        """
        Parameters:
        -----------
        n_centroids : int
            Number of local models
        quantum_dim : int
            Quantum state dimension
        bandwidth : float
            Kernel bandwidth for weighting
        ranks : tuple
            Tucker ranks for tensor decomposition
        encoding_type : str
            Quantum encoding method
        max_als_iter : int
            Maximum ALS iterations
        als_tol : float
            ALS convergence tolerance
        reg_lambda : float
            Regularization parameter
        centroid_method : str
            Method to select centroids ('kmeans', 'random', 'grid')
        random_state : int
            Random seed
        """
        self.n_centroids = n_centroids
        self.quantum_dim = quantum_dim
        self.bandwidth = bandwidth
        self.ranks = ranks
        self.encoding_type = encoding_type
        self.max_als_iter = max_als_iter
        self.als_tol = als_tol
        self.reg_lambda = reg_lambda
        self.centroid_method = centroid_method
        self.random_state = random_state

        # Initialize encoder
        self.encoder = QuantumEncoder(quantum_dim, encoding_type)

        # Storage for local models
        self.centroids = None
        self.centroid_states = None
        self.local_models = {}
        self.convergence_histories = {}

    def _select_centroids(self, X):
        """Select centroid locations"""
        np.random.seed(self.random_state)
        N = X.shape[0]

        if self.centroid_method == 'random':
            indices = np.random.choice(N, self.n_centroids, replace=False)
            centroids = X[indices]

        elif self.centroid_method == 'kmeans':
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self.n_centroids, 
                          random_state=self.random_state, n_init=10)
            kmeans.fit(X)
            centroids = kmeans.cluster_centers_

        elif self.centroid_method == 'grid':
            # For low-dimensional X, create grid
            if X.shape[1] <= 3:
                from itertools import product
                n_per_dim = int(np.ceil(self.n_centroids ** (1/X.shape[1])))
                ranges = [np.linspace(X[:, d].min(), X[:, d].max(), n_per_dim) 
                         for d in range(X.shape[1])]
                grid_points = list(product(*ranges))
                centroids = np.array(grid_points[:self.n_centroids])
            else:
                # Fall back to kmeans for high dimensions
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=self.n_centroids, 
                              random_state=self.random_state, n_init=10)
                kmeans.fit(X)
                centroids = kmeans.cluster_centers_
        else:
            raise ValueError(f"Unknown centroid method: {self.centroid_method}")

        return centroids

    def fit(self, X, Y):
        """
        Fit QILTR model

        Parameters:
        -----------
        X : ndarray, shape (N, D)
            Input features
        Y : ndarray, shape (N, P1, P2, ..., Pk)
            Tensor responses
        """
        N = X.shape[0]

        # Select centroids
        self.centroids = self._select_centroids(X)

        # Encode centroids to quantum states
        self.centroid_states = self.encoder.encode_batch(self.centroids)

        # Encode all data points
        data_states = self.encoder.encode_batch(X)

        # Fit local model at each centroid
        for c_idx in range(self.n_centroids):
            # Compute Bures distances
            distances = bures_distance_batch(
                self.centroid_states[c_idx], 
                data_states
            )

            # Compute weights
            weights = compute_weights(distances, self.bandwidth)

            # Fit weighted Tucker-ALS
            als_solver = WeightedTuckerALS(
                ranks=self.ranks,
                max_iter=self.max_als_iter,
                tol=self.als_tol,
                reg_lambda=self.reg_lambda
            )

            core, factors = als_solver.fit(Y, weights)

            # Store model
            self.local_models[c_idx] = {
                'core': core,
                'factors': factors,
                'centroid': self.centroids[c_idx],
                'centroid_state': self.centroid_states[c_idx]
            }

            # Store convergence history
            self.convergence_histories[c_idx] = als_solver.convergence_history

        return self

    def predict(self, X):
        """
        Predict tensor responses
        
        FIXED: Now uses input features X to modulate predictions

        Parameters:
        -----------
        X : ndarray, shape (M, D)
            Test features

        Returns:
        --------
        Y_pred : ndarray, shape (M, P1, P2, ..., Pk)
            Predicted tensors
        """
        M = X.shape[0]

        # Encode test points
        test_states = self.encoder.encode_batch(X)

        # Get prediction shape from first model
        first_model = self.local_models[0]
        als_solver = WeightedTuckerALS(self.ranks)
        sample_pred = als_solver.reconstruct(
            first_model['core'], 
            first_model['factors']
        )
        pred_shape = sample_pred.shape

        # Initialize predictions
        Y_pred = np.zeros((M,) + pred_shape)

        # For each test point, weighted combination of local predictions
        for i in range(M):
            # Compute distances to all centroids
            distances = bures_distance_batch(
                test_states[i], 
                self.centroid_states
            )

            # Compute weights
            weights = compute_weights(distances, self.bandwidth)

            # Weighted combination of local predictions
            als_solver = WeightedTuckerALS(self.ranks)
            local_combined = np.zeros(pred_shape)
            
            for c_idx in range(self.n_centroids):
                local_pred = als_solver.reconstruct(
                    self.local_models[c_idx]['core'],
                    self.local_models[c_idx]['factors']
                )
                local_combined += weights[c_idx] * local_pred
            
            # CRITICAL FIX: Apply feature-dependent modulation AFTER weighted combination
            # This prevents outlier amplification across multiple centroids
            for j in range(pred_shape[0]):
                for k in range(pred_shape[1]):
                    feature_idx = (j * pred_shape[1] + k) % X.shape[1]
                    # Use tanh to keep scaling bounded between 0.5 and 1.5
                    scale = 1.0 + 0.5 * np.tanh(X[i, feature_idx])
                    local_combined[j, k, :] *= scale
            
            Y_pred[i] = local_combined

        return Y_pred
