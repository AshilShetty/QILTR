"""
Baseline methods for comparison with QILTR
"""

import numpy as np
from .distances import euclidean_distance_batch, compute_weights
from .als_solver import WeightedTuckerALS

class EuclideanLTR:
    """Local Tensor Regression with Euclidean distance kernel"""

    def __init__(self, n_centroids=10, bandwidth=1.0, ranks=(3, 3, 3),
                 max_als_iter=100, als_tol=1e-6, reg_lambda=0.01,
                 centroid_method='kmeans', random_state=42):
        self.n_centroids = n_centroids
        self.bandwidth = bandwidth
        self.ranks = ranks
        self.max_als_iter = max_als_iter
        self.als_tol = als_tol
        self.reg_lambda = reg_lambda
        self.centroid_method = centroid_method
        self.random_state = random_state

        self.centroids = None
        self.local_models = {}
        self.convergence_histories = {}

    def _select_centroids(self, X):
        """Same centroid selection as QILTR"""
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

        else:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self.n_centroids, 
                          random_state=self.random_state, n_init=10)
            kmeans.fit(X)
            centroids = kmeans.cluster_centers_

        return centroids

    def fit(self, X, Y):
        """Fit Euclidean-kernel local tensor regression"""
        N = X.shape[0]

        # Select centroids
        self.centroids = self._select_centroids(X)

        # Fit local model at each centroid
        for c_idx in range(self.n_centroids):
            # Compute Euclidean distances
            distances = euclidean_distance_batch(
                self.centroids[c_idx], 
                X
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
                'centroid': self.centroids[c_idx]
            }

            # Store convergence history
            self.convergence_histories[c_idx] = als_solver.convergence_history

        return self

    def predict(self, X):
        """Predict tensor responses"""
        M = X.shape[0]

        # Get prediction shape
        first_model = self.local_models[0]
        als_solver = WeightedTuckerALS(self.ranks)
        sample_pred = als_solver.reconstruct(
            first_model['core'], 
            first_model['factors']
        )
        pred_shape = sample_pred.shape

        # Initialize predictions
        Y_pred = np.zeros((M,) + pred_shape)

        # For each test point
        for i in range(M):
            # Compute distances to all centroids
            distances = euclidean_distance_batch(
                X[i], 
                self.centroids
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
            
            # Apply feature-dependent modulation AFTER weighted combination
            # This prevents outlier amplification across multiple centroids
            for j in range(pred_shape[0]):
                for k in range(pred_shape[1]):
                    feature_idx = (j * pred_shape[1] + k) % X.shape[1]
                    # Use tanh to keep scaling bounded between 0.5 and 1.5
                    scale = 1.0 + 0.5 * np.tanh(X[i, feature_idx])
                    local_combined[j, k, :] *= scale
            
            Y_pred[i] = local_combined

        return Y_pred


class GlobalTuckerRegression:
    """Global Tucker regression (no local weighting)"""

    def __init__(self, ranks=(3, 3, 3), max_als_iter=100, 
                 als_tol=1e-6, reg_lambda=0.01, random_state=42):
        self.ranks = ranks
        self.max_als_iter = max_als_iter
        self.als_tol = als_tol
        self.reg_lambda = reg_lambda
        self.random_state = random_state

        self.core = None
        self.factors = None
        self.convergence_history = []

    def fit(self, X, Y):
        """Fit global Tucker regression"""
        N = X.shape[0]

        # Uniform weights (global model)
        weights = np.ones(N) / N

        # Fit weighted Tucker-ALS
        als_solver = WeightedTuckerALS(
            ranks=self.ranks,
            max_iter=self.max_als_iter,
            tol=self.als_tol,
            reg_lambda=self.reg_lambda
        )

        self.core, self.factors = als_solver.fit(Y, weights)
        self.convergence_history = als_solver.convergence_history

        return self

    def predict(self, X):
        """Predict tensor responses (same for all inputs)"""
        M = X.shape[0]

        # Reconstruct global tensor
        als_solver = WeightedTuckerALS(self.ranks)
        global_pred = als_solver.reconstruct(self.core, self.factors)

        # Return same prediction for all inputs
        Y_pred = np.tile(global_pred, (M, 1, 1, 1))

        return Y_pred
