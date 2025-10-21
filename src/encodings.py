"""
Quantum state encoding methods for QILTR
"""

import numpy as np
from scipy.linalg import sqrtm

class QuantumEncoder:
    """Encode classical feature vectors to quantum density matrices"""

    def __init__(self, quantum_dim=4, encoding_type='amplitude'):
        """
        Parameters:
        -----------
        quantum_dim : int
            Dimension of quantum state space
        encoding_type : str
            Type of encoding: 'amplitude', 'gaussian', 'parametric'
        """
        self.d = quantum_dim
        self.encoding_type = encoding_type

    def encode(self, x):
        """
        Encode feature vector x to density matrix

        Parameters:
        -----------
        x : array-like, shape (D,)
            Input feature vector

        Returns:
        --------
        rho : ndarray, shape (d, d)
            Density matrix (positive semidefinite, trace-1)
        """
        if self.encoding_type == 'amplitude':
            return self._amplitude_encoding(x)
        elif self.encoding_type == 'gaussian':
            return self._gaussian_encoding(x)
        elif self.encoding_type == 'parametric':
            return self._parametric_encoding(x)
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type}")

    def _amplitude_encoding(self, x):
        """Amplitude encoding: embed into quantum state"""
        x = np.asarray(x).flatten()

        # Pad or truncate to match quantum dimension
        if len(x) < self.d:
            x_padded = np.zeros(self.d)
            x_padded[:len(x)] = x
        else:
            x_padded = x[:self.d]

        # Normalize to unit vector
        norm = np.linalg.norm(x_padded)
        if norm < 1e-10:
            x_padded = np.ones(self.d) / np.sqrt(self.d)
        else:
            x_padded = x_padded / norm

        # Create pure state density matrix
        psi = x_padded.reshape(-1, 1)
        rho = psi @ psi.conj().T

        return rho

    def _gaussian_encoding(self, x):
        """Gaussian kernel encoding with basis states"""
        x = np.asarray(x).flatten()

        # Create basis states (equally spaced in feature space projection)
        basis_centers = np.linspace(-1, 1, self.d)

        # Compute Gaussian weights
        x_proj = np.mean(x) if len(x) > 0 else 0  # Simple projection
        weights = np.exp(-0.5 * (basis_centers - x_proj)**2)
        weights = weights / np.sum(weights)

        # Diagonal density matrix
        rho = np.diag(weights)

        return rho

    def _parametric_encoding(self, x):
        """Parametric encoding using feature-dependent rotation"""
        x = np.asarray(x).flatten()

        # Use features to parameterize a density matrix
        # For higher dimensions, use diagonal with feature-dependent weights
        weights = np.abs(np.fft.fft(x, n=self.d))[:self.d]
        weights = weights / np.sum(weights)
        rho = np.diag(weights)

        return rho

    def encode_batch(self, X):
        """
        Encode batch of feature vectors

        Parameters:
        -----------
        X : ndarray, shape (N, D)
            Batch of feature vectors

        Returns:
        --------
        rhos : ndarray, shape (N, d, d)
            Batch of density matrices
        """
        N = X.shape[0]
        rhos = np.zeros((N, self.d, self.d), dtype=complex)

        for i in range(N):
            rhos[i] = self.encode(X[i])

        return rhos
