"""
Real-World and Semi-Synthetic Data Loaders

This module provides data loading utilities for validating QILTR on real-world datasets.
Includes both semi-synthetic bridges (MNIST tensor regions) and real dataset loaders.

Author: QILTR Research Team
Date: October 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings

try:
    from sklearn.datasets import fetch_openml
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("sklearn not available - some datasets may not load")


class RealWorldDataLoader:
    """
    Unified interface for loading various real-world and semi-synthetic datasets.
    
    Supported datasets:
    - 'mnist_tensor': MNIST digits as tensor regression (semi-synthetic)
    - 'qm9': Quantum chemistry molecular properties (requires download)
    - 'ucf101': Video action recognition (requires download)
    
    All loaders return standardized format:
        X_train, X_test, Y_train, Y_test
    """
    
    def __init__(self, dataset_name, data_dir='./data', random_state=42):
        """
        Initialize data loader.
        
        Args:
            dataset_name: Name of dataset to load
            data_dir: Directory for storing/loading data
            random_state: Random seed for reproducibility
        """
        self.dataset_name = dataset_name
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        
        self.loaders = {
            'mnist_tensor': self.load_mnist_tensor,
            'qm9': self.load_qm9,
            'ucf101': self.load_ucf101_placeholder
        }
    
    def load(self, **kwargs):
        """Load specified dataset."""
        if self.dataset_name not in self.loaders:
            raise ValueError(f"Unknown dataset: {self.dataset_name}. "
                           f"Available: {list(self.loaders.keys())}")
        
        return self.loaders[self.dataset_name](**kwargs)
    
    # ========================================================================
    # SEMI-SYNTHETIC: MNIST Tensor Regions (RECOMMENDED FOR QUICK VALIDATION)
    # ========================================================================
    
    def load_mnist_tensor(self, n_samples=2000, tensor_shape=(5, 5, 3),
                         test_size=0.2, add_noise=True):
        """
        Load MNIST as tensor regression problem.
        
        Task: Predict local tensor regions of digit images from global features.
        This creates a bridge between synthetic and real-world validation.
        
        Args:
            n_samples: Number of samples to use
            tensor_shape: Shape of output tensors (extracted from images)
            test_size: Fraction for test set
            add_noise: Whether to add observation noise
        
        Returns:
            Dictionary with X_train, X_test, Y_train, Y_test, metadata
        """
        print("Loading MNIST Tensor Regression Dataset...")
        print(f"  Samples: {n_samples}")
        print(f"  Tensor shape: {tensor_shape}")
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn required for MNIST. Install: pip install scikit-learn")
        
        # Load MNIST
        print("  Downloading MNIST (first time may take a moment)...")
        mnist = fetch_openml('mnist_784', version=1, parser='auto')
        X_mnist = mnist.data.to_numpy() if hasattr(mnist.data, 'to_numpy') else mnist.data
        y_mnist = mnist.target.to_numpy() if hasattr(mnist.target, 'to_numpy') else mnist.target
        
        # Reshape to images
        images = X_mnist.reshape(-1, 28, 28)
        
        # Select subset
        indices = np.random.RandomState(self.random_state).choice(
            len(images), min(n_samples, len(images)), replace=False
        )
        images = images[indices]
        labels = y_mnist[indices]
        
        print(f"  Processing {len(images)} images...")
        
        # Create input features (global image statistics)
        X = self._extract_global_features(images)
        
        # Create output tensors (local image regions)
        Y = self._extract_local_tensors(images, tensor_shape)
        
        # Add noise if requested
        if add_noise:
            noise = np.random.RandomState(self.random_state).normal(
                0, 0.05, Y.shape
            )
            Y = Y + noise
        
        # Split
        X_train, X_test, Y_train, Y_test, labels_train, labels_test = train_test_split(
            X, Y, labels, test_size=test_size, random_state=self.random_state
        )
        
        # Normalize inputs
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        print(f"  Training set: {len(X_train)} samples")
        print(f"  Test set: {len(X_test)} samples")
        print(f"  Input dimension: {X_train.shape[1]}")
        print(f"  Output tensor shape: {Y_train.shape[1:]}")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'Y_train': Y_train,
            'Y_test': Y_test,
            'labels_train': labels_train,
            'labels_test': labels_test,
            'metadata': {
                'dataset': 'MNIST Tensor Regression',
                'n_train': len(X_train),
                'n_test': len(X_test),
                'input_dim': X_train.shape[1],
                'tensor_shape': tensor_shape,
                'description': 'Predict local image regions from global features',
                'source': 'MNIST via OpenML',
                'citation': 'LeCun et al. (1998) Gradient-based learning applied to document recognition'
            }
        }
    
    def _extract_global_features(self, images):
        """Extract global statistical features from images."""
        n_images = len(images)
        features = []
        
        for i in range(n_images):
            img = images[i]
            
            # Global statistics
            feat = [
                np.mean(img),                    # Mean intensity
                np.std(img),                     # Std intensity
                np.median(img),                  # Median
                np.max(img),                     # Max
                np.min(img),                     # Min
                np.percentile(img, 25),          # Q1
                np.percentile(img, 75),          # Q3
            ]
            
            # Moments
            feat.extend([
                np.mean(img ** 2),               # Second moment
                np.mean(img ** 3),               # Third moment (skewness-related)
            ])
            
            # Spatial moments (center of mass)
            y_coords, x_coords = np.mgrid[0:28, 0:28]
            total_mass = np.sum(img) + 1e-10
            feat.extend([
                np.sum(x_coords * img) / total_mass,  # X center of mass
                np.sum(y_coords * img) / total_mass,  # Y center of mass
            ])
            
            # Quadrant means
            h, w = 14, 14
            feat.extend([
                np.mean(img[:h, :w]),            # Top-left
                np.mean(img[:h, w:]),            # Top-right
                np.mean(img[h:, :w]),            # Bottom-left
                np.mean(img[h:, w:]),            # Bottom-right
            ])
            
            # Gradient statistics
            grad_y = np.abs(np.diff(img, axis=0)).mean()
            grad_x = np.abs(np.diff(img, axis=1)).mean()
            feat.extend([grad_x, grad_y])
            
            features.append(feat)
        
        return np.array(features)
    
    def _extract_local_tensors(self, images, tensor_shape):
        """
        Extract local tensor regions from images.
        
        Creates 3D tensors by sampling multiple regions from each image.
        """
        n_images = len(images)
        d1, d2, d3 = tensor_shape
        
        tensors = np.zeros((n_images, d1, d2, d3))
        
        for i in range(n_images):
            img = images[i]
            
            # Extract d3 different regions of size d1 x d2
            for k in range(d3):
                # Random offset for diversity
                max_y = 28 - d1
                max_x = 28 - d2
                
                if max_y > 0 and max_x > 0:
                    # Stratified sampling to ensure coverage
                    y_start = int((k / d3) * max_y)
                    x_start = int((k / d3) * max_x)
                    
                    region = img[y_start:y_start+d1, x_start:x_start+d2]
                    
                    # Normalize region
                    region = region / 255.0
                    
                    tensors[i, :, :, k] = region
                else:
                    # If requested size too large, downsample
                    from scipy.ndimage import zoom
                    region = zoom(img, (d1/28, d2/28), order=1)
                    tensors[i, :, :, k] = region / 255.0
        
        return tensors
    
    # ========================================================================
    # REAL-WORLD: QM9 Quantum Chemistry
    # ========================================================================
    
    def load_qm9(self, subset_size=1000, test_size=0.2, tensor_shape=(5, 5, 5)):
        """
        Load QM9 quantum chemistry dataset for tensor regression.
        
        Uses genuine molecular data from QM9 dataset:
        - Input (X): Molecular properties (19 quantum mechanical properties)
        - Output (Y): 3D spatial grid constructed from atomic positions
        
        Args:
            subset_size: Number of molecules to use (full dataset is ~134k)
            test_size: Fraction for test set
            tensor_shape: Shape of output 3D tensors
        
        Returns:
            Dictionary with X_train, X_test, Y_train, Y_test, metadata
        """
        print("Loading QM9 Quantum Chemistry Dataset...")
        print(f"  Subset size: {subset_size}")
        print(f"  Tensor shape: {tensor_shape}")
        
        try:
            from torch_geometric.datasets import QM9
            print("  PyTorch Geometric found ✓")
        except ImportError:
            raise ImportError(
                "QM9 requires torch_geometric. Install with:\n"
                "  pip install torch torch-geometric"
            )
        
        # Download QM9 dataset
        print("  Downloading QM9 (first time may take several minutes)...")
        qm9_dir = self.data_dir / 'QM9'
        qm9_dir.mkdir(parents=True, exist_ok=True)
        
        dataset = QM9(root=str(qm9_dir))
        print(f"  ✓ QM9 loaded: {len(dataset)} molecules available")
        
        # Use subset
        subset_size = min(subset_size, len(dataset))
        indices = np.random.RandomState(self.random_state).choice(
            len(dataset), subset_size, replace=False
        )
        print(f"  Using {subset_size} molecules")
        
        # Process molecules
        print("  Processing molecules...")
        X_list = []
        Y_list = []
        
        for idx in indices:
            if len(X_list) % 100 == 0 and len(X_list) > 0:
                print(f"    Processed {len(X_list)}/{subset_size}...")
            
            data = dataset[int(idx)]
            
            # Input: Use the 19 molecular properties directly
            # Properties: mu, alpha, homo, lumo, gap, r2, zpve, U0, U, H, G, Cv, ...
            mol_properties = data.y.numpy().flatten()  # Shape: (19,)
            X_list.append(mol_properties)
            
            # Output: Construct 3D tensor from atomic positions
            # This creates a spatial grid representation of the molecule
            pos = data.pos.numpy()  # Atomic positions (N_atoms, 3)
            z = data.z.numpy()      # Atomic numbers (N_atoms,)
            
            # Create 3D grid from atomic positions
            tensor_3d = self._positions_to_tensor(pos, z, tensor_shape)
            Y_list.append(tensor_3d)
        
        X = np.array(X_list)
        Y = np.array(Y_list)
        
        print(f"  ✓ Processed {len(X)} molecules")
        print(f"    Input shape: {X.shape}")
        print(f"    Output shape: {Y.shape}")
        
        # Train/test split
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=test_size, random_state=self.random_state
        )
        
        # Normalize inputs
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        print(f"\n  Final dataset:")
        print(f"    Train: {len(X_train)} molecules")
        print(f"    Test:  {len(X_test)} molecules")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'Y_train': Y_train,
            'Y_test': Y_test,
            'metadata': {
                'dataset': 'QM9 Quantum Chemistry',
                'n_train': len(X_train),
                'n_test': len(X_test),
                'input_dim': X_train.shape[1],
                'tensor_shape': tensor_shape,
                'description': 'Real quantum chemistry data: molecular properties → 3D structure',
                'source': 'QM9 via PyTorch Geometric',
                'citation': 'Ramakrishnan et al. (2014) Scientific Data 1:140022'
            }
        }
    
    def _positions_to_tensor(self, pos, z, tensor_shape):
        """
        Convert atomic positions to 3D tensor grid.
        
        Simple mapping: places atoms on grid weighted by atomic number.
        
        Args:
            pos: Atomic positions (N_atoms, 3)
            z: Atomic numbers (N_atoms,)
            tensor_shape: Output grid shape (d1, d2, d3)
        
        Returns:
            3D numpy array
        """
        d1, d2, d3 = tensor_shape
        tensor = np.zeros(tensor_shape, dtype=np.float32)
        
        if len(pos) == 0:
            return tensor
        
        # Normalize positions to [0, 1]
        pos_min = pos.min(axis=0)
        pos_max = pos.max(axis=0)
        pos_range = pos_max - pos_min + 1e-10
        pos_norm = (pos - pos_min) / pos_range
        
        # Map atoms to grid
        for i in range(len(pos)):
            ix = int(pos_norm[i, 0] * (d1 - 1))
            iy = int(pos_norm[i, 1] * (d2 - 1))
            iz = int(pos_norm[i, 2] * (d3 - 1))
            
            ix = np.clip(ix, 0, d1 - 1)
            iy = np.clip(iy, 0, d2 - 1)
            iz = np.clip(iz, 0, d3 - 1)
            
            # Weight by atomic number (electron count proxy)
            tensor[ix, iy, iz] += float(z[i])
        
        # Normalize
        if tensor.max() > 0:
            tensor = tensor / tensor.max()
        
        return tensor
    
    # ========================================================================
    # REAL-WORLD: UCF101 Video
    # ========================================================================
    
    def load_ucf101_placeholder(self):
        """
        Placeholder for UCF101 video dataset.
        
        See docs/REAL_WORLD_DATASETS.md for implementation instructions.
        """
        print("UCF101 loader is a placeholder - see docs/REAL_WORLD_DATASETS.md")
        raise NotImplementedError(
            "UCF101 loader requires manual implementation. "
            "See docs/REAL_WORLD_DATASETS.md for detailed instructions."
        )


# ============================================================================
# Helper Functions for Custom Datasets
# ============================================================================

def create_tensor_regression_dataset(X, Y_images, tensor_shape, 
                                    extraction_method='patches'):
    """
    Generic function to convert image-like data to tensor regression format.
    
    Args:
        X: Input features (n_samples, n_features)
        Y_images: Image outputs (n_samples, height, width) or (n_samples, height, width, channels)
        tensor_shape: Desired output tensor shape
        extraction_method: 'patches', 'downsample', or 'moments'
    
    Returns:
        X, Y_tensors where Y_tensors has shape (n_samples, *tensor_shape)
    """
    # Implementation for custom datasets
    pass


def load_custom_csv_tensor_dataset(csv_path, input_cols, tensor_cols, 
                                  tensor_shape, test_size=0.2):
    """
    Load tensor regression dataset from CSV file.
    
    Expected CSV format:
        - Rows: samples
        - Columns: input_features, tensor_flattened_values
    
    Args:
        csv_path: Path to CSV file
        input_cols: List of column names for inputs X
        tensor_cols: List of column names for tensor values (will be reshaped)
        tensor_shape: Shape to reshape tensor values into
        test_size: Fraction for test set
    
    Returns:
        Dictionary with X_train, X_test, Y_train, Y_test
    """
    df = pd.read_csv(csv_path)
    
    X = df[input_cols].values
    Y_flat = df[tensor_cols].values
    
    # Reshape to tensors
    n_samples = len(X)
    Y = Y_flat.reshape(n_samples, *tensor_shape)
    
    # Split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=42
    )
    
    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'Y_train': Y_train,
        'Y_test': Y_test,
        'metadata': {
            'source': csv_path,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'input_dim': X_train.shape[1],
            'tensor_shape': tensor_shape
        }
    }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("Real-World Data Loader Demo")
    print("="*80)
    
    # Example 1: MNIST Tensor (semi-synthetic, works out of the box)
    print("\n1. Loading MNIST Tensor Dataset...")
    loader = RealWorldDataLoader('mnist_tensor', random_state=42)
    data = loader.load(n_samples=500, tensor_shape=(5, 5, 3), test_size=0.2)
    
    print(f"\nDataset loaded successfully!")
    print(f"  X_train shape: {data['X_train'].shape}")
    print(f"  Y_train shape: {data['Y_train'].shape}")
    print(f"  X_test shape: {data['X_test'].shape}")
    print(f"  Y_test shape: {data['Y_test'].shape}")
    
    print("\nMetadata:")
    for key, value in data['metadata'].items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*80)
    print("To use in experiments, see: experiments/exp8_realworld_validation.py")
    print("="*80)
