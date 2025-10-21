"""
Experiment 2: Computational Complexity Analysis
Measures computational complexity and validates polynomial-time scaling.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sys
sys.path.append('.')

from config import *
from src.qiltr import QILTR
from src.baselines import EuclideanLTR
from src.synthetic_data import generate_synthetic_tensor_regression

np.random.seed(RANDOM_SEED)
sns.set_style("whitegrid")

def measure_complexity():
    """
    Measure computational complexity by varying different parameters.
    """
    print("=" * 80)
    print("EXPERIMENT 2: COMPUTATIONAL COMPLEXITY ANALYSIS")
    print("=" * 80)

    results_list = []

    print("\n" + "-" * 60)
    print("Scaling with sample size N")
    print("-" * 60)

    sample_sizes = [100, 200, 500, 1000, 2000]

    for N in sample_sizes:
        print(f"\nTesting N={N}...")

        X, Y, _, _ = generate_synthetic_tensor_regression(
            n_samples=N,
            input_dim=20,
            tensor_shape=(5, 5, 5),
            tucker_ranks=(3, 3, 3),
            n_regions=3,
            noise_level=0.5,
            random_state=RANDOM_SEED
        )

        # QILTR
        start = time.time()
        qiltr = QILTR(
            n_centroids=5,
            quantum_dim=4,
            bandwidth=1.0,
            ranks=(3, 3, 3),
            max_als_iter=50,
            random_state=RANDOM_SEED
        )
        qiltr.fit(X, Y)
        qiltr_time = time.time() - start

        # Euclidean baseline
        start = time.time()
        eucl = EuclideanLTR(
            n_centroids=5,
            bandwidth=1.0,
            ranks=(3, 3, 3),
            max_als_iter=50,
            random_state=RANDOM_SEED
        )
        eucl.fit(X, Y)
        eucl_time = time.time() - start

        results_list.append({
            'Experiment': 'Sample_Size',
            'Parameter': 'N',
            'Value': N,
            'QILTR_Time': qiltr_time,
            'Euclidean_Time': eucl_time
        })

        print(f"  QILTR: {qiltr_time:.3f}s | Euclidean: {eucl_time:.3f}s")

    #
    # Experiment 4b: Scaling with quantum dimension d
    #
    print("\n" + "-" * 80)
    print("Experiment 4b: Scaling with quantum dimension d")
    print("-" * 80)

    quantum_dims = [2, 4, 8, 16]

    # Generate fixed dataset
    X, Y, _, _ = generate_synthetic_tensor_regression(
        n_samples=500,
        input_dim=20,
        tensor_shape=(5, 5, 5),
        tucker_ranks=(3, 3, 3),
        n_regions=3,
        noise_level=0.5,
        random_state=RANDOM_SEED
    )

    for d in quantum_dims:
        print(f"\nTesting d={d}...")

        start = time.time()
        qiltr = QILTR(
            n_centroids=5,
            quantum_dim=d,
            bandwidth=1.0,
            ranks=(3, 3, 3),
            max_als_iter=50,
            random_state=RANDOM_SEED
        )
        qiltr.fit(X, Y)
        qiltr_time = time.time() - start

        results_list.append({
            'Experiment': 'Quantum_Dim',
            'Parameter': 'd',
            'Value': d,
            'QILTR_Time': qiltr_time,
            'Euclidean_Time': np.nan  # N/A for quantum dimension
        })

        print(f"  QILTR: {qiltr_time:.3f}s")

    #
    # Experiment 4c: Scaling with input dimension D
    #
    print("\n" + "-" * 80)
    print("Experiment 4c: Scaling with input dimension D")
    print("-" * 80)

    input_dims = [10, 20, 50, 100]

    for D in input_dims:
        print(f"\nTesting D={D}...")

        # Generate data
        X, Y, _, _ = generate_synthetic_tensor_regression(
            n_samples=500,
            input_dim=D,
            tensor_shape=(5, 5, 5),
            tucker_ranks=(3, 3, 3),
            n_regions=3,
            noise_level=0.5,
            random_state=RANDOM_SEED
        )

        # QILTR
        start = time.time()
        qiltr = QILTR(
            n_centroids=5,
            quantum_dim=4,
            bandwidth=1.0,
            ranks=(3, 3, 3),
            max_als_iter=50,
            random_state=RANDOM_SEED
        )
        qiltr.fit(X, Y)
        qiltr_time = time.time() - start

        # Euclidean baseline
        start = time.time()
        eucl = EuclideanLTR(
            n_centroids=5,
            bandwidth=1.0,
            ranks=(3, 3, 3),
            max_als_iter=50,
            random_state=RANDOM_SEED
        )
        eucl.fit(X, Y)
        eucl_time = time.time() - start

        results_list.append({
            'Experiment': 'Input_Dim',
            'Parameter': 'D',
            'Value': D,
            'QILTR_Time': qiltr_time,
            'Euclidean_Time': eucl_time
        })

        print(f"  QILTR: {qiltr_time:.3f}s | Euclidean: {eucl_time:.3f}s")

    #
    # VISUALIZATION
    #
    print("\n" + "-" * 80)
    print("Creating visualizations...")

    results_df = pd.DataFrame(results_list)

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Scaling with N
    data_n = results_df[results_df['Experiment'] == 'Sample_Size']
    axes[0].plot(data_n['Value'], data_n['QILTR_Time'], 'o-', 
                label='QILTR', linewidth=2, markersize=8)
    axes[0].plot(data_n['Value'], data_n['Euclidean_Time'], 's-', 
                label='Euclidean-LTR', linewidth=2, markersize=8)
    axes[0].set_xlabel('Sample Size (N)', fontsize=12)
    axes[0].set_ylabel('Training Time (seconds)', fontsize=12)
    axes[0].set_title('Scaling with Sample Size', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Scaling with d
    data_d = results_df[results_df['Experiment'] == 'Quantum_Dim']
    axes[1].plot(data_d['Value'], data_d['QILTR_Time'], 'o-', 
                linewidth=2, markersize=8, color='#1f77b4')

    # Fit cubic curve (theoretical O(d^3))
    d_vals = data_d['Value'].values
    time_vals = data_d['QILTR_Time'].values
    coeffs = np.polyfit(d_vals**3, time_vals, 1)
    d_fit = np.linspace(d_vals.min(), d_vals.max(), 100)
    time_fit = coeffs[0] * d_fit**3 + coeffs[1]
    axes[1].plot(d_fit, time_fit, '--', label='O(d³) fit', 
                linewidth=2, color='red', alpha=0.7)

    axes[1].set_xlabel('Quantum Dimension (d)', fontsize=12)
    axes[1].set_ylabel('Training Time (seconds)', fontsize=12)
    axes[1].set_title('Scaling with Quantum Dimension', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Scaling with D
    data_D = results_df[results_df['Experiment'] == 'Input_Dim']
    axes[2].plot(data_D['Value'], data_D['QILTR_Time'], 'o-', 
                label='QILTR', linewidth=2, markersize=8)
    axes[2].plot(data_D['Value'], data_D['Euclidean_Time'], 's-', 
                label='Euclidean-LTR', linewidth=2, markersize=8)
    axes[2].set_xlabel('Input Dimension (D)', fontsize=12)
    axes[2].set_ylabel('Training Time (seconds)', fontsize=12)
    axes[2].set_title('Scaling with Input Dimension', fontsize=14, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIRS['figures'] + 'complexity_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIRS['figures']}complexity_analysis.png")
    plt.close()

    #
    # SAVE RESULTS
    #
    print("\n" + "-" * 80)
    print("Saving results...")

    results_df.to_csv(OUTPUT_DIRS['tables'] + 'complexity_results.csv', index=False)
    print(f"Saved: {OUTPUT_DIRS['tables']}complexity_results.csv")

    print("\n" + "=" * 80)
    print("EXPERIMENT 4-5 COMPLETED SUCCESSFULLY")
    print("=" * 80)

    return results_df

if __name__ == '__main__':
    results_df = measure_complexity()
    print("\nComplexity Results:")
    print(results_df.to_string(index=False))
