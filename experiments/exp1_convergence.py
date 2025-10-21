"""
Experiment 1-3: Convergence Analysis
Validates convergence properties of QILTR under different scenarios.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import pickle
import time
import sys
sys.path.append('.')

from config import *
from src.qiltr import QILTR
from src.baselines import EuclideanLTR, GlobalTuckerRegression
from src.synthetic_data import generate_synthetic_tensor_regression, add_outliers
from src.metrics import compute_all_metrics, convergence_rate_analysis

np.random.seed(RANDOM_SEED)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def run_single_scenario(scenario_name, scenario_config, common_params):
    """
    Run convergence experiment for a single data scenario.
    
    Parameters:
    -----------
    scenario_name : str
        Name of the scenario
    scenario_config : dict
        Configuration for data generation
    common_params : dict
        Common model parameters
        
    Returns:
    --------
    results : dict
        Results for all methods
    """
    print("\n" + "=" * 80)
    print(f"SCENARIO: {scenario_config['name']}")
    print("=" * 80)
    
    # Generate data based on scenario
    print(f"\nGenerating {scenario_name} data...")
    X, Y, region_labels, true_tensors = generate_synthetic_tensor_regression(
        n_samples=1000,
        input_dim=20,
        tensor_shape=(5, 5, 5),
        tucker_ranks=(3, 3, 3),
        n_regions=scenario_config.get('n_regions', 3),
        noise_level=0.5,
        random_state=RANDOM_SEED,
        ill_conditioned=scenario_config.get('ill_conditioned', False),
        condition_number=scenario_config.get('condition_number', 100.0),
        high_nonstationarity=scenario_config.get('high_nonstationarity', False),
        region_diversity_scale=scenario_config.get('region_diversity_scale', 1.0)
    )
    
    # Add outliers if specified
    if scenario_config.get('outlier_fraction', 0.0) > 0:
        print(f"Adding {scenario_config['outlier_fraction']*100:.0f}% outliers with magnitude {scenario_config['outlier_magnitude']}")
        X, Y, outlier_indices = add_outliers(
            X, Y,
            outlier_fraction=scenario_config['outlier_fraction'],
            outlier_magnitude=scenario_config['outlier_magnitude'],
            random_state=RANDOM_SEED
        )
    
    print(f"Data shape: X={X.shape}, Y={Y.shape}, n_regions={scenario_config.get('n_regions', 3)}")
    
    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=RANDOM_SEED
    )
    
    results = {}
    
    #
    # 1. QILTR
    #
    print("\n" + "-" * 80)
    print("Training QILTR...")
    start_time = time.time()
    
    qiltr = QILTR(
        quantum_dim=4,
        encoding_type='amplitude',
        **common_params
    )
    qiltr.fit(X_train, Y_train)
    
    qiltr_time = time.time() - start_time
    print(f"QILTR training time: {qiltr_time:.2f}s")
    
    # Predict
    Y_pred_qiltr = qiltr.predict(X_test)
    
    # Metrics
    qiltr_metrics = compute_all_metrics(Y_test, Y_pred_qiltr)
    print(f"QILTR Test MSE: {qiltr_metrics['mse']:.6f}")
    
    results['QILTR'] = {
        'metrics': qiltr_metrics,
        'time': qiltr_time,
        'model': qiltr
    }
    
    #
    # 2. Euclidean LTR (Baseline)
    #
    print("\n" + "-" * 80)
    print("Training Euclidean-LTR...")
    start_time = time.time()
    
    euclidean_ltr = EuclideanLTR(**common_params)
    euclidean_ltr.fit(X_train, Y_train)
    
    euclidean_time = time.time() - start_time
    print(f"Euclidean-LTR training time: {euclidean_time:.2f}s")
    
    # Predict
    Y_pred_euclidean = euclidean_ltr.predict(X_test)
    
    # Metrics
    euclidean_metrics = compute_all_metrics(Y_test, Y_pred_euclidean)
    print(f"Euclidean-LTR Test MSE: {euclidean_metrics['mse']:.6f}")
    
    results['Euclidean-LTR'] = {
        'metrics': euclidean_metrics,
        'time': euclidean_time,
        'model': euclidean_ltr
    }
    
    #
    # 3. Global Tucker Regression (Baseline)
    #
    print("\n" + "-" * 80)
    print("Training Global Tucker Regression...")
    start_time = time.time()
    
    global_tr = GlobalTuckerRegression(
        ranks=(3, 3, 3),
        max_als_iter=100,
        als_tol=1e-6,
        random_state=RANDOM_SEED
    )
    global_tr.fit(X_train, Y_train)
    
    global_time = time.time() - start_time
    print(f"Global Tucker training time: {global_time:.2f}s")
    
    # Predict
    Y_pred_global = global_tr.predict(X_test)
    
    # Metrics
    global_metrics = compute_all_metrics(Y_test, Y_pred_global)
    print(f"Global Tucker Test MSE: {global_metrics['mse']:.6f}")
    
    results['Global-Tucker'] = {
        'metrics': global_metrics,
        'time': global_time,
        'model': global_tr
    }
    
    # Print summary comparison
    print("\n" + "-" * 80)
    print(f"SUMMARY for {scenario_config['name']}:")
    print("-" * 80)
    for method in ['QILTR', 'Euclidean-LTR', 'Global-Tucker']:
        mse = results[method]['metrics']['mse']
        time_taken = results[method]['time']
        print(f"{method:20s} | MSE: {mse:.6f} | Time: {time_taken:.2f}s")
    
    # Calculate performance advantage
    qiltr_mse = results['QILTR']['metrics']['mse']
    eucl_mse = results['Euclidean-LTR']['metrics']['mse']
    improvement = ((eucl_mse - qiltr_mse) / eucl_mse) * 100
    print(f"\nQILTR vs Euclidean-LTR: {improvement:+.2f}% {'improvement' if improvement > 0 else 'degradation'}")
    
    return results


def run_convergence_experiment():
    """
    Main convergence experiment comparing QILTR vs baselines.
    Tests QILTR under challenging data conditions.
    """
    print("=" * 80)
    print("EXPERIMENT 1-3: CONVERGENCE ANALYSIS")
    print("Testing QILTR under challenging data conditions")
    print("=" * 80)
    
    # Common parameters for all scenarios
    common_params = {
        'n_centroids': 10,
        'bandwidth': 1.0,
        'ranks': (3, 3, 3),
        'max_als_iter': 100,
        'als_tol': 1e-6,
        'random_state': RANDOM_SEED
    }
    
    # Storage for all scenario results
    all_scenarios_results = {}
    all_summaries = []
    
    # Run each enabled scenario
    for scenario_key, scenario_config in CHALLENGING_SCENARIOS.items():
        if scenario_key == 'run_all_scenarios':
            continue
            
        if not scenario_config.get('enabled', False):
            print(f"\nSkipping scenario: {scenario_config['name']}")
            continue
        
        # Run the scenario
        scenario_results = run_single_scenario(
            scenario_key, 
            scenario_config, 
            common_params
        )
        
        # Store results
        all_scenarios_results[scenario_key] = scenario_results
        
        # Create summary row
        for method in ['QILTR', 'Euclidean-LTR', 'Global-Tucker']:
            summary_row = {
                'Scenario': scenario_config['name'],
                'Method': method,
                'Test MSE': scenario_results[method]['metrics']['mse'],
                'Test MAE': scenario_results[method]['metrics']['mae'],
                'Relative Error': scenario_results[method]['metrics']['relative_error'],
                'R² Score': scenario_results[method]['metrics']['r2_score'],
                'Training Time (s)': scenario_results[method]['time']
            }
            all_summaries.append(summary_row)
    
    #
    # COMPREHENSIVE VISUALIZATION
    #
    print("\n" + "=" * 80)
    print("Creating comprehensive visualizations...")
    print("=" * 80)
    
    # Create comparison figure across scenarios
    scenarios_list = [k for k in all_scenarios_results.keys()]
    methods = ['QILTR', 'Euclidean-LTR', 'Global-Tucker']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('QILTR Performance Across Challenging Scenarios', fontsize=16, fontweight='bold')
    
    # Plot 1: MSE comparison
    ax = axes[0, 0]
    x = np.arange(len(scenarios_list))
    width = 0.25
    
    for i, method in enumerate(methods):
        mse_values = [all_scenarios_results[s][method]['metrics']['mse'] 
                     for s in scenarios_list]
        ax.bar(x + i*width, mse_values, width, label=method)
    
    ax.set_xlabel('Scenario', fontsize=11)
    ax.set_ylabel('Test MSE', fontsize=11)
    ax.set_title('Test MSE Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([CHALLENGING_SCENARIOS[s]['name'] for s in scenarios_list], 
                       rotation=15, ha='right', fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Relative error comparison
    ax = axes[0, 1]
    for i, method in enumerate(methods):
        rel_err_values = [all_scenarios_results[s][method]['metrics']['relative_error'] 
                         for s in scenarios_list]
        ax.bar(x + i*width, rel_err_values, width, label=method)
    
    ax.set_xlabel('Scenario', fontsize=11)
    ax.set_ylabel('Relative Error', fontsize=11)
    ax.set_title('Relative Error Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([CHALLENGING_SCENARIOS[s]['name'] for s in scenarios_list], 
                       rotation=15, ha='right', fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Training time comparison
    ax = axes[1, 0]
    for i, method in enumerate(methods):
        time_values = [all_scenarios_results[s][method]['time'] 
                      for s in scenarios_list]
        ax.bar(x + i*width, time_values, width, label=method)
    
    ax.set_xlabel('Scenario', fontsize=11)
    ax.set_ylabel('Training Time (s)', fontsize=11)
    ax.set_title('Training Time Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([CHALLENGING_SCENARIOS[s]['name'] for s in scenarios_list], 
                       rotation=15, ha='right', fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: QILTR advantage over Euclidean (percentage improvement)
    ax = axes[1, 1]
    improvements = []
    for s in scenarios_list:
        qiltr_mse = all_scenarios_results[s]['QILTR']['metrics']['mse']
        eucl_mse = all_scenarios_results[s]['Euclidean-LTR']['metrics']['mse']
        improvement = ((eucl_mse - qiltr_mse) / eucl_mse) * 100
        improvements.append(improvement)
    
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    ax.bar(x, improvements, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Scenario', fontsize=11)
    ax.set_ylabel('QILTR Improvement over Euclidean (%)', fontsize=11)
    ax.set_title('QILTR Performance Advantage', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([CHALLENGING_SCENARIOS[s]['name'] for s in scenarios_list], 
                       rotation=15, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIRS['figures'] + 'challenging_scenarios_comparison.png', 
                dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIRS['figures']}challenging_scenarios_comparison.png")
    plt.close()
    
    #
    # SAVE COMPREHENSIVE RESULTS
    #
    print("\nSaving comprehensive results...")
    
    # Save summary table
    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(OUTPUT_DIRS['tables'] + 'challenging_scenarios_summary.csv', index=False)
    print(f"Saved: {OUTPUT_DIRS['tables']}challenging_scenarios_summary.csv")
    
    # Save detailed results
    with open(OUTPUT_DIRS['results'] + 'challenging_scenarios_results.pkl', 'wb') as f:
        pickle.dump(all_scenarios_results, f)
    print(f"Saved: {OUTPUT_DIRS['results']}challenging_scenarios_results.pkl")
    
    #
    # GENERATE MARKDOWN SUMMARY
    #
    print("\nGenerating publication-ready summary...")
    
    markdown_summary = "# QILTR Performance Under Challenging Conditions\n\n"
    markdown_summary += "## Summary of Results\n\n"
    
    for scenario_key in scenarios_list:
        scenario_name = CHALLENGING_SCENARIOS[scenario_key]['name']
        markdown_summary += f"### {scenario_name}\n\n"
        
        results = all_scenarios_results[scenario_key]
        
        markdown_summary += "| Method | Test MSE | Test MAE | Relative Error | Training Time (s) |\n"
        markdown_summary += "|--------|----------|----------|----------------|-------------------|\n"
        
        for method in methods:
            mse = results[method]['metrics']['mse']
            mae = results[method]['metrics']['mae']
            rel_err = results[method]['metrics']['relative_error']
            time_val = results[method]['time']
            markdown_summary += f"| {method} | {mse:.6f} | {mae:.6f} | {rel_err:.6f} | {time_val:.2f} |\n"
        
        # Add improvement note
        qiltr_mse = results['QILTR']['metrics']['mse']
        eucl_mse = results['Euclidean-LTR']['metrics']['mse']
        improvement = ((eucl_mse - qiltr_mse) / eucl_mse) * 100
        
        if improvement > 0:
            markdown_summary += f"\n**Key Finding**: QILTR shows {improvement:.2f}% improvement over Euclidean-LTR in this challenging scenario.\n\n"
        else:
            markdown_summary += f"\n**Key Finding**: Euclidean-LTR performs {-improvement:.2f}% better than QILTR in this scenario.\n\n"
    
    # Save markdown
    with open(OUTPUT_DIRS['results'] + 'publication_summary.md', 'w') as f:
        f.write(markdown_summary)
    print(f"Saved: {OUTPUT_DIRS['results']}publication_summary.md")
    
    #
    # PRINT TERMINAL SUMMARY
    #
    print("\n" + "=" * 80)
    print("COMPREHENSIVE ANALYSIS SUMMARY")
    print("=" * 80)
    
    for scenario_key in scenarios_list:
        scenario_name = CHALLENGING_SCENARIOS[scenario_key]['name']
        results = all_scenarios_results[scenario_key]
        
        print(f"\n{scenario_name}:")
        print("-" * 80)
        
        qiltr_mse = results['QILTR']['metrics']['mse']
        eucl_mse = results['Euclidean-LTR']['metrics']['mse']
        global_mse = results['Global-Tucker']['metrics']['mse']
        
        improvement_vs_eucl = ((eucl_mse - qiltr_mse) / eucl_mse) * 100
        improvement_vs_global = ((global_mse - qiltr_mse) / global_mse) * 100
        
        print(f"  QILTR MSE:       {qiltr_mse:.6f}")
        print(f"  Euclidean MSE:   {eucl_mse:.6f} (QILTR is {improvement_vs_eucl:+.2f}% {'better' if improvement_vs_eucl > 0 else 'worse'})")
        print(f"  Global MSE:      {global_mse:.6f} (QILTR is {improvement_vs_global:+.2f}% {'better' if improvement_vs_global > 0 else 'worse'})")
        
        # Highlight where QILTR shines
        if improvement_vs_eucl > 5:
            print(f"  ✓ QILTR SHINES: Significant advantage in {scenario_name.lower()}")
        elif improvement_vs_eucl > 0:
            print(f"  ✓ QILTR ADVANTAGE: Modest improvement in {scenario_name.lower()}")
        else:
            print(f"  ⚠ QILTR CHALLENGED: Performance gap in {scenario_name.lower()}")
    
    print("\n" + "=" * 80)
    print("EXPERIMENT 1-3 COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print("\nKey Outputs:")
    print(f"  - Comparison figure: {OUTPUT_DIRS['figures']}challenging_scenarios_comparison.png")
    print(f"  - Summary table: {OUTPUT_DIRS['tables']}challenging_scenarios_summary.csv")
    print(f"  - Publication summary: {OUTPUT_DIRS['results']}publication_summary.md")
    
    return all_scenarios_results, summary_df

if __name__ == '__main__':
    results, summary = run_convergence_experiment()
    print("\n" + "=" * 80)
    print("Full Summary Table:")
    print("=" * 80)
    print(summary.to_string(index=False))
