"""
Experiment 6: Statistical Validation & Significance Testing

Rigorous statistical analysis with multiple trials, confidence intervals,
and hypothesis testing for publication standards.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pickle
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.qiltr import QILTR
from src.baselines import EuclideanLTR, GlobalTuckerRegression
from src.synthetic_data import generate_synthetic_tensor_regression
from src.metrics import compute_all_metrics
from config import *

plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


def run_multiple_trials(scenario_name, scenario_config, common_params, n_trials=30):
    """
    Run experiment multiple times with different random seeds for statistical validation.
    
    Args:
        scenario_name: Name of scenario
        scenario_config: Configuration dictionary
        common_params: Common parameters for all methods
        n_trials: Number of independent trials
    
    Returns:
        Dictionary with results from all trials
    """
    print(f"\n{'='*80}")
    print(f"STATISTICAL VALIDATION: {scenario_name}")
    print(f"Running {n_trials} independent trials...")
    print(f"{'='*80}\n")
    
    results = {
        'QILTR': {'mse': [], 'mae': [], 'r2': [], 'time': []},
        'Euclidean-LTR': {'mse': [], 'mae': [], 'r2': [], 'time': []},
        'Global-Tucker': {'mse': [], 'mae': [], 'r2': [], 'time': []}
    }
    
    for trial in range(n_trials):
        seed = RANDOM_SEED + trial  # Different seed for each trial
        print(f"Trial {trial+1}/{n_trials} (seed={seed})...", end=' ')
        
        # Generate data with this trial's seed (using same defaults as exp1)
        data = generate_synthetic_tensor_regression(
            n_samples=1000,
            input_dim=20,
            tensor_shape=(5, 5, 5),
            tucker_ranks=(3, 3, 3),
            n_regions=scenario_config.get('n_regions', 3),
            noise_level=0.5,
            random_state=seed,
            ill_conditioned=scenario_config.get('ill_conditioned', False),
            condition_number=scenario_config.get('condition_number', 100.0),
            high_nonstationarity=scenario_config.get('high_nonstationarity', False),
            region_diversity_scale=scenario_config.get('region_diversity_scale', 1.0)
        )
        
        # Unpack data
        X, Y, region_labels, true_tensors = data
        
        # Train/test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=seed
        )
        
        # Add outliers if specified
        if scenario_config.get('outlier_fraction', 0) > 0:
            from src.synthetic_data import add_outliers
            X_train, Y_train, _ = add_outliers(
                X_train,
                Y_train,
                outlier_fraction=scenario_config['outlier_fraction'],
                outlier_magnitude=scenario_config.get('outlier_magnitude', 5.0),
                random_state=seed
            )
        
        # Train and evaluate each method
        methods = {
            'QILTR': QILTR(
                n_centroids=common_params['n_centroids'],
                bandwidth=common_params['bandwidth'],
                ranks=common_params['ranks'],
                max_als_iter=common_params['max_als_iter'],
                als_tol=common_params['als_tol'],
                random_state=seed
            ),
            'Euclidean-LTR': EuclideanLTR(
                n_centroids=common_params['n_centroids'],
                bandwidth=common_params['bandwidth'],
                ranks=common_params['ranks'],
                max_als_iter=common_params['max_als_iter'],
                als_tol=common_params['als_tol'],
                random_state=seed
            ),
            'Global-Tucker': GlobalTuckerRegression(
                ranks=common_params['ranks'],
                max_als_iter=common_params['max_als_iter'],
                als_tol=common_params['als_tol'],
                random_state=seed
            )
        }
        
        for method_name, method in methods.items():
            # Train
            import time
            start = time.time()
            method.fit(X_train, Y_train)
            train_time = time.time() - start
            
            # Predict
            Y_pred = method.predict(X_test)
            
            # Compute metrics
            metrics = compute_all_metrics(Y_test, Y_pred)
            
            # Store results
            results[method_name]['mse'].append(metrics['mse'])
            results[method_name]['mae'].append(metrics['mae'])
            if 'r2' in metrics:
                results[method_name]['r2'].append(metrics['r2'])
            results[method_name]['time'].append(train_time)
        
        print(f"✓ (QILTR MSE: {results['QILTR']['mse'][-1]:.4f})")
    
    # Convert lists to numpy arrays
    for method_name in results:
        for metric in results[method_name]:
            results[method_name][metric] = np.array(results[method_name][metric])
    
    return results


def compute_statistics(results, metric='mse'):
    """
    Compute comprehensive statistics for a given metric.
    
    Returns:
        DataFrame with mean, std, median, CI, etc.
    """
    stats_dict = {}
    
    for method_name in results:
        data = results[method_name][metric]
        n = len(data)
        
        # Basic statistics
        mean = np.mean(data)
        std = np.std(data, ddof=1)  # Sample standard deviation
        median = np.median(data)
        
        # 95% Confidence Interval (t-distribution)
        ci_level = 0.95
        t_critical = stats.t.ppf((1 + ci_level) / 2, n - 1)
        margin_error = t_critical * (std / np.sqrt(n))
        ci_lower = mean - margin_error
        ci_upper = mean + margin_error
        
        # Interquartile range
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        
        stats_dict[method_name] = {
            'mean': mean,
            'std': std,
            'median': median,
            'min': np.min(data),
            'max': np.max(data),
            'q1': q1,
            'q3': q3,
            'iqr': iqr,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_margin': margin_error,
            'n_trials': n
        }
    
    return pd.DataFrame(stats_dict).T


def perform_significance_tests(results, metric='mse'):
    """
    Perform paired hypothesis tests between methods.
    
    Returns:
        DataFrame with p-values and effect sizes
    """
    methods = list(results.keys())
    n_methods = len(methods)
    
    # Initialize result matrices
    p_values_t = np.zeros((n_methods, n_methods))
    p_values_wilcoxon = np.zeros((n_methods, n_methods))
    cohens_d = np.zeros((n_methods, n_methods))
    
    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if i == j:
                p_values_t[i, j] = 1.0
                p_values_wilcoxon[i, j] = 1.0
                cohens_d[i, j] = 0.0
                continue
            
            data1 = results[method1][metric]
            data2 = results[method2][metric]
            
            # Paired t-test (parametric)
            t_stat, p_t = stats.ttest_rel(data1, data2)
            p_values_t[i, j] = p_t
            
            # Wilcoxon signed-rank test (non-parametric)
            try:
                w_stat, p_w = stats.wilcoxon(data1, data2, alternative='two-sided')
                p_values_wilcoxon[i, j] = p_w
            except:
                p_values_wilcoxon[i, j] = np.nan
            
            # Cohen's d effect size
            pooled_std = np.sqrt((np.var(data1, ddof=1) + np.var(data2, ddof=1)) / 2)
            if pooled_std > 0:
                cohens_d[i, j] = (np.mean(data1) - np.mean(data2)) / pooled_std
            else:
                cohens_d[i, j] = 0.0
    
    # Create DataFrames
    df_t = pd.DataFrame(p_values_t, index=methods, columns=methods)
    df_wilcoxon = pd.DataFrame(p_values_wilcoxon, index=methods, columns=methods)
    df_cohens = pd.DataFrame(cohens_d, index=methods, columns=methods)
    
    return {
        'paired_t_test': df_t,
        'wilcoxon_test': df_wilcoxon,
        'cohens_d': df_cohens
    }


def interpret_significance(p_value):
    """Convert p-value to significance marker for plots."""
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return 'ns'


def visualize_statistical_results(all_results, output_dir):
    """
    Create publication-quality visualizations with error bars and significance.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    scenarios = list(all_results.keys())
    n_scenarios = len(scenarios)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, scenario in enumerate(scenarios):
        ax = axes[idx]
        
        results = all_results[scenario]['trial_results']
        stats_df = all_results[scenario]['statistics']['mse']
        sig_tests = all_results[scenario]['significance_tests']
        
        methods = list(results.keys())
        x_pos = np.arange(len(methods))
        
        # Extract means and confidence intervals
        means = [stats_df.loc[m, 'mean'] for m in methods]
        ci_margins = [stats_df.loc[m, 'ci_margin'] for m in methods]
        
        # Bar plot with error bars (95% CI)
        bars = ax.bar(x_pos, means, yerr=ci_margins, 
                     capsize=5, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Color bars
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Add significance markers
        # Compare QILTR vs Euclidean
        p_val = sig_tests['paired_t_test'].loc['QILTR', 'Euclidean-LTR']
        sig_marker = interpret_significance(p_val)
        
        if sig_marker != 'ns':
            # Add significance line
            y_max = max(means) + max(ci_margins)
            y_sig = y_max * 1.1
            ax.plot([0, 1], [y_sig, y_sig], 'k-', linewidth=1.5)
            ax.text(0.5, y_sig * 1.02, sig_marker, ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods, rotation=15, ha='right')
        ax.set_ylabel('MSE', fontsize=11, fontweight='bold')
        ax.set_title(f'{scenario}\n(n=30 trials, 95% CI)', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add sample statistics as text
        qiltr_mean = stats_df.loc['QILTR', 'mean']
        eucl_mean = stats_df.loc['Euclidean-LTR', 'mean']
        improvement = ((eucl_mean - qiltr_mean) / eucl_mean) * 100
        
        ax.text(0.95, 0.95, f'Improvement: {improvement:+.1f}%\np={p_val:.4f}',
               transform=ax.transAxes, ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               fontsize=9)
    
    # Remove empty subplots
    for idx in range(n_scenarios, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'statistical_validation_bars.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'statistical_validation_bars.png'}")
    plt.close()
    
    # Create box plots for distribution visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, scenario in enumerate(scenarios):
        ax = axes[idx]
        
        results = all_results[scenario]['trial_results']
        
        # Prepare data for box plot
        data_to_plot = [results[m]['mse'] for m in results.keys()]
        
        bp = ax.boxplot(data_to_plot, labels=list(results.keys()), 
                        patch_artist=True, notch=True, showmeans=True)
        
        # Color boxes
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('MSE', fontsize=11, fontweight='bold')
        ax.set_title(f'{scenario}\n(Box plot with median & mean)', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Rotate labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right')
    
    # Remove empty subplots
    for idx in range(n_scenarios, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'statistical_validation_boxplots.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'statistical_validation_boxplots.png'}")
    plt.close()


def generate_statistical_report(all_results, output_file):
    """
    Generate comprehensive statistical report for publication.
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Statistical Validation Report\n\n")
        f.write("**Rigorous Statistical Analysis for Q1 Journal Submission**\n\n")
        f.write(f"- Number of independent trials per scenario: 30\n")
        f.write(f"- Confidence level: 95%\n")
        f.write(f"- Significance threshold: alpha = 0.05\n")
        f.write(f"- Statistical tests: Paired t-test (parametric), Wilcoxon signed-rank (non-parametric)\n\n")
        f.write("---\n\n")
        
        for scenario in all_results:
            f.write(f"## {scenario}\n\n")
            
            # Descriptive statistics
            f.write("### Descriptive Statistics (MSE)\n\n")
            stats_df = all_results[scenario]['statistics']['mse']
            f.write(stats_df.to_markdown())
            f.write("\n\n")
            
            # Significance tests
            f.write("### Hypothesis Testing\n\n")
            f.write("**Paired t-test p-values:**\n\n")
            t_test_df = all_results[scenario]['significance_tests']['paired_t_test']
            f.write(t_test_df.to_markdown())
            f.write("\n\n")
            
            f.write("**Wilcoxon signed-rank test p-values:**\n\n")
            wilcoxon_df = all_results[scenario]['significance_tests']['wilcoxon_test']
            f.write(wilcoxon_df.to_markdown())
            f.write("\n\n")
            
            f.write("**Cohen's d effect sizes:**\n\n")
            cohens_df = all_results[scenario]['significance_tests']['cohens_d']
            f.write(cohens_df.to_markdown())
            f.write("\n\n")
            
            # Interpretation
            f.write("### Interpretation\n\n")
            
            qiltr_mean = stats_df.loc['QILTR', 'mean']
            qiltr_ci = (stats_df.loc['QILTR', 'ci_lower'], stats_df.loc['QILTR', 'ci_upper'])
            
            eucl_mean = stats_df.loc['Euclidean-LTR', 'mean']
            eucl_ci = (stats_df.loc['Euclidean-LTR', 'ci_lower'], stats_df.loc['Euclidean-LTR', 'ci_upper'])
            
            improvement = ((eucl_mean - qiltr_mean) / eucl_mean) * 100
            
            p_val_t = t_test_df.loc['QILTR', 'Euclidean-LTR']
            p_val_w = wilcoxon_df.loc['QILTR', 'Euclidean-LTR']
            cohens = cohens_df.loc['QILTR', 'Euclidean-LTR']
            
            f.write(f"- **QILTR MSE**: {qiltr_mean:.4f} (95% CI: [{qiltr_ci[0]:.4f}, {qiltr_ci[1]:.4f}])\n")
            f.write(f"- **Euclidean-LTR MSE**: {eucl_mean:.4f} (95% CI: [{eucl_ci[0]:.4f}, {eucl_ci[1]:.4f}])\n")
            f.write(f"- **Improvement**: {improvement:+.2f}%\n")
            f.write(f"- **Paired t-test p-value**: {p_val_t:.6f} {interpret_significance(p_val_t)}\n")
            f.write(f"- **Wilcoxon test p-value**: {p_val_w:.6f} {interpret_significance(p_val_w)}\n")
            f.write(f"- **Cohen's d**: {cohens:.4f} ")
            
            # Interpret effect size
            if abs(cohens) < 0.2:
                f.write("(negligible effect)\n")
            elif abs(cohens) < 0.5:
                f.write("(small effect)\n")
            elif abs(cohens) < 0.8:
                f.write("(medium effect)\n")
            else:
                f.write("(large effect)\n")
            
            # Statistical conclusion
            if p_val_t < 0.05 and improvement > 0:
                f.write(f"\n**Conclusion**: QILTR shows **statistically significant** improvement over Euclidean-LTR (p < 0.05).\n")
            elif p_val_t < 0.05 and improvement < 0:
                f.write(f"\n**Conclusion**: Euclidean-LTR shows **statistically significant** improvement over QILTR (p < 0.05).\n")
            else:
                f.write(f"\n**Conclusion**: No statistically significant difference between methods (p ≥ 0.05).\n")
            
            f.write("\n---\n\n")
        
        # Summary
        f.write("## Overall Summary\n\n")
        f.write("### Key Findings:\n\n")
        
        significant_improvements = []
        for scenario in all_results:
            stats_df = all_results[scenario]['statistics']['mse']
            t_test_df = all_results[scenario]['significance_tests']['paired_t_test']
            
            qiltr_mean = stats_df.loc['QILTR', 'mean']
            eucl_mean = stats_df.loc['Euclidean-LTR', 'mean']
            improvement = ((eucl_mean - qiltr_mean) / eucl_mean) * 100
            p_val = t_test_df.loc['QILTR', 'Euclidean-LTR']
            
            if p_val < 0.05 and improvement > 0:
                significant_improvements.append(f"- **{scenario}**: {improvement:+.2f}% (p={p_val:.4f})")
        
        if significant_improvements:
            f.write("**Scenarios with statistically significant QILTR improvements:**\n\n")
            for item in significant_improvements:
                f.write(item + "\n")
        else:
            f.write("No scenarios show statistically significant improvements.\n")
        
        f.write("\n### Significance Legend:\n\n")
        f.write("- `***`: p < 0.001 (highly significant)\n")
        f.write("- `**`: p < 0.01 (very significant)\n")
        f.write("- `*`: p < 0.05 (significant)\n")
        f.write("- `ns`: p ≥ 0.05 (not significant)\n\n")
    
    print(f"Saved: {output_file}")


def main():
    """Run statistical validation experiments."""
    print("="*80)
    print("EXPERIMENT 6: STATISTICAL VALIDATION")
    print("Running rigorous statistical analysis with multiple trials")
    print("="*80)
    
    # Define common parameters
    common_params = {
        'n_centroids': 10,
        'bandwidth': 1.0,
        'ranks': (3, 3, 3),
        'max_als_iter': 100,
        'als_tol': 1e-6
    }
    
    # Select key scenarios for statistical validation
    scenarios_to_test = {
        'Standard': CHALLENGING_SCENARIOS['standard'],
        'High Nonstationarity': CHALLENGING_SCENARIOS['high_nonstationarity'],
        'Ill-Conditioned': CHALLENGING_SCENARIOS['ill_conditioned'],
        'Outliers': CHALLENGING_SCENARIOS['outliers'],
        'Combined': CHALLENGING_SCENARIOS['combined']
    }
    
    all_results = {}
    
    # Run multiple trials for each scenario
    for scenario_name, scenario_config in scenarios_to_test.items():
        if not scenario_config['enabled']:
            continue
        
        # Run trials
        trial_results = run_multiple_trials(
            scenario_name, 
            scenario_config, 
            common_params, 
            n_trials=30
        )
        
        # Compute statistics
        stats_mse = compute_statistics(trial_results, metric='mse')
        stats_mae = compute_statistics(trial_results, metric='mae')
        
        # Perform significance tests
        sig_tests = perform_significance_tests(trial_results, metric='mse')
        
        # Store all results
        all_results[scenario_name] = {
            'trial_results': trial_results,
            'statistics': {
                'mse': stats_mse,
                'mae': stats_mae
            },
            'significance_tests': sig_tests
        }
        
        # Print summary
        print(f"\n{'-'*80}")
        print(f"SUMMARY: {scenario_name}")
        print(f"{'-'*80}")
        print("\nMSE Statistics (mean ± std):")
        for method in trial_results:
            mean = stats_mse.loc[method, 'mean']
            std = stats_mse.loc[method, 'std']
            ci_lower = stats_mse.loc[method, 'ci_lower']
            ci_upper = stats_mse.loc[method, 'ci_upper']
            print(f"  {method:20s}: {mean:.4f} ± {std:.4f}  [95% CI: {ci_lower:.4f}, {ci_upper:.4f}]")
        
        print("\nPaired t-test (QILTR vs Euclidean-LTR):")
        p_val = sig_tests['paired_t_test'].loc['QILTR', 'Euclidean-LTR']
        print(f"  p-value: {p_val:.6f} {interpret_significance(p_val)}")
        print(f"{'-'*80}\n")
    
    # Save results
    RESULTS_DIR = "results"  # Set your desired results directory path here or import from config
    results_dir = Path(RESULTS_DIR)
    
    # Visualizations
    visualize_statistical_results(all_results, results_dir / 'figures')
    
    # Statistical report
    generate_statistical_report(all_results, results_dir / 'statistical_validation_report.md')
    
    # Save raw data
    with open(results_dir / 'statistical_validation_data.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    print(f"Saved: {results_dir / 'statistical_validation_data.pkl'}")
    
    print("\n" + "="*80)
    print("STATISTICAL VALIDATION COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"\nResults saved to: {results_dir}")


if __name__ == "__main__":
    main()
