"""
MAIN RUNNER SCRIPT
Execute all QILTR experiments with a single command
"""

import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("")
    print("=" * 80)
    print(" " * 20 + "QILTR EXPERIMENTAL VALIDATION")
    print(" " * 15 + "Complete Empirical Evaluation Framework")
    print("=" * 80)
    print("")

    # Create output directories
    from config import OUTPUT_DIRS
    for dir_path in OUTPUT_DIRS.values():
        os.makedirs(dir_path, exist_ok=True)

    print("Select experiment to run:")
    print("  1. Experiment 1-3: Convergence Analysis")
    print("  2. Experiment 4-5: Computational Complexity")
    print("  3. Run ALL basic experiments (1-5)")
    print("  4. Experiment 6: Statistical Validation (30 trials)")
    print("  5. Experiment 7: Ablation Studies")
    print("  6. Experiment 8: Real-World Dataset Validation")
    print("  7. Run COMPLETE validation suite (1-8, recommended for thesis)")
    print("  0. Exit")
    print("")

    choice = input("Enter your choice (0-7): ")

    if choice == '1':
        print("\nRunning Convergence Analysis...")
        from experiments.exp1_convergence import run_convergence_experiment
        results, summary = run_convergence_experiment()
        print("\nDone! Check results/ directory for outputs.")

    elif choice == '2':
        print("\nRunning Computational Complexity Analysis...")
        from experiments.exp2_complexity import measure_complexity
        results_df = measure_complexity()
        print("\nDone! Check results/ directory for outputs.")

    elif choice == '3':
        print("\nRunning ALL basic experiments (1-5)...")
        print("")

        # Experiment 1-3
        print("STEP 1/2: Convergence Analysis")
        print("-" * 80)
        from experiments.exp1_convergence import run_convergence_experiment
        results1, summary1 = run_convergence_experiment()
        print("")

        # Experiment 4-5
        print("STEP 2/2: Computational Complexity")
        print("-" * 80)
        from experiments.exp2_complexity import measure_complexity
        results2 = measure_complexity()
        print("")

        print("=" * 80)
        print("BASIC EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("")
        print("Results saved to:")
        print("  - Figures: results/figures/")
        print("  - Tables: results/tables/")
        print("  - Raw data: results/")
        print("")

    elif choice == '4':
        print("\nRunning Statistical Validation (this may take 30-45 minutes)...")
        print("")
        from experiments.exp6_statistical_validation import main as run_stat_validation
        run_stat_validation()
        print("\nDone! Check results/statistical_validation_report.md")

    elif choice == '5':
        print("\nRunning Ablation Studies (this may take 20-30 minutes)...")
        print("")
        from experiments.exp7_ablation_studies import main as run_ablations
        run_ablations()
        print("\nDone! Check results/ablation_study_report.md")

    elif choice == '6':
        print("\nRunning Real-World Dataset Validation...")
        print("")
        from experiments.exp8_realworld_validation import main as run_realworld
        run_realworld()
        print("\nDone! Check results/realworld_validation_report.md")

    elif choice == '7':
        print("\n" + "="*80)
        print("RUNNING COMPLETE VALIDATION SUITE FOR THESIS SUBMISSION")
        print("This will take approximately 1-2 hours")
        print("="*80)
        print("")
        
        confirm = input("Continue? (y/n): ")
        if confirm.lower() != 'y':
            print("Cancelled.")
            return
        
        # Run all experiments in sequence
        print("\n[1/6] Convergence Analysis...")
        from experiments.exp1_convergence import run_convergence_experiment
        run_convergence_experiment()
        
        print("\n[2/6] Computational Complexity...")
        from experiments.exp2_complexity import measure_complexity
        measure_complexity()
        
        print("\n[3/6] Statistical Validation (30 trials)...")
        from experiments.exp6_statistical_validation import main as run_stat_validation
        run_stat_validation()
        
        print("\n[4/6] Ablation Studies...")
        from experiments.exp7_ablation_studies import main as run_ablations
        run_ablations()
        
        print("\n[5/6] Real-World Validation...")
        from experiments.exp8_realworld_validation import main as run_realworld
        run_realworld()
        
        print("\n[6/6] Generating final summary...")
        # Generate comprehensive thesis summary
        from pathlib import Path
        summary_path = Path("results/THESIS_SUBMISSION_SUMMARY.md")
        with open(summary_path, 'w') as f:
            f.write("# QILTR Thesis Submission - Complete Validation Summary\n\n")
            f.write("## Experiments Completed\n\n")
            f.write("✅ **Experiment 1-3**: Convergence Analysis (5 scenarios)\n")
            f.write("✅ **Experiment 4-5**: Computational Complexity Analysis\n")
            f.write("✅ **Experiment 6**: Statistical Validation (30 trials per scenario)\n")
            f.write("✅ **Experiment 7**: Ablation Studies (5 component analyses)\n")
            f.write("✅ **Experiment 8**: Real-World Dataset Validation\n\n")
            f.write("## Key Results Files\n\n")
            f.write("- `publication_summary.md` - Main results overview\n")
            f.write("- `statistical_validation_report.md` - Significance tests\n")
            f.write("- `ablation_study_report.md` - Design choice analysis\n")
            f.write("- `realworld_validation_report.md` - Practical applicability\n")
            f.write("- `figures/` - All publication-quality figures\n")
            f.write("- `tables/` - Numerical results in CSV format\n\n")
            f.write("## Reproducibility\n\n")
            f.write("All experiments use fixed random seeds (config.RANDOM_SEED=42).\n")
            f.write("See individual experiment files for detailed parameters.\n\n")
            f.write("## Next Steps for Thesis\n\n")
            f.write("1. Review all markdown reports in results/\n")
            f.write("2. Select key figures for thesis (check results/figures/)\n")
            f.write("3. Copy statistical results to thesis text\n")
            f.write("4. Document any limitations or negative results honestly\n")
            f.write("5. Prepare supplementary materials with code\n\n")
        
        print(f"Saved: {summary_path}")
        
        print("\n" + "="*80)
        print("COMPLETE VALIDATION SUITE FINISHED!")
        print("="*80)
        print("\nAll results saved. Key files:")
        print(f"  - {summary_path}")
        print("  - results/publication_summary.md")
        print("  - results/statistical_validation_report.md")
        print("  - results/ablation_study_report.md")
        print("  - results/realworld_validation_report.md")
        print("")
        print("Your QILTR framework is now ready for Q1 journal submission!")
        print("")

    elif choice == '0':
        print("Exiting...")
        return
    else:
        print("Invalid choice. Exiting...")
        return

if __name__ == '__main__':
    main()
