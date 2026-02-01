#!/usr/bin/env python3
"""
Analyze corruption sweep results and generate statistics.
Computes mean ± std, paired tests, and degradation curves.
"""
import numpy as np
import json
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt

def load_results(corruption_rate, method, n_seeds=5):
    """Load results for a method at a given corruption rate."""
    results_dir = Path("artifacts/corruption_sweep") / f"corruption_{int(corruption_rate*100):02d}"
    
    metrics = {'f1': [], 'auprc': [], 'shd': [], 'precision': [], 'recall': []}
    
    for seed in range(1, n_seeds + 1):
        result_file = results_dir / f"{method}_seed{seed}.json"
        if result_file.exists():
            with open(result_file) as f:
                data = json.load(f)
                for key in metrics:
                    if key in data:
                        metrics[key].append(data[key])
    
    # Convert to arrays and compute stats
    stats_dict = {}
    for key, values in metrics.items():
        if values:
            stats_dict[key] = {
                'mean': np.mean(values),
                'std': np.std(values, ddof=1),
                'values': values
            }
    
    return stats_dict

def paired_test(values_a, values_b, method='wilcoxon'):
    """Run paired statistical test."""
    if len(values_a) != len(values_b):
        return None
    
    if method == 'wilcoxon':
        statistic, pvalue = stats.wilcoxon(values_a, values_b, alternative='greater')
    else: # t-test
        statistic, pvalue = stats.ttest_rel(values_a, values_b, alternative='greater')
    
    return {'statistic': statistic, 'pvalue': pvalue}

def main():
    corruption_rates = [0.0, 0.10, 0.20, 0.30, 0.40]
    
    print("=" * 80)
    print("CORRUPTION SWEEP ANALYSIS")
    print("=" * 80)
    print()
    
    # Collect results
    rcgnn_results = {}
    notears_results = {}
    
    for rate in corruption_rates:
        rcgnn_results[rate] = load_results(rate, 'rcgnn')
        notears_results[rate] = load_results(rate, 'notears')
    
    # Print summary table
    print("RESULTS SUMMARY (mean ± std)")
    print("-" * 80)
    print(f"{'Rate':<8} {'Method':<12} {'F1':<18} {'AUPRC':<18} {'SHD':<12}")
    print("-" * 80)
    
    for rate in corruption_rates:
        # RC-GNN
        rcgnn = rcgnn_results[rate]
        if 'f1' in rcgnn:
            print(f"{rate:<8.0%} {'RC-GNN':<12} "
                  f"{rcgnn['f1']['mean']:.3f}±{rcgnn['f1']['std']:.3f} "
                  f"{rcgnn['auprc']['mean']:.3f}±{rcgnn['auprc']['std']:.3f} "
                  f"{rcgnn['shd']['mean']:.1f}±{rcgnn['shd']['std']:.1f}")
        
        # NOTEARS
        notears = notears_results[rate]
        if 'f1' in notears:
            print(f"{'':<8} {'NOTEARS':<12} "
                  f"{notears['f1']['mean']:.3f}±{notears['f1']['std']:.3f} "
                  f"{notears['auprc']['mean']:.3f}±{notears['auprc']['std']:.3f} "
                  f"{notears['shd']['mean']:.1f}±{notears['shd']['std']:.1f}")
        print()
    
    print("-" * 80)
    print()
    
    # Statistical tests
    print("PAIRED STATISTICAL TESTS (RC-GNN vs NOTEARS)")
    print("-" * 80)
    print(f"{'Rate':<8} {'Metric':<10} {'RC-GNN > NOTEARS':<20} {'p-value':<12}")
    print("-" * 80)
    
    for rate in corruption_rates:
        rcgnn = rcgnn_results[rate]
        notears = notears_results[rate]
        
        for metric in ['f1', 'auprc']:
            if metric in rcgnn and metric in notears:
                test = paired_test(rcgnn[metric]['values'], notears[metric]['values'])
                if test:
                    significance = "***" if test['pvalue'] < 0.001 else "**" if test['pvalue'] < 0.01 else "*" if test['pvalue'] < 0.05 else "ns"
                    print(f"{rate:<8.0%} {metric.upper():<10} {significance:<20} {test['pvalue']:.4f}")
    
    print("-" * 80)
    print()
    
    # Save aggregated results
    output_file = Path("artifacts/corruption_sweep/summary.json")
    summary = {
        'rcgnn': {str(rate): rcgnn_results[rate] for rate in corruption_rates},
        'notears': {str(rate): notears_results[rate] for rate in corruption_rates}
    }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    
    print(f"[OK] Summary saved to: {output_file}")
    print()
    
    # Generate degradation curve plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, metric in enumerate(['f1', 'auprc']):
        ax = axes[idx]
        
        # RC-GNN
        rcgnn_means = [rcgnn_results[r][metric]['mean'] for r in corruption_rates if metric in rcgnn_results[r]]
        rcgnn_stds = [rcgnn_results[r][metric]['std'] for r in corruption_rates if metric in rcgnn_results[r]]
        
        # NOTEARS
        notears_means = [notears_results[r][metric]['mean'] for r in corruption_rates if metric in notears_results[r]]
        notears_stds = [notears_results[r][metric]['std'] for r in corruption_rates if metric in notears_results[r]]
        
        rates_pct = [r * 100 for r in corruption_rates]
        
        ax.errorbar(rates_pct, rcgnn_means, yerr=rcgnn_stds, marker='o', label='RC-GNN', capsize=5)
        ax.errorbar(rates_pct, notears_means, yerr=notears_stds, marker='s', label='NOTEARS', capsize=5)
        
        ax.set_xlabel('Missing Data Rate (%)')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} vs Corruption Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = Path("artifacts/corruption_sweep/degradation_curve.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"[OK] Plot saved to: {plot_file}")
    print()
    
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
