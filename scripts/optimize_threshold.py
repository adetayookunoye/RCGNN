#!/usr/bin/env python3
"""
Threshold optimization and comparison script.

Finds optimal binary threshold for adjacency matrix by:
1. Testing multiple threshold values
2. Computing precision, recall, F1 scores
3. Generating precision-recall curves
4. Recommending optimal threshold

Usage:
  python scripts/optimize_threshold.py --adjacency artifacts/adjacency/A_mean.npy --data-root data/interim/uci_air
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score, roc_auc_score

import path_helper # noqa: F401


def compute_metrics_at_threshold(A_pred, A_true, threshold):
    """Compute metrics at a specific threshold."""
    A_pred_bin = (A_pred > threshold).astype(int)
    A_true_bin = (A_true > 0).astype(int)
    
    A_pred_flat = A_pred_bin.flatten()
    A_true_flat = A_true_bin.flatten()
    
    tp = np.sum((A_pred_flat == 1) & (A_true_flat == 1))
    fp = np.sum((A_pred_flat == 1) & (A_true_flat == 0))
    fn = np.sum((A_pred_flat == 0) & (A_true_flat == 1))
    tn = np.sum((A_pred_flat == 0) & (A_true_flat == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Structural Hamming Distance
    shd = int(np.sum(A_pred_bin != A_true_bin))
    
    # False positive rate, true negative rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tpr = recall # same as recall
    
    # Sparsity (what % of edges are non-zero in prediction)
    sparsity = np.sum(A_pred_bin) / A_pred_bin.size
    
    return {
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'shd': shd,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn),
        'fpr': fpr,
        'tpr': tpr,
        'sparsity': sparsity,
    }


def find_optimal_threshold(A_pred, A_true, thresholds=None, verbose=True):
    """Find optimal threshold by grid search over F1 scores."""
    if thresholds is None:
        # Test thresholds from 0 to max(A_pred) with linear spacing
        # This ensures we test intermediate values, not just extreme ones
        max_val = A_pred.max()
        if max_val > 0:
            # Use linear spacing from 0 to max with 100 points
            thresholds = np.linspace(0, max_val, 100)
        else:
            # Fallback if all values are 0
            thresholds = np.linspace(0, 1, 100)
    
    results = []
    for thr in thresholds:
        metrics = compute_metrics_at_threshold(A_pred, A_true, thr)
        results.append(metrics)
    
    # Find threshold with maximum F1
    f1_scores = [r['f1'] for r in results]
    best_idx = np.argmax(f1_scores)
    best_threshold = results[best_idx]['threshold']
    best_f1 = results[best_idx]['f1']
    
    if verbose:
        print(f"\n[DONE] Optimal threshold found: {best_threshold:.6f}")
        print(f" F1 Score: {best_f1:.4f}")
        print(f" Precision: {results[best_idx]['precision']:.4f}")
        print(f" Recall: {results[best_idx]['recall']:.4f}")
        print(f" SHD: {results[best_idx]['shd']}")
        print(f" Sparsity: {results[best_idx]['sparsity']:.2%}")
    
    return best_threshold, results


def plot_threshold_analysis(results, output_path='artifacts/threshold_analysis.png'):
    """Plot precision-recall and F1 curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    thresholds = [r['threshold'] for r in results]
    precisions = [r['precision'] for r in results]
    recalls = [r['recall'] for r in results]
    f1s = [r['f1'] for r in results]
    shds = [r['shd'] for r in results]
    sparsities = [r['sparsity'] for r in results]
    
    # Precision-Recall curve
    ax = axes[0, 0]
    ax.plot(recalls, precisions, 'b-', linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # F1 vs Threshold
    ax = axes[0, 1]
    best_idx = np.argmax(f1s)
    ax.plot(thresholds, f1s, 'g-', linewidth=2, marker='o', markersize=4)
    ax.axvline(thresholds[best_idx], color='r', linestyle='--', linewidth=2, 
               label=f'Optimal: {thresholds[best_idx]:.6f}')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score vs Threshold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # SHD vs Threshold
    ax = axes[1, 0]
    ax.plot(thresholds, shds, 'r-', linewidth=2, marker='o', markersize=4)
    ax.axvline(thresholds[best_idx], color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Structural Hamming Distance')
    ax.set_title('SHD vs Threshold (Lower is Better)')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # Sparsity vs Threshold
    ax = axes[1, 1]
    ax.plot(thresholds, sparsities, 'purple', linewidth=2, marker='o', markersize=4)
    ax.axvline(thresholds[best_idx], color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Sparsity (% non-zero edges)')
    ax.set_title('Sparsity vs Threshold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[DONE] Saved threshold analysis: {output_path}")
    plt.close()


def plot_threshold_comparison(results, output_path='artifacts/threshold_metrics_table.png'):
    """Create comparison table of top thresholds."""
    # Sort by F1 score
    sorted_results = sorted(results, key=lambda x: x['f1'], reverse=True)[:10]
    
    # Create table
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    for i, r in enumerate(sorted_results, 1):
        table_data.append([
            f"{i}",
            f"{r['threshold']:.6f}",
            f"{r['precision']:.4f}",
            f"{r['recall']:.4f}",
            f"{r['f1']:.4f}",
            f"{r['shd']}",
            f"{r['sparsity']:.1%}",
        ])
    
    columns = ['Rank', 'Threshold', 'Precision', 'Recall', 'F1', 'SHD', 'Sparsity']
    table = ax.table(cellText=table_data, colLabels=columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight best row
    for j in range(len(columns)):
        table[(1, j)].set_facecolor('#90EE90')
    
    plt.title('Top 10 Thresholds by F1 Score', fontsize=14, fontweight='bold', pad=20)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[DONE] Saved threshold comparison table: {output_path}")
    plt.close()


def save_threshold_report(best_threshold, results, output_path='artifacts/threshold_report.txt'):
    """Save detailed threshold optimization report."""
    report = []
    report.append("=" * 80)
    report.append("THRESHOLD OPTIMIZATION REPORT")
    report.append("=" * 80)
    report.append("")
    
    best_result = [r for r in results if r['threshold'] == best_threshold][0]
    
    report.append("## RECOMMENDED THRESHOLD")
    report.append(f"Value: {best_threshold:.6f}")
    report.append(f"F1 Score: {best_result['f1']:.4f}")
    report.append(f"Precision: {best_result['precision']:.4f}")
    report.append(f"Recall: {best_result['recall']:.4f}")
    report.append(f"Structural Hamming Distance: {best_result['shd']}")
    report.append(f"Predicted Sparsity: {best_result['sparsity']:.1%} non-zero edges")
    report.append("")
    
    report.append("## INTERPRETATION")
    report.append(f"At threshold {best_threshold:.6f}:")
    report.append(f" • {best_result['tp']} True Positives (correct edge predictions)")
    report.append(f" • {best_result['fp']} False Positives (incorrect edge predictions)")
    report.append(f" • {best_result['fn']} False Negatives (missed edges)")
    report.append(f" • {best_result['tn']} True Negatives (correct non-edge predictions)")
    report.append("")
    
    report.append("## RECOMMENDATIONS")
    if best_result['sparsity'] > 0.9:
        report.append("[WARN] WARNING: Very high sparsity (>90%)")
        report.append(" Consider lowering threshold to capture more causal relationships")
    elif best_result['sparsity'] < 0.1:
        report.append("[WARN] WARNING: Very low sparsity (<10%)")
        report.append(" Consider raising threshold to reduce false positives")
    else:
        report.append("[DONE] Sparsity is reasonable (10-90%)")
    
    if best_result['precision'] < 0.5:
        report.append("[WARN] Low precision - many false positives")
        report.append(" Consider raising threshold for stricter edge selection")
    
    if best_result['recall'] < 0.5:
        report.append("[WARN] Low recall - missing many true edges")
        report.append(" Consider lowering threshold to capture more relationships")
    
    report.append("")
    report.append("## USAGE IN DOWNSTREAM ANALYSIS")
    report.append(f"Use threshold: {best_threshold:.6f}")
    report.append("")
    report.append("Python:")
    report.append(f" A_binary = (A_pred > {best_threshold:.6f}).astype(int)")
    report.append("")
    report.append("Command line:")
    report.append(f" python scripts/validate_and_visualize.py --threshold {best_threshold:.6f}")
    report.append("")
    
    report.append("## FULL THRESHOLD RESULTS")
    report.append(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'SHD':<8} {'Sparsity':<12}")
    report.append("-" * 80)
    for r in sorted(results, key=lambda x: x['f1'], reverse=True)[:20]:
        report.append(
            f"{r['threshold']:<12.6f} {r['precision']:<12.4f} {r['recall']:<12.4f} "
            f"{r['f1']:<12.4f} {r['shd']:<8} {r['sparsity']:<12.1%}"
        )
    
    report.append("=" * 80)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"[DONE] Saved threshold report: {output_path}")
    
    # Print to stdout
    print("\n" + '\n'.join(report))


def main():
    parser = argparse.ArgumentParser(
        description="Optimize binary threshold for adjacency matrix"
    )
    parser.add_argument('--adjacency', default='artifacts/adjacency/A_mean.npy',
                       help='Path to learned adjacency matrix')
    parser.add_argument('--data-root', required=True,
                       help='Data root directory (must contain A_true.npy)')
    parser.add_argument('--export', default='artifacts',
                       help='Export directory for visualizations')
    parser.add_argument('--n-thresholds', type=int, default=50,
                       help='Number of thresholds to test (default: 50)')
    
    args = parser.parse_args()
    
    # Load adjacency
    if not os.path.exists(args.adjacency):
        print(f"[FAIL] Error: Adjacency not found at {args.adjacency}")
        sys.exit(1)
    
    A_pred = np.load(args.adjacency)
    print(f"[DONE] Loaded predicted adjacency: {A_pred.shape}")
    
    # Load ground truth
    gt_path = os.path.join(args.data_root, 'A_true.npy')
    if not os.path.exists(gt_path):
        print(f"[FAIL] Error: Ground truth not found at {gt_path}")
        print(f" This script requires ground truth for threshold optimization")
        sys.exit(1)
    
    A_true = np.load(gt_path)
    print(f"[DONE] Loaded ground truth adjacency: {A_true.shape}")
    
    # Generate thresholds - use linear spacing for better threshold exploration
    max_val = A_pred.max()
    if max_val > 0:
        thresholds = np.linspace(0, max_val * 1.1, args.n_thresholds)
    else:
        thresholds = np.linspace(0, 1, args.n_thresholds)
    
    # Find optimal threshold
    print("\n" + "=" * 80)
    print("THRESHOLD OPTIMIZATION")
    print("=" * 80)
    
    best_threshold, results = find_optimal_threshold(A_pred, A_true, thresholds)
    
    # Generate visualizations
    print("\n Generating visualizations...")
    plot_threshold_analysis(results, 
                           os.path.join(args.export, 'threshold_analysis.png'))
    plot_threshold_comparison(results, 
                             os.path.join(args.export, 'threshold_comparison_table.png'))
    
    # Save report
    save_threshold_report(best_threshold, results,
                         os.path.join(args.export, 'threshold_report.txt'))
    
    print("\n[DONE] Threshold optimization complete!")
    print(f" Best threshold: {best_threshold:.6f}")
    print(f" Visualizations saved to: {args.export}/")


if __name__ == "__main__":
    main()
