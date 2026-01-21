#!/usr/bin/env python3
"""
Compute TRUE correlation baseline for causal discovery comparison.

This computes what a correlation-based method would achieve on the dataset,
giving us a principled baseline to compare against.

The correlation baseline:
1. Loads the observed data X
2. Computes pairwise Pearson/Spearman correlations (handles missing data)
3. Takes top-K correlations by absolute value
4. Evaluates TopK-F1, Skeleton-F1, SHD against A_true

This is the CORRECT baseline - not the model's early-epoch predictions!

Usage:
    python scripts/compute_correlation_baseline.py \
        --data_dir data/interim/uci_air_c/compound_mnar_bias \
        --k 13
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
from scipy import stats
from typing import Tuple, Dict


def compute_correlation_matrix(X: np.ndarray, method: str = "pearson") -> np.ndarray:
    """
    Compute correlation matrix, handling missing values (NaN).
    
    Args:
        X: Data matrix [N, T, d] or [N*T, d]
        method: 'pearson' or 'spearman'
    
    Returns:
        Correlation matrix [d, d]
    """
    # Flatten if 3D
    if X.ndim == 3:
        N, T, d = X.shape
        X = X.reshape(N * T, d)
    
    n_samples, d = X.shape
    
    corr_matrix = np.zeros((d, d))
    
    for i in range(d):
        for j in range(d):
            if i == j:
                corr_matrix[i, j] = 0  # No self-loops
                continue
            
            # Get valid pairs (both non-NaN)
            valid_mask = ~(np.isnan(X[:, i]) | np.isnan(X[:, j]))
            xi = X[valid_mask, i]
            xj = X[valid_mask, j]
            
            if len(xi) < 10:  # Too few valid samples
                corr_matrix[i, j] = 0
                continue
            
            if method == "pearson":
                corr, _ = stats.pearsonr(xi, xj)
            else:  # spearman
                corr, _ = stats.spearmanr(xi, xj)
            
            corr_matrix[i, j] = abs(corr) if not np.isnan(corr) else 0
    
    return corr_matrix


def topk_edges_from_matrix(A: np.ndarray, k: int) -> set:
    """Get top-K edges from a matrix by absolute value."""
    d = A.shape[0]
    A_clean = A.copy()
    np.fill_diagonal(A_clean, 0)
    
    # Get top-k indices
    flat = np.abs(A_clean).flatten()
    top_indices = np.argsort(flat)[::-1][:k]
    
    edges = set()
    for idx in top_indices:
        i, j = idx // d, idx % d
        if flat[idx] > 0:  # Only non-zero
            edges.add((i, j))
    
    return edges


def evaluate_against_truth(pred_edges: set, A_true: np.ndarray, k: int) -> Dict:
    """Evaluate predicted edges against ground truth."""
    # Get true edges
    true_edges = set(zip(*np.where(A_true > 0)))
    n_true = len(true_edges)
    
    # Skeleton (undirected) comparison
    pred_skeleton = set()
    for (i, j) in pred_edges:
        pred_skeleton.add((min(i,j), max(i,j)))
    
    true_skeleton = set()
    for (i, j) in true_edges:
        true_skeleton.add((min(i,j), max(i,j)))
    
    # Directed metrics
    tp = len(true_edges & pred_edges)
    fp = len(pred_edges - true_edges)
    fn = len(true_edges - pred_edges)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Skeleton metrics
    skel_tp = len(true_skeleton & pred_skeleton)
    skel_precision = skel_tp / len(pred_skeleton) if pred_skeleton else 0
    skel_recall = skel_tp / len(true_skeleton) if true_skeleton else 0
    skel_f1 = 2 * skel_precision * skel_recall / (skel_precision + skel_recall) if (skel_precision + skel_recall) > 0 else 0
    
    # Reversed direction check
    reversed_edges = set()
    for (i, j) in pred_edges:
        if (j, i) in true_edges:
            reversed_edges.add((i, j))
    
    return {
        "k": k,
        "true_edges": n_true,
        "pred_edges": len(pred_edges),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "topk_precision": precision,
        "topk_recall": recall,
        "topk_f1": f1,
        "skeleton_f1": skel_f1,
        "reversed_direction": len(reversed_edges),
        "true_edge_list": sorted([f"{i}->{j}" for i, j in true_edges]),
        "pred_edge_list": sorted([f"{i}->{j}" for i, j in pred_edges]),
        "correct_edges": sorted([f"{i}->{j}" for i, j in (true_edges & pred_edges)]),
        "reversed_edges": sorted([f"{i}->{j}" for i, j in reversed_edges]),
    }


def main():
    parser = argparse.ArgumentParser(description="Compute correlation baseline")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--k", type=int, default=None, help="Top-K edges (default: true edge count)")
    parser.add_argument("--method", choices=["pearson", "spearman"], default="pearson")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    print("=" * 70)
    print("CORRELATION BASELINE COMPUTATION")
    print("=" * 70)
    print(f"Dataset: {data_dir.name}")
    print(f"Method:  {args.method}")
    
    # Load data
    X = np.load(data_dir / "X.npy")
    A_true = np.load(data_dir / "A_true.npy")
    
    d = A_true.shape[0]
    n_true_edges = int((A_true > 0).sum())
    k = args.k if args.k else n_true_edges
    
    print(f"d = {d}")
    print(f"True edges = {n_true_edges}")
    print(f"K = {k}")
    
    # Check for missingness
    miss_rate = np.isnan(X).mean() * 100
    print(f"Missing rate = {miss_rate:.1f}%")
    
    # Compute correlation matrix
    print(f"\nComputing {args.method} correlation matrix...")
    corr_matrix = compute_correlation_matrix(X, args.method)
    
    print(f"Correlation matrix stats:")
    print(f"  Max:  {corr_matrix.max():.4f}")
    print(f"  Mean: {corr_matrix.mean():.4f}")
    print(f"  Min (non-zero): {corr_matrix[corr_matrix > 0].min():.4f}" if (corr_matrix > 0).any() else "  No positive correlations")
    
    # Get top-K edges
    pred_edges = topk_edges_from_matrix(corr_matrix, k)
    
    # Evaluate
    print(f"\n" + "=" * 70)
    print(f"CORRELATION BASELINE RESULTS (K={k})")
    print("=" * 70)
    
    results = evaluate_against_truth(pred_edges, A_true, k)
    
    print(f"\n*** CORRELATION BASELINE TopK-F1 = {results['topk_f1']:.4f} ***")
    print(f"    Skeleton-F1 = {results['skeleton_f1']:.4f}")
    print(f"    TP = {results['tp']}/{n_true_edges}")
    print(f"    Reversed direction = {results['reversed_direction']}")
    
    print(f"\nTrue edges: {results['true_edge_list']}")
    print(f"\nTop-{k} correlation edges: {results['pred_edge_list']}")
    print(f"\nCorrectly identified: {results['correct_edges']}")
    print(f"Reversed (wrong direction): {results['reversed_edges']}")
    
    # Save correlation matrix
    output_file = data_dir / "correlation_baseline.npy"
    np.save(output_file, corr_matrix)
    print(f"\nCorrelation matrix saved to: {output_file}")
    
    # Print summary for easy copy
    print(f"\n" + "=" * 70)
    print("SUMMARY (for comparison with model)")
    print("=" * 70)
    print(f"Dataset:           {data_dir.name}")
    print(f"Correlation TopK-F1: {results['topk_f1']:.4f}")
    print(f"Correlation Skel-F1: {results['skeleton_f1']:.4f}")
    print(f"True Positives:      {results['tp']}/{n_true_edges}")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()
