#!/usr/bin/env python3
"""
Proper evaluation: RC-GNN vs baselines on UCI Air.
- Loads best checkpoint
- Evaluates at optimal threshold
- Runs NOTEARS baseline
- Reports fair comparison
"""

import os
import sys
import json
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, average_precision_score

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.rcgnn import RCGNN


def compute_metrics(A_pred, A_true, threshold=0.5):
    """Compute SHD, F1, Precision, Recall at given threshold."""
    A_bin = (A_pred >= threshold).astype(int)
    
    # Flatten for metrics
    pred_flat = A_bin.flatten()
    true_flat = A_true.flatten()
    
    tp = ((pred_flat == 1) & (true_flat == 1)).sum()
    fp = ((pred_flat == 1) & (true_flat == 0)).sum()
    fn = ((pred_flat == 0) & (true_flat == 1)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    shd = int((A_bin != A_true).sum())
    
    return {
        'threshold': threshold,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'shd': shd,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn)
    }


def best_threshold_metrics(A_pred, A_true):
    """Find optimal threshold and compute metrics."""
    pred_flat = A_pred.flatten()
    true_flat = A_true.flatten().astype(int)
    
    # AUPRC
    auprc = average_precision_score(true_flat, pred_flat)
    
    # Search thresholds
    best_f1 = 0
    best_t = 0.5
    best_metrics = None
    
    for t in np.linspace(0.05, 0.95, 50):
        m = compute_metrics(A_pred, A_true, t)
        if m['f1'] > best_f1:
            best_f1 = m['f1']
            best_t = t
            best_metrics = m
    
    best_metrics['auprc'] = auprc
    return best_metrics


def top_k_metrics(A_pred, A_true, k=None):
    """Top-k edge recovery metrics."""
    if k is None:
        k = int(A_true.sum())  # Number of true edges
    
    # Get top-k predicted edges
    flat_pred = A_pred.flatten()
    flat_true = A_true.flatten().astype(int)
    
    top_k_idx = np.argsort(flat_pred)[-k:]
    A_topk = np.zeros_like(flat_pred)
    A_topk[top_k_idx] = 1
    
    tp = ((A_topk == 1) & (flat_true == 1)).sum()
    precision = tp / k if k > 0 else 0
    recall = tp / flat_true.sum() if flat_true.sum() > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'k': k,
        'top_k_f1': f1,
        'top_k_precision': precision,
        'top_k_recall': recall,
        'top_k_tp': int(tp)
    }


def notears_baseline(X, A_true, lambda1=0.1, max_iter=100):
    """
    Simple NOTEARS implementation for baseline comparison.
    X: [N, T, d] or [N, d]
    """
    # Flatten time if needed
    if X.ndim == 3:
        X_flat = X.reshape(-1, X.shape[-1])  # [N*T, d]
    else:
        X_flat = X
    
    # Handle NaN by mean imputation
    X_flat = np.nan_to_num(X_flat, nan=np.nanmean(X_flat, axis=0))
    
    # Standardize
    X_flat = (X_flat - X_flat.mean(axis=0)) / (X_flat.std(axis=0) + 1e-8)
    
    d = X_flat.shape[1]
    n = X_flat.shape[0]
    
    # Initialize W
    W = np.zeros((d, d))
    
    # Simple gradient descent on least squares + L1
    lr = 0.01
    for _ in range(max_iter):
        # Gradient of ||X - XW||^2
        residual = X_flat - X_flat @ W
        grad = -2 * X_flat.T @ residual / n
        
        # Add L1 regularization gradient (soft thresholding direction)
        grad += lambda1 * np.sign(W)
        
        # Zero diagonal (no self-loops)
        np.fill_diagonal(grad, 0)
        
        W = W - lr * grad
        
        # Zero diagonal
        np.fill_diagonal(W, 0)
    
    # Threshold to get binary adjacency
    A_notears = np.abs(W)
    
    return A_notears


def correlation_baseline(X, A_true, quantile=0.9):
    """Correlation-based baseline (undirected)."""
    if X.ndim == 3:
        X_flat = X.reshape(-1, X.shape[-1])
    else:
        X_flat = X
    
    X_flat = np.nan_to_num(X_flat, nan=np.nanmean(X_flat, axis=0))
    
    d = X_flat.shape[1]
    C = np.abs(np.corrcoef(X_flat, rowvar=False))
    np.fill_diagonal(C, 0)
    
    return C


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/interim/uci_air')
    parser.add_argument('--checkpoint', default='artifacts/checkpoints/rcgnn_v2_best.pt')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    
    print("=" * 60)
    print("RC-GNN vs Baselines Evaluation")
    print("=" * 60)
    
    # Load data
    X = np.load(os.path.join(args.data_root, 'X.npy'))
    A_true = np.load(os.path.join(args.data_root, 'A_true.npy'))
    M = np.load(os.path.join(args.data_root, 'M.npy'))
    
    print(f"\nDataset: {args.data_root}")
    print(f"  X shape: {X.shape}")
    print(f"  Missing ratio: {(M == 0).mean():.1%}")
    print(f"  True edges: {int(A_true.sum())}")
    
    d = X.shape[-1]
    num_true_edges = int(A_true.sum())
    
    results = {}
    
    # ========== RC-GNN ==========
    print("\n" + "-" * 40)
    print("RC-GNN (Best Checkpoint)")
    print("-" * 40)
    
    if os.path.exists(args.checkpoint):
        # Load model
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        
        # Reconstruct model (need to match architecture)
        model = RCGNN(
            d=d,
            input_dim=1,  # univariate per node
            hidden_dim=64,
            latent_dim=32,
            n_envs=1
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Get adjacency
        with torch.no_grad():
            A_rcgnn = model.get_adjacency_matrix().numpy()
        
        # Metrics
        rcgnn_best = best_threshold_metrics(A_rcgnn, A_true)
        rcgnn_topk = top_k_metrics(A_rcgnn, A_true, k=num_true_edges)
        
        print(f"  Best threshold: {rcgnn_best['threshold']:.3f}")
        print(f"  F1: {rcgnn_best['f1']:.4f}")
        print(f"  Precision: {rcgnn_best['precision']:.4f}")
        print(f"  Recall: {rcgnn_best['recall']:.4f}")
        print(f"  SHD: {rcgnn_best['shd']}")
        print(f"  AUPRC: {rcgnn_best['auprc']:.4f}")
        print(f"  Top-{num_true_edges} F1: {rcgnn_topk['top_k_f1']:.4f}")
        
        results['rcgnn'] = {**rcgnn_best, **rcgnn_topk}
    else:
        print(f"  Checkpoint not found: {args.checkpoint}")
        results['rcgnn'] = None
    
    # ========== NOTEARS Baseline ==========
    print("\n" + "-" * 40)
    print("NOTEARS (Linear, λ=0.1)")
    print("-" * 40)
    
    A_notears = notears_baseline(X, A_true, lambda1=0.1)
    notears_best = best_threshold_metrics(A_notears, A_true)
    notears_topk = top_k_metrics(A_notears, A_true, k=num_true_edges)
    
    print(f"  Best threshold: {notears_best['threshold']:.3f}")
    print(f"  F1: {notears_best['f1']:.4f}")
    print(f"  Precision: {notears_best['precision']:.4f}")
    print(f"  Recall: {notears_best['recall']:.4f}")
    print(f"  SHD: {notears_best['shd']}")
    print(f"  AUPRC: {notears_best['auprc']:.4f}")
    print(f"  Top-{num_true_edges} F1: {notears_topk['top_k_f1']:.4f}")
    
    results['notears'] = {**notears_best, **notears_topk}
    
    # ========== Correlation Baseline ==========
    print("\n" + "-" * 40)
    print("Correlation Baseline (Top-k edges)")
    print("-" * 40)
    
    A_corr = correlation_baseline(X, A_true)
    corr_best = best_threshold_metrics(A_corr, A_true)
    corr_topk = top_k_metrics(A_corr, A_true, k=num_true_edges)
    
    print(f"  Best threshold: {corr_best['threshold']:.3f}")
    print(f"  F1: {corr_best['f1']:.4f}")
    print(f"  Precision: {corr_best['precision']:.4f}")
    print(f"  Recall: {corr_best['recall']:.4f}")
    print(f"  SHD: {corr_best['shd']}")
    print(f"  AUPRC: {corr_best['auprc']:.4f}")
    print(f"  Top-{num_true_edges} F1: {corr_topk['top_k_f1']:.4f}")
    
    results['correlation'] = {**corr_best, **corr_topk}
    
    # ========== Summary Table ==========
    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60)
    print(f"{'Method':<20} {'F1':>8} {'AUPRC':>8} {'SHD':>6} {'Top-k F1':>10}")
    print("-" * 60)
    
    for method, m in results.items():
        if m is not None:
            print(f"{method:<20} {m['f1']:>8.4f} {m['auprc']:>8.4f} {m['shd']:>6} {m['top_k_f1']:>10.4f}")
    
    # Save results
    os.makedirs('artifacts/baseline_comparison', exist_ok=True)
    with open('artifacts/baseline_comparison/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to artifacts/baseline_comparison/results.json")
    
    return results


if __name__ == '__main__':
    main()
