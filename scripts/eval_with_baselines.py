#!/usr/bin/env python3
"""
Proper evaluation: RC-GNN vs ALL 7 baselines on UCI Air.
- Loads best checkpoint
- Evaluates at optimal threshold and Top-K
- Runs ALL 7 IID-appropriate baselines with mask handling
- Reports fair comparison

BASELINES (all IID-appropriate, NO temporal methods):
  - PC: Constraint-based (Fisher z-test, Meek rules)
  - GES: Score-based (BIC-greedy DAG search)
  - NOTEARS: Single-shot acyclicity penalty (linear)
  - NOTEARS-MLP: Neural NOTEARS (MLP)
  - GOLEM: Likelihood-based neural DAG learning
  - DAG-GNN: Graph neural network DAG learning  
  - GraN-DAG: Gradient-based neural DAG learning
  - Correlation: Simple correlation baseline (undirected)
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

# Import ALL 7 IID-appropriate baselines from centralized module
from src.training.baselines import (
    pc_algorithm,
    ges_algorithm,
    notears_linear,
    notears_mlp,
    golem,
    dag_gnn,
    gran_dag,
    correlation_baseline,
    impute_with_mask,
)


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
        k = int(A_true.sum()) # Number of true edges
    
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


# Local baseline implementations REMOVED - now using centralized module
# from src/training/baselines.py which includes:
# - pc_algorithm(Xw, Mw) 
# - ges_algorithm(Xw, Mw)
# - notears_linear(Xw, Mw)
# - notears_mlp(Xw, Mw)
# - golem(Xw, Mw)
# - dag_gnn(Xw, Mw)
# - gran_dag(Xw, Mw)
# - correlation_baseline(Xw, Mw)
# All methods handle missingness mask properly via mean imputation.


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/interim/uci_air')
    parser.add_argument('--checkpoint', default='artifacts/checkpoints/rcgnn_v2_best.pt')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    
    print("=" * 70)
    print("RC-GNN vs ALL 7 BASELINES Evaluation")
    print("=" * 70)
    
    # Load data
    X = np.load(os.path.join(args.data_root, 'X.npy'))
    A_true = np.load(os.path.join(args.data_root, 'A_true.npy'))
    
    # Load missingness mask (CRITICAL for fair baseline comparison)
    M_path = os.path.join(args.data_root, 'M.npy')
    if os.path.exists(M_path):
        M = np.load(M_path)
    else:
        M = np.ones_like(X)  # No missingness
        print("[WARN] M.npy not found, assuming no missing data")
    
    print(f"\nDataset: {args.data_root}")
    print(f" X shape: {X.shape}")
    print(f" Missing ratio: {(M == 0).mean():.1%}")
    print(f" True edges: {int(A_true.sum())}")
    
    d = X.shape[-1]
    num_true_edges = int(A_true.sum())
    
    results = {}
    
    # ========== RC-GNN ==========
    print("\n" + "-" * 50)
    print("RC-GNN (Best Checkpoint)")
    print("-" * 50)
    
    if os.path.exists(args.checkpoint):
        # Load model
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        
        # Reconstruct model (need to match architecture)
        model = RCGNN(
            d=d,
            input_dim=1, # univariate per node
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
        
        print(f" Best threshold: {rcgnn_best['threshold']:.3f}")
        print(f" F1: {rcgnn_best['f1']:.4f}")
        print(f" Precision: {rcgnn_best['precision']:.4f}")
        print(f" Recall: {rcgnn_best['recall']:.4f}")
        print(f" SHD: {rcgnn_best['shd']}")
        print(f" AUPRC: {rcgnn_best['auprc']:.4f}")
        print(f" Top-{num_true_edges} F1: {rcgnn_topk['top_k_f1']:.4f}")
        
        results['RC-GNN'] = {**rcgnn_best, **rcgnn_topk}
    else:
        print(f" Checkpoint not found: {args.checkpoint}")
        results['RC-GNN'] = None
    
    # ========== ALL 7 BASELINES (with mask handling) ==========
    print("\n" + "=" * 70)
    print("RUNNING ALL 7 IID-APPROPRIATE BASELINES")
    print("(All baselines use missingness mask for proper imputation)")
    print("=" * 70)
    
    # Define all 7 baselines with their functions
    baseline_methods = {
        'Correlation': lambda X, M: correlation_baseline(X, Mw=M),
        'PC': lambda X, M: pc_algorithm(X, Mw=M),
        'GES': lambda X, M: ges_algorithm(X, Mw=M),
        'NOTEARS': lambda X, M: notears_linear(X, Mw=M),
        'NOTEARS-MLP': lambda X, M: notears_mlp(X, Mw=M),
        'GOLEM': lambda X, M: golem(X, Mw=M),
        'DAG-GNN': lambda X, M: dag_gnn(X, Mw=M),
        'GraN-DAG': lambda X, M: gran_dag(X, Mw=M),
    }
    
    for method_name, method_fn in baseline_methods.items():
        print(f"\n{'-'*50}")
        print(f"{method_name}")
        print(f"{'-'*50}")
        
        try:
            A_pred = method_fn(X, M)
            method_best = best_threshold_metrics(A_pred, A_true)
            method_topk = top_k_metrics(A_pred, A_true, k=num_true_edges)
            
            print(f" Best threshold: {method_best['threshold']:.3f}")
            print(f" F1: {method_best['f1']:.4f}")
            print(f" Precision: {method_best['precision']:.4f}")
            print(f" Recall: {method_best['recall']:.4f}")
            print(f" SHD: {method_best['shd']}")
            print(f" AUPRC: {method_best['auprc']:.4f}")
            print(f" Top-{num_true_edges} F1: {method_topk['top_k_f1']:.4f}")
            
            results[method_name] = {**method_best, **method_topk}
        except Exception as e:
            print(f" ERROR: {str(e)[:80]}")
            results[method_name] = None
    
    # ========== Summary Table ==========
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON (RC-GNN vs 7 Baselines)")
    print("=" * 70)
    print(f"{'Method':<15} {'F1':>8} {'AUPRC':>8} {'SHD':>6} {'Top-k F1':>10}")
    print("-" * 70)
    
    for method, m in results.items():
        if m is not None:
            print(f"{method:<15} {m['f1']:>8.4f} {m['auprc']:>8.4f} {m['shd']:>6} {m['top_k_f1']:>10.4f}")
    
    # Save results
    os.makedirs('artifacts/baseline_comparison', exist_ok=True)
    with open('artifacts/baseline_comparison/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[DONE] Results saved to artifacts/baseline_comparison/results.json")
    print(f"[INFO] All 7 baselines used missingness mask for fair comparison")
    
    return results


if __name__ == '__main__':
    main()
