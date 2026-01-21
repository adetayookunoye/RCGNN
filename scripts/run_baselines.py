
#!/usr/bin/env python3
"""
Run individual baseline methods on datasets.

Supports: notears_lite, notears_linear, granger, pcmci+, dag_gnn_simple, correlation

Usage:
  python scripts/run_baselines.py --method notears_lite --config configs/data.yaml
  python scripts/run_baselines.py --method notears_linear --config configs/data.yaml
  python scripts/run_baselines.py --method granger --config configs/data.yaml
  python scripts/run_baselines.py --method pcmci_plus --config configs/data.yaml
  python scripts/run_baselines.py --method dag_gnn_simple --config configs/data.yaml
"""

import argparse, yaml, os, numpy as np
from pathlib import Path

import path_helper  # noqa: F401

from src.training.baselines import (
    notears_lite, notears_linear, granger_causality,
    pcmci_plus, dag_gnn_simple, compute_correlation_adjacency
)

def compute_correlation_adjacency(X, threshold=None):
    """Compute correlation-based adjacency."""
    if X.ndim == 3:
        X_flat = X.reshape(-1, X.shape[-1])
    else:
        X_flat = X
    
    corr = np.abs(np.corrcoef(X_flat.T))
    np.fill_diagonal(corr, 0)
    corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=0.0)
    
    if np.ptp(corr) > 1e-12:
        A_corr = (corr - corr.min()) / (corr.max() - corr.min() + 1e-10)
    else:
        A_corr = corr
    
    return A_corr

def main():
    parser = argparse.ArgumentParser(description="Run baseline methods")
    parser.add_argument("--method", default="notears_lite", 
                       choices=["correlation", "notears_lite", "notears_linear", 
                               "granger", "pcmci_plus", "dag_gnn_simple"],
                       help="Baseline method to run")
    parser.add_argument("--config", required=True, help="Data config YAML")
    parser.add_argument("--output-dir", default="reports/tables", help="Output directory")
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        dc = yaml.safe_load(f)
    
    root = dc["paths"]["root"]
    X = np.load(os.path.join(root, "X.npy"))  # [N,T,d] or [N,d]
    A_true = np.load(os.path.join(root, "A_true.npy"))
    
    print(f"\n{'='*80}")
    print(f"Running {args.method} baseline")
    print(f"{'='*80}")
    print(f"Data shape: X={X.shape}, A_true={A_true.shape}")
    
    # Run baseline
    try:
        if args.method == "correlation":
            print("📊 Computing Correlation adjacency...")
            A_hat = compute_correlation_adjacency(X)
            A_hat = (A_hat > np.median(A_hat)).astype(float)
        
        elif args.method == "notears_lite":
            print("📊 Computing NOTears-lite...")
            if X.ndim == 2:
                X_3d = X.reshape(1, X.shape[0], X.shape[1])
            else:
                X_3d = X
            A_hat = notears_lite(X_3d)
        
        elif args.method == "notears_linear":
            print("📊 Computing Full NOTears (Lagrangian)...")
            if X.ndim == 3:
                X_avg = X.mean(axis=1)
            else:
                X_avg = X
            A_hat = notears_linear(X_avg, lambda1=0.1, lambda2=5.0, max_iter=100)
        
        elif args.method == "granger":
            print("📊 Computing Granger Causality...")
            if X.ndim == 2:
                d = X.shape[1]
                N = min(X.shape[0] // 10, 100)
                X_ts = X[:N*10].reshape(N, 10, d)
            else:
                X_ts = X
            A_hat = granger_causality(X_ts, max_lag=2, significance=0.05)
        
        elif args.method == "pcmci_plus":
            print("📊 Computing PCMCI+...")
            if X.ndim == 2:
                d = X.shape[1]
                N = min(X.shape[0] // 10, 100)
                X_ts = X[:N*10].reshape(N, 10, d)
            else:
                X_ts = X
            A_hat = pcmci_plus(X_ts, significance=0.05, max_lag=2)
        
        elif args.method == "dag_gnn_simple":
            print("📊 Computing DAG-GNN...")
            A_hat = dag_gnn_simple(X, hidden_dim=64, num_layers=2)
        
        # Compute SHD
        shd = int((A_hat != A_true).sum())
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save results
        output_file = os.path.join(args.output_dir, f"baseline_{args.method}_adj.npy")
        np.save(output_file, A_hat)
        
        print(f"\n✅ Results:")
        print(f"   SHD (Structural Hamming Distance): {shd}")
        print(f"   Edges predicted: {int(A_hat.sum())}")
        print(f"   Edges in ground truth: {int(A_true.sum())}")
        print(f"   Saved to: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Error running {args.method}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
