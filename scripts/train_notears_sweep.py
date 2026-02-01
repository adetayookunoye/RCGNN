#!/usr/bin/env python3
"""
Train NOTEARS baseline across corruption levels with multiple seeds.
Uses mean imputation for fair comparison.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import json
import argparse
from notears import notears_linear
from src.models.utils import evaluate_adjacency, best_threshold_f1

def impute_mean(X, M):
    """Simple mean imputation for missing values."""
    X_imp = X.copy()
    for i in range(X.shape[2]): # For each feature
        feature_data = X[:, :, i]
        observed = M[:, :, i] == 1
        if observed.sum() > 0:
            mean_val = feature_data[observed].mean()
            X_imp[observed == False, i] = mean_val
    return X_imp

def train_notears(data_dir, seed, lambda_param=0.1):
    """Train NOTEARS with given seed."""
    np.random.seed(seed)
    
    # Load data
    X = np.load(Path(data_dir) / "X.npy")
    M = np.load(Path(data_dir) / "M.npy")
    A_true = np.load(Path(data_dir) / "A_true.npy")
    
    # Impute missing values
    X_imp = impute_mean(X, M)
    
    # Reshape to [N*T, d]
    N, T, d = X.shape
    X_flat = X_imp.reshape(-1, d)
    
    # Run NOTEARS
    W_est = notears_linear(X_flat, lambda1=lambda_param, loss_type='l2')
    A_pred = (W_est != 0).astype(float)
    
    # Evaluate
    metrics = evaluate_adjacency(A_pred, A_true)
    best_f1_info = best_threshold_f1(np.abs(W_est), A_true)
    
    return {
        'f1': best_f1_info['f1'],
        'precision': best_f1_info['precision'],
        'recall': best_f1_info['recall'],
        'auprc': metrics['auprc'],
        'auroc': metrics['auroc'],
        'shd': metrics['shd'],
        'nnz': metrics['nnz']
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corruption', type=float, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--lambda_param', type=float, default=0.1)
    args = parser.parse_args()
    
    data_dir = f"data/interim/uci_air_{int(args.corruption*100):02d}"
    
    print("=" * 60)
    print(f"TRAINING NOTEARS: {args.corruption:.0%} missing, seed={args.seed}")
    print("=" * 60)
    print(f"Data: {data_dir}")
    print(f"Lambda: {args.lambda_param}")
    print()
    
    # Train
    results = train_notears(data_dir, args.seed, args.lambda_param)
    
    # Save results
    output_dir = Path("artifacts") / "corruption_sweep" / f"corruption_{int(args.corruption*100):02d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"notears_seed{args.seed}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"F1: {results['f1']:.3f}")
    print(f"AUPRC: {results['auprc']:.3f}")
    print(f"SHD: {results['shd']}")
    print()
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    main()
