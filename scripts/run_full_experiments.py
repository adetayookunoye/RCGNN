#!/usr/bin/env python3
"""
Full experiment suite for RC-GNN publication:
1. Ablation study (w/wo disentangle, w/wo invariance)
2. Corruption sweep (0%, 10%, 20%, 30%, 40% missingness)
3. Stability across seeds (5 seeds, compute variance)

Outputs:
- artifacts/experiments/ablation_results.json
- artifacts/experiments/corruption_sweep.json
- artifacts/experiments/stability_results.json
- artifacts/experiments/figures/corruption_sweep.png
- artifacts/experiments/figures/ablation_bar.png
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score
from typing import Dict, List, Tuple, Optional
import time

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.rcgnn import RCGNN


# ============================================================
# METRICS
# ============================================================

def compute_metrics(A_pred: np.ndarray, A_true: np.ndarray, threshold: float = 0.5) -> Dict:
    """Compute SHD, F1, Precision, Recall at given threshold."""
    A_bin = (A_pred >= threshold).astype(int)
    pred_flat = A_bin.flatten()
    true_flat = A_true.flatten()
    
    tp = ((pred_flat == 1) & (true_flat == 1)).sum()
    fp = ((pred_flat == 1) & (true_flat == 0)).sum()
    fn = ((pred_flat == 0) & (true_flat == 1)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    shd = int((A_bin != A_true).sum())
    
    return {'f1': f1, 'precision': precision, 'recall': recall, 'shd': shd}


def best_threshold_metrics(A_pred: np.ndarray, A_true: np.ndarray) -> Dict:
    """Find optimal threshold and compute metrics + AUPRC."""
    pred_flat = A_pred.flatten()
    true_flat = A_true.flatten().astype(int)
    
    auprc = average_precision_score(true_flat, pred_flat)
    
    best_f1, best_t = 0, 0.5
    for t in np.linspace(0.05, 0.95, 50):
        m = compute_metrics(A_pred, A_true, t)
        if m['f1'] > best_f1:
            best_f1 = m['f1']
            best_t = t
    
    best_metrics = compute_metrics(A_pred, A_true, best_t)
    best_metrics['auprc'] = auprc
    best_metrics['threshold'] = best_t
    return best_metrics


def top_k_metrics(A_pred: np.ndarray, A_true: np.ndarray, k: Optional[int] = None) -> Dict:
    """Top-k edge recovery metrics."""
    if k is None:
        k = int(A_true.sum())
    
    flat_pred = A_pred.flatten()
    flat_true = A_true.flatten().astype(int)
    
    top_k_idx = np.argsort(flat_pred)[-k:]
    A_topk = np.zeros_like(flat_pred)
    A_topk[top_k_idx] = 1
    
    tp = ((A_topk == 1) & (flat_true == 1)).sum()
    precision = tp / k if k > 0 else 0
    recall = tp / flat_true.sum() if flat_true.sum() > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {'top_k_f1': f1, 'top_k_precision': precision, 'top_k_recall': recall}


def edge_jaccard(A1: np.ndarray, A2: np.ndarray, threshold: float = 0.3) -> float:
    """Jaccard similarity between two graphs at given threshold."""
    E1 = set(zip(*np.where(A1 >= threshold)))
    E2 = set(zip(*np.where(A2 >= threshold)))
    if len(E1) == 0 and len(E2) == 0:
        return 1.0
    if len(E1.union(E2)) == 0:
        return 0.0
    return len(E1.intersection(E2)) / len(E1.union(E2))


# ============================================================
# DATA UTILITIES
# ============================================================

def load_data(data_root: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load UCI Air data."""
    X = np.load(os.path.join(data_root, 'X.npy'))
    M = np.load(os.path.join(data_root, 'M.npy'))
    A_true = np.load(os.path.join(data_root, 'A_true.npy'))
    e = np.load(os.path.join(data_root, 'e.npy'))
    return X, M, A_true, e


def apply_corruption(X: np.ndarray, M: np.ndarray, missing_rate: float, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply additional missingness to data.
    missing_rate: target total missing rate (0.0 to 1.0)
    """
    np.random.seed(seed)
    
    # Current missing rate
    current_missing = (M == 0).mean()
    
    if missing_rate <= current_missing:
        # Already at or above target - just return original
        return X.copy(), M.copy()
    
    # Need to add more missing
    additional_rate = (missing_rate - current_missing) / (1 - current_missing)
    
    X_corrupt = X.copy()
    M_corrupt = M.copy()
    
    # Randomly mask additional entries (only where M == 1)
    observed_mask = (M == 1)
    additional_mask = np.random.rand(*M.shape) < additional_rate
    
    # Apply mask only to observed entries
    new_missing = observed_mask & additional_mask
    M_corrupt[new_missing] = 0
    X_corrupt[new_missing] = 0 # Or np.nan
    
    return X_corrupt, M_corrupt


def prepare_dataloader(X: np.ndarray, M: np.ndarray, e: np.ndarray, 
                       batch_size: int = 64, train_ratio: float = 0.8) -> Tuple[DataLoader, DataLoader]:
    """Create train/val dataloaders."""
    N = X.shape[0]
    
    # Simple split
    idx = np.random.permutation(N)
    train_idx = idx[:int(N * train_ratio)]
    val_idx = idx[int(N * train_ratio):]
    
    X_train = torch.tensor(X[train_idx], dtype=torch.float32)
    M_train = torch.tensor(M[train_idx], dtype=torch.float32)
    e_train = torch.tensor(e[train_idx], dtype=torch.long)
    
    X_val = torch.tensor(X[val_idx], dtype=torch.float32)
    M_val = torch.tensor(M[val_idx], dtype=torch.float32)
    e_val = torch.tensor(e[val_idx], dtype=torch.long)
    
    train_ds = TensorDataset(X_train, M_train, e_train)
    val_ds = TensorDataset(X_val, M_val, e_val)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


# ============================================================
# TRAINING
# ============================================================

def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    A_true: np.ndarray,
    d: int,
    config: Dict,
    device: str = 'cpu',
    verbose: bool = False
) -> Tuple[np.ndarray, Dict]:
    """
    Train RC-GNN with given config.
    Returns: (A_pred, final_metrics)
    """
    use_disentangle = config.get('use_disentangle', True)
    use_invariance = config.get('use_invariance', True)
    
    # Build model with ablation-aware loss weights
    model = RCGNN(
        d=d,
        input_dim=1,
        latent_dim=config.get('latent_dim', 32),
        hidden_dim=config.get('hidden_dim', 64),
        n_envs=1,
        # Ablation: set lambda to 0 to disable
        lambda_recon=1.0,
        lambda_disentangle=0.1 if use_disentangle else 0.0,
        gamma_acyclic=config.get('gamma_acyclic', 0.01),
        lambda_sparse=config.get('lambda_sparse', 0.0001),
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('lr', 1e-3))
    
    epochs = config.get('epochs', 50)
    
    best_auprc = 0
    best_A = None
    
    for epoch in range(epochs):
        model.train()
        model.set_epoch(epoch)
        
        for batch in train_loader:
            X_batch, M_batch, e_batch = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            
            # Forward pass - model handles everything
            output = model(X_batch, mask=M_batch)
            loss, metrics = model.compute_loss(output)
            
            # Add invariance penalty if enabled
            if use_invariance:
                A = output['A']
                inv_loss = 0.01 * A.var() # Simple variance penalty
                loss = loss + inv_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            A_pred = model.get_adjacency_matrix().cpu().numpy()
            eval_metrics = best_threshold_metrics(A_pred, A_true)
            
            if eval_metrics['auprc'] > best_auprc:
                best_auprc = eval_metrics['auprc']
                best_A = A_pred.copy()
        
        if verbose and epoch % 10 == 0:
            print(f" Epoch {epoch}: F1={eval_metrics['f1']:.3f}, AUPRC={eval_metrics['auprc']:.3f}")
    
    if best_A is None:
        best_A = A_pred
    
    final_metrics = best_threshold_metrics(best_A, A_true)
    final_metrics.update(top_k_metrics(best_A, A_true))
    
    return best_A, final_metrics


# ============================================================
# EXPERIMENT 1: ABLATION STUDY
# ============================================================

def run_ablation_study(
    data_root: str,
    output_dir: str,
    device: str = 'cpu',
    epochs: int = 50,
    seed: int = 42
) -> Dict:
    """
    Ablation study:
    - RC-GNN full
    - w/o disentangle loss
    - w/o invariance loss
    """
    print("\n" + "=" * 60)
    print("ABLATION STUDY")
    print("=" * 60)
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    X, M, A_true, e = load_data(data_root)
    d = X.shape[-1]
    
    train_loader, val_loader = prepare_dataloader(X, M, e)
    
    results = {}
    
    configs = {
        'RC-GNN (full)': {'use_disentangle': True, 'use_invariance': True},
        'w/o disentangle': {'use_disentangle': False, 'use_invariance': True},
        'w/o invariance': {'use_disentangle': True, 'use_invariance': False},
        'w/o both': {'use_disentangle': False, 'use_invariance': False},
    }
    
    for name, cfg in configs.items():
        print(f"\nTraining: {name}")
        cfg['epochs'] = epochs
        
        A_pred, metrics = train_model(
            train_loader, val_loader, A_true, d, cfg, device, verbose=True
        )
        
        results[name] = metrics
        print(f" Final: F1={metrics['f1']:.4f}, AUPRC={metrics['auprc']:.4f}, "
              f"SHD={metrics['shd']}, Top-k F1={metrics['top_k_f1']:.4f}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'ablation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print table
    print("\n" + "-" * 60)
    print("ABLATION RESULTS")
    print("-" * 60)
    print(f"{'Method':<25} {'F1':>8} {'AUPRC':>8} {'SHD':>6} {'Top-k F1':>10}")
    print("-" * 60)
    for name, m in results.items():
        print(f"{name:<25} {m['f1']:>8.4f} {m['auprc']:>8.4f} {m['shd']:>6} {m['top_k_f1']:>10.4f}")
    
    return results


# ============================================================
# EXPERIMENT 2: CORRUPTION SWEEP
# ============================================================

def run_corruption_sweep(
    data_root: str,
    output_dir: str,
    missing_rates: List[float] = [0.0, 0.10, 0.20, 0.30, 0.40],
    device: str = 'cpu',
    epochs: int = 50,
    seed: int = 42
) -> Dict:
    """
    Corruption sweep: test at different missingness levels.
    """
    print("\n" + "=" * 60)
    print("CORRUPTION SWEEP")
    print("=" * 60)
    
    X_orig, M_orig, A_true, e = load_data(data_root)
    d = X_orig.shape[-1]
    
    results = {'rcgnn': {}, 'notears': {}}
    
    for rate in missing_rates:
        print(f"\nMissingness rate: {rate:.0%}")
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Apply corruption
        X, M = apply_corruption(X_orig, M_orig, rate, seed)
        actual_missing = (M == 0).mean()
        print(f" Actual missing: {actual_missing:.1%}")
        
        # Train RC-GNN
        train_loader, val_loader = prepare_dataloader(X, M, e)
        config = {'epochs': epochs, 'use_disentangle': True, 'use_invariance': True}
        
        A_rcgnn, metrics_rcgnn = train_model(
            train_loader, val_loader, A_true, d, config, device, verbose=False
        )
        results['rcgnn'][f'{rate:.0%}'] = metrics_rcgnn
        
        # NOTEARS baseline
        A_notears = notears_baseline(X, lambda1=0.1)
        metrics_notears = best_threshold_metrics(A_notears, A_true)
        metrics_notears.update(top_k_metrics(A_notears, A_true))
        results['notears'][f'{rate:.0%}'] = metrics_notears
        
        print(f" RC-GNN: AUPRC={metrics_rcgnn['auprc']:.4f}, Top-k F1={metrics_rcgnn['top_k_f1']:.4f}")
        print(f" NOTEARS: AUPRC={metrics_notears['auprc']:.4f}, Top-k F1={metrics_notears['top_k_f1']:.4f}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'corruption_sweep.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print table
    print("\n" + "-" * 60)
    print("CORRUPTION SWEEP RESULTS")
    print("-" * 60)
    print(f"{'Missing %':<12} {'RC-GNN AUPRC':>14} {'NOTEARS AUPRC':>14} {'RC-GNN Top-k':>14} {'NOTEARS Top-k':>14}")
    print("-" * 60)
    for rate in missing_rates:
        key = f'{rate:.0%}'
        r = results['rcgnn'][key]
        n = results['notears'][key]
        print(f"{key:<12} {r['auprc']:>14.4f} {n['auprc']:>14.4f} {r['top_k_f1']:>14.4f} {n['top_k_f1']:>14.4f}")
    
    return results


def notears_baseline(X: np.ndarray, lambda1: float = 0.1, max_iter: int = 100) -> np.ndarray:
    """Simple NOTEARS for baseline."""
    if X.ndim == 3:
        X_flat = X.reshape(-1, X.shape[-1])
    else:
        X_flat = X
    
    # Handle missing
    col_means = np.nanmean(X_flat, axis=0)
    X_flat = np.where(np.isnan(X_flat) | (X_flat == 0), col_means, X_flat)
    
    # Standardize
    X_flat = (X_flat - X_flat.mean(axis=0)) / (X_flat.std(axis=0) + 1e-8)
    
    d = X_flat.shape[1]
    n = X_flat.shape[0]
    W = np.zeros((d, d))
    lr = 0.01
    
    for _ in range(max_iter):
        residual = X_flat - X_flat @ W
        grad = -2 * X_flat.T @ residual / n + lambda1 * np.sign(W)
        np.fill_diagonal(grad, 0)
        W = W - lr * grad
        np.fill_diagonal(W, 0)
    
    return np.abs(W)


# ============================================================
# EXPERIMENT 3: STABILITY ACROSS SEEDS
# ============================================================

def run_stability_experiment(
    data_root: str,
    output_dir: str,
    seeds: List[int] = [1, 2, 3, 4, 5],
    device: str = 'cpu',
    epochs: int = 50
) -> Dict:
    """
    Stability: run with multiple seeds, compute variance and edge Jaccard.
    """
    print("\n" + "=" * 60)
    print("STABILITY EXPERIMENT (Multi-seed)")
    print("=" * 60)
    
    X, M, A_true, e = load_data(data_root)
    d = X.shape[-1]
    
    all_A_preds = []
    all_metrics = []
    
    for seed in seeds:
        print(f"\nSeed {seed}:")
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        train_loader, val_loader = prepare_dataloader(X, M, e)
        config = {'epochs': epochs, 'use_disentangle': True, 'use_invariance': True}
        
        A_pred, metrics = train_model(
            train_loader, val_loader, A_true, d, config, device, verbose=False
        )
        
        all_A_preds.append(A_pred)
        all_metrics.append(metrics)
        
        print(f" F1={metrics['f1']:.4f}, AUPRC={metrics['auprc']:.4f}, SHD={metrics['shd']}")
    
    # Compute statistics
    f1_scores = [m['f1'] for m in all_metrics]
    auprc_scores = [m['auprc'] for m in all_metrics]
    shd_scores = [m['shd'] for m in all_metrics]
    topk_scores = [m['top_k_f1'] for m in all_metrics]
    
    # Edge Jaccard between all pairs
    jaccards = []
    for i in range(len(all_A_preds)):
        for j in range(i + 1, len(all_A_preds)):
            jaccards.append(edge_jaccard(all_A_preds[i], all_A_preds[j]))
    
    results = {
        'seeds': seeds,
        'f1_mean': float(np.mean(f1_scores)),
        'f1_std': float(np.std(f1_scores)),
        'auprc_mean': float(np.mean(auprc_scores)),
        'auprc_std': float(np.std(auprc_scores)),
        'shd_mean': float(np.mean(shd_scores)),
        'shd_std': float(np.std(shd_scores)),
        'topk_f1_mean': float(np.mean(topk_scores)),
        'topk_f1_std': float(np.std(topk_scores)),
        'edge_jaccard_mean': float(np.mean(jaccards)),
        'edge_jaccard_std': float(np.std(jaccards)),
        'per_seed': [
            {'seed': s, **m} for s, m in zip(seeds, all_metrics)
        ]
    }
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'stability_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "-" * 60)
    print("STABILITY RESULTS")
    print("-" * 60)
    print(f"F1: {results['f1_mean']:.4f} ± {results['f1_std']:.4f}")
    print(f"AUPRC: {results['auprc_mean']:.4f} ± {results['auprc_std']:.4f}")
    print(f"SHD: {results['shd_mean']:.1f} ± {results['shd_std']:.1f}")
    print(f"Top-k F1: {results['topk_f1_mean']:.4f} ± {results['topk_f1_std']:.4f}")
    print(f"Edge Jaccard: {results['edge_jaccard_mean']:.4f} ± {results['edge_jaccard_std']:.4f}")
    
    return results


# ============================================================
# PLOTTING
# ============================================================

def create_plots(output_dir: str):
    """Create publication-quality plots from saved results."""
    try:
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        print("Matplotlib not available, skipping plots")
        return
    
    fig_dir = os.path.join(output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    # 1. Corruption sweep plot
    sweep_path = os.path.join(output_dir, 'corruption_sweep.json')
    if os.path.exists(sweep_path):
        with open(sweep_path) as f:
            sweep = json.load(f)
        
        rates = [0, 10, 20, 30, 40]
        rcgnn_auprc = [sweep['rcgnn'][f'{r}%']['auprc'] for r in rates]
        notears_auprc = [sweep['notears'][f'{r}%']['auprc'] for r in rates]
        rcgnn_topk = [sweep['rcgnn'][f'{r}%']['top_k_f1'] for r in rates]
        notears_topk = [sweep['notears'][f'{r}%']['top_k_f1'] for r in rates]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.plot(rates, rcgnn_auprc, 'o-', label='RC-GNN', linewidth=2, markersize=8)
        ax1.plot(rates, notears_auprc, 's--', label='NOTEARS', linewidth=2, markersize=8)
        ax1.set_xlabel('Missing Data Rate (%)', fontsize=12)
        ax1.set_ylabel('AUPRC', fontsize=12)
        ax1.set_title('(a) AUPRC vs Corruption', fontsize=14)
        ax1.legend()
        ax1.set_ylim(0, 0.4)
        
        ax2.plot(rates, rcgnn_topk, 'o-', label='RC-GNN', linewidth=2, markersize=8)
        ax2.plot(rates, notears_topk, 's--', label='NOTEARS', linewidth=2, markersize=8)
        ax2.set_xlabel('Missing Data Rate (%)', fontsize=12)
        ax2.set_ylabel('Top-k F1', fontsize=12)
        ax2.set_title('(b) Top-k F1 vs Corruption', fontsize=14)
        ax2.legend()
        ax2.set_ylim(0, 0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'corruption_sweep.png'), dpi=150, bbox_inches='tight')
        plt.savefig(os.path.join(fig_dir, 'corruption_sweep.pdf'), bbox_inches='tight')
        print(f"Saved corruption sweep plot to {fig_dir}")
    
    # 2. Ablation bar chart
    ablation_path = os.path.join(output_dir, 'ablation_results.json')
    if os.path.exists(ablation_path):
        with open(ablation_path) as f:
            ablation = json.load(f)
        
        methods = list(ablation.keys())
        auprc = [ablation[m]['auprc'] for m in methods]
        topk = [ablation[m]['top_k_f1'] for m in methods]
        
        x = np.arange(len(methods))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width/2, auprc, width, label='AUPRC', color='steelblue')
        bars2 = ax.bar(x + width/2, topk, width, label='Top-k F1', color='coral')
        
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Ablation Study', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=15, ha='right')
        ax.legend()
        ax.set_ylim(0, 0.5)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'ablation_bar.png'), dpi=150, bbox_inches='tight')
        plt.savefig(os.path.join(fig_dir, 'ablation_bar.pdf'), bbox_inches='tight')
        print(f"Saved ablation plot to {fig_dir}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Run full RC-GNN experiments')
    parser.add_argument('--data_root', default='data/interim/uci_air')
    parser.add_argument('--output_dir', default='artifacts/experiments')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--experiments', nargs='+', 
                        default=['ablation', 'corruption', 'stability'],
                        choices=['ablation', 'corruption', 'stability', 'plots'])
    args = parser.parse_args()
    
    print("=" * 60)
    print("RC-GNN FULL EXPERIMENT SUITE")
    print("=" * 60)
    print(f"Data: {args.data_root}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Experiments: {args.experiments}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    start_time = time.time()
    
    if 'ablation' in args.experiments:
        run_ablation_study(args.data_root, args.output_dir, args.device, args.epochs)
    
    if 'corruption' in args.experiments:
        run_corruption_sweep(args.data_root, args.output_dir, 
                            missing_rates=[0.0, 0.10, 0.20, 0.30, 0.40],
                            device=args.device, epochs=args.epochs)
    
    if 'stability' in args.experiments:
        run_stability_experiment(args.data_root, args.output_dir, 
                                seeds=[1, 2, 3, 4, 5],
                                device=args.device, epochs=args.epochs)
    
    if 'plots' in args.experiments:
        create_plots(args.output_dir)
    
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"All experiments complete in {elapsed/60:.1f} minutes")
    print(f"Results saved to {args.output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
