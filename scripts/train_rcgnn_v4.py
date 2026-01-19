#!/usr/bin/env python3
"""
RC-GNN V4 Training Script

V4 adds causal identifiability priors to distinguish causation from correlation:
1. Intervention-aware structure learning
2. Orientation penalty (entropy asymmetry)
3. Edge counterfactual validation
4. Environment-specific decoders

Usage:
    python scripts/train_rcgnn_v4.py --data_root data/interim/uci_air_c/regimes_3
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.rcgnn import RCGNN
from src.models.causal_priors import diagnose_correlation_vs_causation


def load_data(data_root: str) -> Dict[str, np.ndarray]:
    """Load numpy data files."""
    root = Path(data_root)
    
    X = np.load(root / "X.npy")
    M = np.load(root / "M.npy") if (root / "M.npy").exists() else np.ones_like(X)
    e = np.load(root / "e.npy") if (root / "e.npy").exists() else np.zeros(X.shape[0], dtype=np.int64)
    A_true = np.load(root / "A_true.npy") if (root / "A_true.npy").exists() else None
    
    # Load config for intervention info
    config = None
    if (root / "config.json").exists():
        with open(root / "config.json") as f:
            config = json.load(f)
    
    return {
        "X": X,
        "M": M,
        "e": e,
        "A_true": A_true,
        "config": config,
    }


def normalize_data(X: np.ndarray) -> np.ndarray:
    """Z-score normalize."""
    mean = X.mean()
    std = X.std() + 1e-8
    return (X - mean) / std


def compute_topk_f1(A_pred: torch.Tensor, A_true: torch.Tensor, k: int = None) -> Dict[str, float]:
    """Compute TopK metrics."""
    A_pred_np = A_pred.detach().cpu().numpy().copy()
    A_true_np = (A_true.cpu().numpy() > 0.5).astype(int)
    
    n_true_edges = int(A_true_np.sum())
    k = k or n_true_edges
    
    np.fill_diagonal(A_pred_np, 0)
    flat = A_pred_np.flatten()
    top_k_idx = np.argsort(flat)[-k:] if k > 0 else np.array([])
    
    pred_binary = np.zeros_like(flat, dtype=int)
    if len(top_k_idx) > 0:
        pred_binary[top_k_idx] = 1
    
    true_flat = A_true_np.flatten()
    
    TP = int(((pred_binary == 1) & (true_flat == 1)).sum())
    FP = int(((pred_binary == 1) & (true_flat == 0)).sum())
    FN = int(((pred_binary == 0) & (true_flat == 1)).sum())
    
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        "topk_f1": float(f1),
        "topk_precision": float(precision),
        "topk_recall": float(recall),
        "topk_shd": FP + FN,
        "topk_tp": TP,
        "topk_fp": FP,
        "topk_fn": FN,
        "k": k,
    }


def compute_best_f1(A_pred: torch.Tensor, A_true: torch.Tensor) -> Dict[str, float]:
    """Find best F1 over thresholds."""
    A_pred_np = A_pred.detach().cpu().numpy().copy()
    A_true_np = (A_true.cpu().numpy() > 0.5).astype(int)
    
    np.fill_diagonal(A_pred_np, 0)
    
    best_f1 = 0
    best_t = 0.1
    best_shd = float('inf')
    
    for t in np.arange(0.05, 0.55, 0.05):
        pred = (A_pred_np > t).astype(int)
        
        TP = int(((pred == 1) & (A_true_np == 1)).sum())
        FP = int(((pred == 1) & (A_true_np == 0)).sum())
        FN = int(((pred == 0) & (A_true_np == 1)).sum())
        
        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        shd = FP + FN
        
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
            best_shd = shd
    
    return {
        "best_f1": best_f1,
        "best_threshold": best_t,
        "best_shd": best_shd,
    }


def train_v4(
    data_root: str,
    output_dir: str = "artifacts/v4_experiments",
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: str = "cpu",
    lambda_causal: float = 0.2,
):
    """Train RC-GNN V4 model."""
    
    print("=" * 60)
    print("RC-GNN V4 Training (Causal Identifiability Priors)")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Data: {data_root}")
    print(f"Epochs: {epochs}, Batch: {batch_size}, LR: {lr}")
    print(f"Lambda causal: {lambda_causal}")
    print("=" * 60)
    
    # Load data
    data = load_data(data_root)
    X = normalize_data(data["X"])
    M = data["M"]
    e = data["e"]
    A_true = data["A_true"]
    config = data["config"]
    
    print(f"\n[Data] X shape: {X.shape}")
    print(f"[Data] M mean: {M.mean():.4f} (observed fraction)")
    
    regimes = np.unique(e)
    n_regimes = len(regimes)
    print(f"[Data] Regimes: {regimes.tolist()} (n={n_regimes})")
    
    if A_true is not None:
        n_true_edges = int(A_true.sum())
        print(f"[Data] A_true: {n_true_edges} edges")
    
    # Convert to tensors
    X_t = torch.from_numpy(X).float()
    M_t = torch.from_numpy(M).float()
    e_t = torch.from_numpy(e).long()
    A_true_t = torch.from_numpy(A_true).float() if A_true is not None else None
    
    # Train/val split
    n = X_t.shape[0]
    n_train = int(0.8 * n)
    perm = torch.randperm(n)
    train_idx, val_idx = perm[:n_train], perm[n_train:]
    
    train_loader = DataLoader(
        TensorDataset(X_t[train_idx], M_t[train_idx], e_t[train_idx]),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_t[val_idx], M_t[val_idx], e_t[val_idx]),
        batch_size=batch_size,
    )
    
    print(f"[Data] Train: {n_train}, Val: {n - n_train}")
    
    # Initialize model
    d = X_t.shape[-1]
    model = RCGNN(
        d=d,
        latent_dim=32,
        hidden_dim=64,
        n_regimes=n_regimes,
        target_edges=n_true_edges if A_true is not None else 13,
        lambda_causal=lambda_causal,
        use_env_specific_decoder=(n_regimes > 1),
    ).to(device)
    
    # Load intervention targets if available
    if config:
        model.load_intervention_targets_from_config(Path(data_root) / "config.json")
    
    print(f"[Model] V4 with d={d}, n_regimes={n_regimes}")
    print(f"[Model] Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"[Model] Env-specific decoder: {model.use_env_decoder}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    # Output directory
    output_path = Path(output_dir) / Path(data_root).name
    output_path.mkdir(parents=True, exist_ok=True)
    
    best_f1 = 0
    best_epoch = 0
    
    print("\n--- Training ---")
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        epoch_metrics = {}
        
        t_start = time.time()
        
        for X_batch, M_batch, e_batch in train_loader:
            X_batch = X_batch.to(device)
            M_batch = M_batch.to(device)
            e_batch = e_batch.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(X_batch, M_batch, e_batch)
            
            # Compute necessity every 10 epochs (expensive)
            compute_necessity = (epoch % 10 == 0) and (epoch > 10)
            
            loss, metrics = model.compute_loss(
                outputs, X_batch, M_batch, e_batch,
                epoch=epoch, total_epochs=epochs,
                compute_necessity=compute_necessity,
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            for k, v in metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0) + v
        
        scheduler.step()
        
        # Average metrics
        n_batches = len(train_loader)
        epoch_loss /= n_batches
        for k in epoch_metrics:
            epoch_metrics[k] /= n_batches
        
        t_elapsed = time.time() - t_start
        
        # Validation
        model.eval()
        with torch.no_grad():
            A_pred = model.graph_learner.get_mean_adjacency()
            
            if A_true_t is not None:
                topk = compute_topk_f1(A_pred, A_true_t)
                best = compute_best_f1(A_pred, A_true_t)
                
                # Correlation vs causation diagnostic
                diag = {}
                if epoch % 10 == 0:
                    X_sample = X_t[:1000].to(device)
                    diag = diagnose_correlation_vs_causation(
                        A_pred, A_true_t.to(device), X_sample
                    )
        
        # Log
        L_causal = epoch_metrics.get("L_causal_total", epoch_metrics.get("L_causal", 0))
        L_orient = epoch_metrics.get("L_orientation", 0)
        L_inv = epoch_metrics.get("L_inv", 0)
        
        print(f"Epoch {epoch:3d} | loss={epoch_loss:.4f} | "
              f"L_recon={epoch_metrics.get('L_recon', 0):.4f} "
              f"L_causal={L_causal:.4f} "
              f"L_orient={L_orient:.4f} "
              f"L_inv={L_inv:.2e} | "
              f"A_max={epoch_metrics.get('A_max', 0):.3f} | {t_elapsed:.1f}s")
        
        if A_true_t is not None:
            current_f1 = best["best_f1"]
            print(f"         | ★ Best-F1={current_f1:.4f} @t={best['best_threshold']:.2f} "
                  f"SHD={best['best_shd']} | "
                  f"TopK-F1={topk['topk_f1']:.4f} TP={topk['topk_tp']} FP={topk['topk_fp']}")
            
            if epoch % 10 == 0 and diag:
                print(f"         | 🔍 Diagnosis: {diag['diagnosis'].upper()} "
                      f"(pred-corr={diag['pred_corr_overlap']:.2f} "
                      f"pred-true={diag['pred_true_overlap']:.2f})")
            
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_epoch = epoch
                torch.save(model.state_dict(), output_path / "v4_best.pt")
                np.save(output_path / "A_best.npy", A_pred.cpu().numpy())
                print(f"         | 🏆 New best! → {output_path}/v4_best.pt")
    
    # Final summary
    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"Best F1: {best_f1:.4f} @ epoch {best_epoch}")
    
    # Final diagnostic
    if A_true_t is not None:
        model.eval()
        with torch.no_grad():
            A_pred = model.graph_learner.get_mean_adjacency()
            X_sample = X_t[:2000].to(device)
            final_diag = diagnose_correlation_vs_causation(
                A_pred, A_true_t.to(device), X_sample
            )
            
            print(f"\n=== CORRELATION VS CAUSATION ANALYSIS ===")
            print(f"Prediction overlaps with:")
            print(f"  - Correlation top-K: {final_diag['pred_corr_overlap']:.2%}")
            print(f"  - True causal edges: {final_diag['pred_true_overlap']:.2%}")
            print(f"  - (Baseline: corr overlaps true: {final_diag['corr_true_overlap']:.2%})")
            print(f"Average edge correlation:")
            print(f"  - Predicted edges: {final_diag['avg_pred_edge_corr']:.3f}")
            print(f"  - True causal edges: {final_diag['avg_true_edge_corr']:.3f}")
            print(f"DIAGNOSIS: Model is learning {final_diag['diagnosis'].upper()}")
    
    # Save final
    np.save(output_path / "A_final.npy", A_pred.cpu().numpy())
    
    return best_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RC-GNN V4")
    parser.add_argument("--data_root", type=str, required=True, help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="artifacts/v4_experiments")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--lambda_causal", type=float, default=0.2)
    
    args = parser.parse_args()
    
    train_v4(
        data_root=args.data_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        lambda_causal=args.lambda_causal,
    )
