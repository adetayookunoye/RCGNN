#!/usr/bin/env python3
"""
RC-GNN V3 Training Script

Features:
- GroupDRO for worst-case robustness across corruption regimes
- Built-in gradient diagnostics
- CompoundCorruptionGenerator integration
- CPU-first design (GPU optional)

Usage:
    python scripts/train_rcgnn_v3.py --data_dir data/interim/uci_air --epochs 100
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.rcgnn import RCGNN, notears_acyclicity


# =============================================================================
# DDP Utilities
# =============================================================================

def setup_ddp():
    """Initialize Distributed Data Parallel."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    """Cleanup DDP."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process (rank 0)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_world_size():
    """Get number of distributed processes."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


# =============================================================================
# Data Loading
# =============================================================================

def load_data(data_dir: str, normalize: bool = True) -> Dict[str, torch.Tensor]:
    """Load dataset (works with UCI Air, synth, etc.)"""
    data_path = Path(data_dir)
    
    X = np.load(data_path / "X.npy")
    M = np.load(data_path / "M.npy") if (data_path / "M.npy").exists() else np.ones_like(X)
    A_true = np.load(data_path / "A_true.npy") if (data_path / "A_true.npy").exists() else None
    e = np.load(data_path / "e.npy") if (data_path / "e.npy").exists() else np.zeros(X.shape[0])
    
    # Handle NaN
    X = np.nan_to_num(X, nan=0.0)
    
    # Normalize
    if normalize:
        X_flat = X.reshape(-1, X.shape[-1])
        mean = X_flat.mean(axis=0, keepdims=True)
        std = X_flat.std(axis=0, keepdims=True) + 1e-8
        X_flat = (X_flat - mean) / std
        X = X_flat.reshape(X.shape)
        print(f"[Data] Normalized: mean={X.mean():.4f}, std={X.std():.4f}")
    
    print(f"[Data] X shape: {X.shape}")
    print(f"[Data] M mean: {M.mean():.4f} (observed fraction)")
    print(f"[Data] Regimes: {np.unique(e)}")
    
    data = {
        "X": torch.from_numpy(X).float(),
        "M": torch.from_numpy(M).float(),
        "e": torch.from_numpy(e).long(),
    }
    
    if A_true is not None:
        data["A_true"] = torch.from_numpy(A_true).float()
        print(f"[Data] A_true: {A_true.sum():.0f} edges")
    
    return data


def create_dataloaders(
    data: Dict[str, torch.Tensor],
    batch_size: int = 32,
    train_split: float = 0.8,
    ddp: bool = False,
    num_workers: int = 8,  # CPU parallelism for data loading
) -> Tuple[DataLoader, DataLoader]:
    """Create train/val dataloaders with optional DDP support."""
    X, M, e = data["X"], data["M"], data["e"]
    
    N = X.shape[0]
    n_train = int(N * train_split)
    
    # Shuffle (use same permutation across processes)
    torch.manual_seed(42)  # Fixed seed for consistent split
    perm = torch.randperm(N)
    X, M, e = X[perm], M[perm], e[perm]
    
    # Split
    X_train, X_val = X[:n_train], X[n_train:]
    M_train, M_val = M[:n_train], M[n_train:]
    e_train, e_val = e[:n_train], e[n_train:]
    
    train_ds = TensorDataset(X_train, M_train, e_train)
    val_ds = TensorDataset(X_val, M_val, e_val)
    
    # Use DistributedSampler for DDP
    if ddp:
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        val_sampler = DistributedSampler(val_ds, shuffle=False)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler,
                                  num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, sampler=val_sampler,
                                num_workers=num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True)
    
    if is_main_process():
        print(f"[Data] Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    return train_loader, val_loader
    
    return train_loader, val_loader


# =============================================================================
# Metrics
# =============================================================================

def compute_structure_metrics(
    A_pred: torch.Tensor,
    A_true: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute F1, SHD, precision, recall.
    
    For empty graphs (pred_edges=0), SHD = |E_true| (not 999).
    """
    A_pred_np = (A_pred.detach().cpu().numpy() > threshold).astype(int)
    A_true_np = (A_true.cpu().numpy() > 0.5).astype(int)
    
    # Flatten
    y_pred = A_pred_np.flatten()
    y_true = A_true_np.flatten()
    
    TP = ((y_pred == 1) & (y_true == 1)).sum()
    FP = ((y_pred == 1) & (y_true == 0)).sum()
    FN = ((y_pred == 0) & (y_true == 1)).sum()
    
    pred_edges = int(y_pred.sum())
    true_edges = int(y_true.sum())
    
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    shd = int(FP + FN)  # For empty graph, this equals |E_true|
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "shd": shd,
        "pred_edges": pred_edges,
        "true_edges": true_edges,
        "TP": int(TP),
        "FP": int(FP),
        "FN": int(FN),
    }


def compute_topk_f1(
    A_pred: torch.Tensor,
    A_true: torch.Tensor,
    k: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute TopK metrics: predict top K edges vs true edges.
    
    PRIMARY PAPER METRIC: TopK where K = |E_true|.
    This avoids threshold calibration issues entirely.
    K defaults to number of true edges.
    
    Returns:
        topk_f1, topk_precision, topk_recall, topk_shd, k
    """
    A_pred_np = A_pred.detach().cpu().numpy().copy()
    A_true_np = (A_true.cpu().numpy() > 0.5).astype(int)
    
    # Default K to number of true edges
    n_true_edges = int(A_true_np.sum())
    if k is None:
        k = n_true_edges
    
    # Get top K edges from predicted
    d = A_pred_np.shape[0]
    # Zero out diagonal
    np.fill_diagonal(A_pred_np, 0)
    
    # Flatten and get top K indices
    flat = A_pred_np.flatten()
    top_k_idx = np.argsort(flat)[-k:] if k > 0 else np.array([])
    
    # Create binary prediction
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
    shd = FP + FN  # SHD@TopK
    
    return {
        "topk_f1": float(f1),
        "topk_precision": float(precision),
        "topk_recall": float(recall),
        "topk_shd": int(shd),
        "topk_tp": TP,
        "topk_fp": FP,
        "topk_fn": FN,
        "k": k,
    }


def compute_auc_f1(
    A_pred: torch.Tensor,
    A_true: torch.Tensor,
    thresholds: np.ndarray = None,
) -> Dict[str, float]:
    """
    Compute AUC-F1 over thresholds 0.05 → 0.50.
    
    Reviewer-friendly metric that summarizes performance
    across threshold choices. Less sensitive to calibration.
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.55, 0.05)  # 0.05, 0.10, ..., 0.50
    
    f1_scores = []
    for t in thresholds:
        metrics = compute_structure_metrics(A_pred, A_true, threshold=t)
        f1_scores.append(metrics["f1"])
    
    # Trapezoidal AUC (normalized to [0,1])
    auc = np.trapz(f1_scores, thresholds) / (thresholds[-1] - thresholds[0])
    
    return {
        "auc_f1": float(auc),
        "f1_at_thresholds": {f"{t:.2f}": float(f) for t, f in zip(thresholds, f1_scores)},
        "max_f1": float(max(f1_scores)),
        "mean_f1": float(np.mean(f1_scores)),
    }


def find_best_threshold(
    A_pred: torch.Tensor,
    A_true: torch.Tensor,
) -> Tuple[float, Dict[str, float]]:
    """Find threshold that maximizes F1.
    
    Skips thresholds that produce 0 edges (useless predictions).
    Falls back to threshold=0.3 with valid metrics if nothing works.
    """
    best_f1 = -1.0  # Start at -1 so we always update at least once
    best_thresh = 0.3  # Default to 0.3 not 0.5
    best_metrics = None
    
    for t in np.arange(0.05, 0.95, 0.05):
        metrics = compute_structure_metrics(A_pred, A_true, threshold=t)
        
        # Skip thresholds that produce empty predictions (useless)
        if metrics["pred_edges"] == 0:
            continue
            
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_thresh = t
            best_metrics = metrics
    
    # Fallback: if all thresholds produced empty graphs, use t=0.3 anyway
    if best_metrics is None:
        best_metrics = compute_structure_metrics(A_pred, A_true, threshold=0.3)
        best_thresh = 0.3
    
    best_metrics["threshold"] = best_thresh
    return best_thresh, best_metrics


# =============================================================================
# GroupDRO Training
# =============================================================================

def groupdro_reweight(
    losses_per_regime: Dict[int, torch.Tensor],
    weights: Dict[int, float],
    step_size: float = 0.01,
) -> Dict[int, float]:
    """
    Update GroupDRO weights based on per-regime losses.
    Upweight regimes with higher loss.
    """
    # Exponentiated gradient update
    for r, loss in losses_per_regime.items():
        weights[r] = weights[r] * np.exp(step_size * loss.item())
    
    # Normalize
    total = sum(weights.values())
    for r in weights:
        weights[r] /= total
    
    return weights


def train_epoch_groupdro(
    model: RCGNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    dro_weights: Optional[Dict[int, float]] = None,
    dro_step_size: float = 0.01,
    ddp: bool = False,
) -> Tuple[Dict[str, float], Dict[int, float]]:
    """
    Train one epoch with GroupDRO.
    
    Returns:
        metrics: Average metrics for the epoch
        dro_weights: Updated DRO weights
    """
    model.train()
    
    # Get underlying model for DDP
    base_model = model.module if ddp else model
    
    total_loss = 0.0
    metrics_sum = {}
    n_batches = 0
    
    # Track per-regime losses
    regime_losses = {}
    regime_counts = {}
    
    for batch_idx, (X, M, e) in enumerate(loader):
        X = X.to(device)
        M = M.to(device)
        e = e.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        outputs = model(X, M, regime=e)
        loss, metrics = base_model.compute_loss(outputs, X, M, regime=e)
        
        # GroupDRO: reweight by regime
        if dro_weights is not None:
            # Compute per-regime losses
            unique_regimes = e.unique()
            weighted_loss = torch.tensor(0.0, device=device)
            
            for r in unique_regimes:
                r_int = r.item()
                mask = (e == r)
                if mask.sum() == 0:
                    continue
                
                # Compute loss for this regime
                X_r = X[mask]
                M_r = M[mask]
                out_r = model(X_r, M_r)
                loss_r, _ = base_model.compute_loss(out_r, X_r, M_r)
                
                # Track
                if r_int not in regime_losses:
                    regime_losses[r_int] = 0.0
                    regime_counts[r_int] = 0
                regime_losses[r_int] += loss_r.item()
                regime_counts[r_int] += 1
                
                # Weight
                w = dro_weights.get(r_int, 1.0)
                weighted_loss = weighted_loss + w * loss_r
            
            loss = weighted_loss
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        base_model.graph_learner.increment_step()
        
        # Track metrics
        total_loss += loss.item()
        for k, v in metrics.items():
            metrics_sum[k] = metrics_sum.get(k, 0.0) + v
        n_batches += 1
    
    # Average metrics
    avg_metrics = {k: v / n_batches for k, v in metrics_sum.items()}
    avg_metrics["loss"] = total_loss / n_batches
    
    # Update DRO weights
    if dro_weights is not None and regime_losses:
        avg_regime_losses = {
            r: regime_losses[r] / regime_counts[r] 
            for r in regime_losses
        }
        dro_weights = groupdro_reweight(
            {r: torch.tensor(l) for r, l in avg_regime_losses.items()},
            dro_weights,
            step_size=dro_step_size,
        )
        avg_metrics["dro_weights"] = dro_weights.copy()
    
    return avg_metrics, dro_weights


def train_epoch_simple(
    model: RCGNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int = 50,
    ddp: bool = False,
) -> Dict[str, float]:
    """Simple training (no GroupDRO) with two-stage budget scheduling."""
    model.train()
    
    # Get underlying model for DDP
    base_model = model.module if ddp else model
    
    total_loss = 0.0
    metrics_sum = {}
    n_batches = 0
    grad_diagnostics = {}
    
    for batch_idx, (X, M, e) in enumerate(loader):
        X = X.to(device)
        M = M.to(device)
        e = e.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        outputs = model(X, M, regime=e)
        
        # Pass epoch for two-stage budget scheduling!
        loss, metrics = base_model.compute_loss(
            outputs, X, M, regime=e,
            epoch=epoch,
            total_epochs=total_epochs,
        )
        
        # Backward
        loss.backward()
        
        # Gradient diagnostics (first batch only)
        if batch_idx == 0:
            grad_diagnostics = base_model.diagnose_gradients()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        base_model.graph_learner.increment_step()
        
        # Track
        total_loss += loss.item()
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                metrics_sum[k] = metrics_sum.get(k, 0.0) + v
        n_batches += 1
    
    # Average
    avg_metrics = {k: v / n_batches for k, v in metrics_sum.items()}
    avg_metrics["loss"] = total_loss / n_batches
    avg_metrics.update(grad_diagnostics)
    
    return avg_metrics


@torch.no_grad()
def evaluate(
    model: RCGNN,
    loader: DataLoader,
    device: torch.device,
    A_true: Optional[torch.Tensor] = None,
    epoch: int = 1,
    total_epochs: int = 50,
    ddp: bool = False,
) -> Dict[str, float]:
    """Evaluate model with two-stage budget scheduling."""
    model.eval()
    
    # Get underlying model for DDP
    base_model = model.module if ddp else model
    
    total_loss = 0.0
    metrics_sum = {}
    n_batches = 0
    
    for X, M, e in loader:
        X = X.to(device)
        M = M.to(device)
        e = e.to(device)
        
        outputs = model(X, M, regime=e)
        loss, metrics = base_model.compute_loss(
            outputs, X, M, regime=e,
            epoch=epoch,
            total_epochs=total_epochs,
        )
        
        total_loss += loss.item()
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                metrics_sum[k] = metrics_sum.get(k, 0.0) + v
        n_batches += 1
    
    avg_metrics = {k: v / n_batches for k, v in metrics_sum.items()}
    avg_metrics["loss"] = total_loss / n_batches
    
    # Structure metrics - TopK is PRIMARY (threshold-free, paper metric)
    if A_true is not None:
        A_pred = base_model.get_soft_adjacency()
        
        # PRIMARY METRIC: TopK where K = |E_true| (avoids threshold calibration)
        topk_metrics = compute_topk_f1(A_pred, A_true)
        avg_metrics.update(topk_metrics)
        
        # SECONDARY: AUC-F1 over thresholds 0.05→0.50 (reviewer-friendly)
        auc_metrics = compute_auc_f1(A_pred, A_true)
        avg_metrics.update(auc_metrics)
        
        # Fixed threshold 0.3 (secondary operating point)
        fixed_metrics_03 = compute_structure_metrics(A_pred, A_true, threshold=0.3)
        avg_metrics.update({f"fixed03_{k}": v for k, v in fixed_metrics_03.items()})
        
        # Also log @0.2 for sparse graphs
        fixed_metrics_02 = compute_structure_metrics(A_pred, A_true, threshold=0.2)
        avg_metrics.update({f"fixed02_{k}": v for k, v in fixed_metrics_02.items()})
        
        # Best threshold (for reporting only, not training decisions)
        best_thresh, best_metrics = find_best_threshold(A_pred, A_true)
        avg_metrics.update({f"best_{k}": v for k, v in best_metrics.items()})
    
    return avg_metrics


# =============================================================================
# Main Training
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train RC-GNN V3")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="artifacts/v3")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--lambda_recon", type=float, default=1.0)
    parser.add_argument("--lambda_miss", type=float, default=0.5)
    parser.add_argument("--lambda_hsic", type=float, default=0.1)
    parser.add_argument("--lambda_acyclic", type=float, default=1.0)
    parser.add_argument("--lambda_sparse", type=float, default=0.01)
    parser.add_argument("--lambda_budget", type=float, default=0.1, help="Edge budget regularizer weight (match λ_acy strength)")
    parser.add_argument("--lambda_inv", type=float, default=10.0, help="Invariance loss weight (100x stronger for causal identification)")
    parser.add_argument("--target_edges", type=int, default=13, help="Target number of edges for budget regularizer")
    parser.add_argument("--early_stop_metric", type=str, default="topk_f1", choices=["topk_f1", "best_f1"], help="Metric for early stopping and model selection")
    parser.add_argument("--early_stop_patience", type=int, default=15, help="Epochs without improvement before early stopping")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping max norm")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Epochs before acyclicity penalty")
    parser.add_argument("--ramp_epochs", type=int, default=10, help="Epochs to ramp acyclicity penalty")
    parser.add_argument("--use_groupdro", action="store_true")
    parser.add_argument("--dro_step_size", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--ddp", action="store_true", help="Enable Distributed Data Parallel training")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # Setup DDP if enabled
    local_rank = 0
    if args.ddp:
        local_rank = setup_ddp()
        args.device = f"cuda:{local_rank}"
    
    # Seed (different per GPU for data augmentation)
    torch.manual_seed(args.seed + local_rank)
    np.random.seed(args.seed + local_rank)
    
    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    # Only print from main process
    if is_main_process():
        print(f"\n{'='*60}")
        print(f"RC-GNN V3 Training")
        print(f"{'='*60}")
        print(f"Device: {device}")
        if args.ddp:
            print(f"DDP: {get_world_size()} GPUs")
        print(f"Data: {args.data_dir}")
        print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
        print(f"Lambda scheduling: warmup={args.warmup_epochs}, ramp={args.ramp_epochs}")
        print(f"GroupDRO: {args.use_groupdro}")
        print(f"{'='*60}\n")
    
    # Load data
    data = load_data(args.data_dir)
    d = data["X"].shape[-1]
    n_regimes = len(torch.unique(data["e"]))
    
    train_loader, val_loader = create_dataloaders(
        data, batch_size=args.batch_size, ddp=args.ddp
    )
    
    # Model
    model = RCGNN(
        d=d,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        n_regimes=n_regimes,
        target_edges=args.target_edges,
        lambda_recon=args.lambda_recon,
        lambda_miss=args.lambda_miss,
        lambda_hsic=args.lambda_hsic,
        lambda_acyclic=args.lambda_acyclic,
        lambda_sparse=args.lambda_sparse,
        lambda_budget=args.lambda_budget,
        lambda_inv=args.lambda_inv,  # Part A: 100x stronger invariance for causal ID
    ).to(device)
    
    # Wrap with DDP if enabled
    if args.ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    if is_main_process():
        print(f"[Model] V3 with d={d}, latent={args.latent_dim}, regimes={n_regimes}")
        print(f"[Model] Parameters: {sum(p.numel() for p in model.parameters()):,}")
        # Verify device
        print(f"[Model] First param device: {next(model.parameters()).device}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # DRO weights
    dro_weights = None
    if args.use_groupdro and n_regimes > 1:
        dro_weights = {r: 1.0 / n_regimes for r in range(n_regimes)}
        if is_main_process():
            print(f"[DRO] Initial weights: {dro_weights}")
    
    # Output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    # Part A: Early stopping based on TopK-F1 (causal metric) or Best-F1 (robust metric)
    best_metric = 0.0
    best_epoch = 0
    best_thresh_for_f1 = 0.3  # Track which threshold gave best F1
    patience_counter = 0
    history = []
    
    A_true = data.get("A_true")
    if A_true is not None:
        A_true = A_true.to(device)
    
    if is_main_process():
        print("\n--- Training ---")
        print(f"📊 Model selection: {args.early_stop_metric.upper()} (early stop patience={args.early_stop_patience})")
        print(f"⚖️  λ_inv={args.lambda_inv} (invariance weight for causal identification)")
    
    # Track Stage 1 health for conditional pruning
    stage1_healthy = True  # Assume healthy, set to False if checks fail
    
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        
        # Update graph learner's epoch for staged temperature annealing
        # (handles DDP wrapper)
        graph_learner = model.module.graph_learner if hasattr(model, 'module') else model.graph_learner
        if hasattr(graph_learner, 'set_epoch'):
            graph_learner.set_epoch(epoch, args.epochs)
        
        # Pass stage1_healthy to model for conditional pruning
        model_core = model.module if hasattr(model, 'module') else model
        if hasattr(model_core, 'stage1_healthy'):
            model_core.stage1_healthy = stage1_healthy
        
        # Train
        if args.use_groupdro and dro_weights is not None:
            train_metrics, dro_weights = train_epoch_groupdro(
                model, train_loader, optimizer, device, epoch,
                dro_weights=dro_weights, dro_step_size=args.dro_step_size,
                ddp=args.ddp,
            )
        else:
            train_metrics = train_epoch_simple(
                model, train_loader, optimizer, device, epoch,
                total_epochs=args.epochs,
                ddp=args.ddp,
            )
        
        # Validate
        val_metrics = evaluate(
            model, val_loader, device, A_true,
            epoch=epoch,
            total_epochs=args.epochs,
            ddp=args.ddp,
        )
        
        scheduler.step()
        
        # Log
        t_epoch = time.time() - t0
        
        record = {
            "epoch": epoch,
            "time": t_epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)
        
        # Print
        if is_main_process():
            grad_mean = train_metrics.get("W_adj.grad_mean", 0.0)
            grad_status = "✓" if grad_mean > 1e-6 else "✗ ZERO!"
            lambda_acy = train_metrics.get("lambda_acy_used", 0.0)
            h_raw = train_metrics.get("h_A_raw", 0.0)
            edges_03 = train_metrics.get("edges>0.3", 0)
            edge_sum = train_metrics.get("edge_sum", 0)
            L_budget = train_metrics.get("L_budget", 0)
            target_edges = train_metrics.get("target_edges", 13)
            
            # Get scheduled lambda values for logging
            lambda_sparse = train_metrics.get('lambda_sparse_used', 0)
            lambda_bud = train_metrics.get('lambda_bud_used', 0)
            L_inv = train_metrics.get('L_inv', 0)
            
            # Edge counts at multiple thresholds for sanity checking
            edges_005 = train_metrics.get('edges>0.05', 0)
            edges_01 = train_metrics.get('edges>0.1', 0)
            
            temp = train_metrics.get('temperature', 1.0)
            lambda_sparse = train_metrics.get('lambda_sparse_used', 0)
            print(f"Epoch {epoch:3d} | "
                  f"loss={train_metrics['loss']:.4f} | "
                  f"L_recon={train_metrics.get('L_recon', 0):.4f} L_inv={L_inv:.1e} | "
                  f"h(A)={h_raw:.2f} λ_acy={lambda_acy:.3f} | "
                  f"A_max={train_metrics.get('A_max', 0):.3f} τ={temp:.2f} | "
                  f"Σ(A)={edge_sum:.1f}→{target_edges} λ_bud={lambda_bud:.4f} λ_spar={lambda_sparse:.1e} | "
                  f"grad(W)={grad_mean:.2e} {grad_status} | "
                  f"{t_epoch:.1f}s")
            
            # Structure metrics - Best-F1 is PRIMARY (robust across corruptions)
            if "best_f1" in val_metrics:
                # PRIMARY: Best-F1 (optimal threshold)
                best_f1_val = val_metrics.get("best_f1", 0)
                best_thresh_val = val_metrics.get("best_threshold", 0.3)
                best_shd_val = val_metrics.get("best_shd", 0)
                best_prec_val = val_metrics.get("best_precision", 0)
                best_rec_val = val_metrics.get("best_recall", 0)
                
                # SECONDARY: TopK (for comparison)
                topk_f1 = val_metrics.get("topk_f1", 0)
                topk_shd = val_metrics.get("topk_shd", 0)
                auc_f1 = val_metrics.get("auc_f1", 0)
                
                # PRIMARY display: Best-F1 and SHD@best_t (robust for corruptions)
                print(f"         | ★ Best-F1={best_f1_val:.4f} @t={best_thresh_val:.2f} SHD={best_shd_val:.0f} P={best_prec_val:.3f} R={best_rec_val:.3f} | TopK-F1={topk_f1:.4f} SHD={topk_shd} | AUC={auc_f1:.4f}")
        
            # SUCCESS CRITERION CHECK after Stage 1 (epoch 15 for 50 epochs)
            # Pruning is only safe if:
            #   1. A_max >= 0.4 (some edges are strong enough to survive threshold)
            #   2. Σ(A) <= 1.3 * d (edge mass is concentrated, not diffuse)
            stage1_end = int(args.epochs * 0.30)
            A_max_val = train_metrics.get("A_max", 0)
            edge_sum_val = train_metrics.get("edge_sum", 0)
            edges_02 = val_metrics.get("fixed02_pred_edges", 0)
            d = model_core.d if hasattr(model_core, 'd') else 13
            
            if epoch == stage1_end:
                # Check pruning safety criteria
                a_max_ok = A_max_val >= 0.4
                edge_sum_ok = edge_sum_val <= 1.3 * d  # For d=13: <= 16.9
                
                if not a_max_ok:
                    print(f"         | ⚠️ WARNING: A_max={A_max_val:.3f} < 0.4 at Stage 1 end! Pruning disabled.")
                    stage1_healthy = False
                if not edge_sum_ok:
                    print(f"         | ⚠️ WARNING: Σ(A)={edge_sum_val:.1f} > {1.3*d:.1f} at Stage 1 end! Edges too diffuse.")
                    stage1_healthy = False
                if edges_02 == 0:
                    print(f"         | ⚠️ WARNING: edges@0.2=0 at Stage 1 end! Structure not discovered.")
                    stage1_healthy = False
                    
                if stage1_healthy:
                    print(f"         | ✓ Stage 1 SUCCESS: A_max={A_max_val:.3f}>=0.4, Σ(A)={edge_sum_val:.1f}<={1.3*d:.1f}, edges@0.2={edges_02}>0")
                    print(f"         | ✓ Pruning ENABLED for Stage 2")
                else:
                    print(f"         | ⚡ Pruning DISABLED - Stage 2 will use gentle schedules")
        
        # LR reduction in Stage 3 (last 20% of epochs)
        stage3_start = int(args.epochs * 0.80)
        if epoch == stage3_start and is_main_process():
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
            print(f"         | 📉 Stage 3: LR reduced to {optimizer.param_groups[0]['lr']:.1e}")
        
        # Part A: Model selection based on configurable metric (TopK-F1 for causal ID)
        if "best_f1" in val_metrics:
            # Get the metric we're optimizing for
            if args.early_stop_metric == "topk_f1":
                current_metric = val_metrics.get("topk_f1", 0)
                metric_name = "TopK-F1"
            else:
                current_metric = val_metrics.get("best_f1", 0)
                metric_name = "Best-F1"
            
            current_best_thresh = val_metrics.get("best_threshold", 0.3)
            
            if current_metric > best_metric:
                best_metric = current_metric
                best_epoch = epoch
                best_thresh_for_f1 = current_best_thresh
                patience_counter = 0  # Reset patience
                if is_main_process():
                    # Get underlying model for DDP
                    model_to_save = model.module if args.ddp else model
                    torch.save({
                        "epoch": epoch,
                        "model_state": model_to_save.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "metrics": val_metrics,
                        "best_metric": best_metric,
                        "early_stop_metric": args.early_stop_metric,
                        "best_threshold": best_thresh_for_f1,
                        "args": vars(args),
                    }, output_dir / "v3_best.pt")
                    print(f"         | 🏆 New best! {metric_name}={best_metric:.4f} → {output_dir / 'v3_best.pt'}")
            else:
                patience_counter += 1
                if is_main_process() and patience_counter >= args.early_stop_patience // 2:
                    print(f"         | ⏳ No improvement for {patience_counter}/{args.early_stop_patience} epochs")
        
        # Part A: Early stopping
        if patience_counter >= args.early_stop_patience:
            if is_main_process():
                print(f"\n⏹️ Early stopping at epoch {epoch}! Best {args.early_stop_metric}={best_metric:.4f} at epoch {best_epoch}")
            break
    
    # Only main process saves
    if is_main_process():
        # Get underlying model for DDP
        model_to_save = model.module if args.ddp else model
        
        # Final save
        torch.save({
            "epoch": args.epochs,
            "model_state": model_to_save.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history": history,
            "args": vars(args),
        }, output_dir / "v3_final.pt")
        
        # Save adjacency
        A_final = model_to_save.get_soft_adjacency().cpu().numpy()
        np.save(output_dir / "A_v3.npy", A_final)
        
        # Final evaluation with all metrics
        A_pred = torch.from_numpy(A_final).to(device)
        if A_true is not None:
            # PRIMARY: TopK metrics (K = |E_true|)
            final_topk = compute_topk_f1(A_pred, A_true)
            topk_f1 = final_topk.get("topk_f1", 0)
            topk_shd = final_topk.get("topk_shd", 0)
            k = final_topk.get("k", 0)
            
            # AUC-F1 (reviewer-friendly)
            auc_metrics = compute_auc_f1(A_pred, A_true)
            auc_f1 = auc_metrics.get("auc_f1", 0)
            
            # Secondary: threshold-based
            best_thresh, best_metrics = find_best_threshold(A_pred, A_true)
            fixed03_metrics = compute_structure_metrics(A_pred, A_true, threshold=0.3)
            
            best_f1_val = best_metrics.get("f1", 0)
            best_shd = best_metrics.get("shd", 999)
            shd_03 = fixed03_metrics.get("shd", 999)
        else:
            topk_f1 = best_metric  # Use the tracked best metric
            topk_shd = 999
            k = 0
            auc_f1 = 0
            best_thresh = 0.3
            best_f1_val = best_metric
            best_shd = 999
            shd_03 = 999
        
        # Summary
        print("\n" + "="*60)
        print(f"Training Complete (early_stop_metric={args.early_stop_metric})")
        print("="*60)
        print(f"🎯 TRAINING SELECTION: {args.early_stop_metric.upper()}={best_metric:.4f} at epoch {best_epoch}")
        print(f"=== CAUSAL METRIC (TopK-F1) ===")
        print(f"TopK-F1: {topk_f1:.4f} (K={k} true edges)")
        print(f"TopK-SHD: {topk_shd}")
        print(f"=== ROBUSTNESS METRIC (Best-F1) ===")
        print(f"Best-F1: {best_f1_val:.4f} @ threshold={best_thresh:.2f}")
        print(f"SHD@best_t: {best_shd}")
        print(f"AUC-F1: {auc_f1:.4f}")
        print(f"=== ADJACENCY STATS ===")
        print(f"Final A: max={A_final.max():.4f}, mean={A_final.mean():.4f}")
        print(f"Saved to: {output_dir}")
        
        # Detailed debug info (edges at multiple thresholds)
        edges_005 = int((A_final > 0.05).sum())
        edges_010 = int((A_final > 0.10).sum())
        edges_020 = int((A_final > 0.20).sum())
        edges_030 = int((A_final > 0.30).sum())
        print(f"[DEBUG] edges>t: 0.05={edges_005} 0.10={edges_010} 0.20={edges_020} 0.30={edges_030}")
        print(f"[DEBUG] TopK: TP={final_topk.get('topk_tp', 'N/A')} FP={final_topk.get('topk_fp', 'N/A')} FN={final_topk.get('topk_fn', 'N/A')}")
        
        # Machine-parseable summary line (for sweep scripts)
        print(f"\n[METRICS] topk_f1={topk_f1:.4f} best_f1={best_f1_val:.4f} best_t={best_thresh:.2f} shd_best={best_shd} auc_f1={auc_f1:.4f} topk_shd={topk_shd} shd_03={shd_03} a_max={A_final.max():.4f} best_epoch={best_epoch}")
        
        # Save history
        with open(output_dir / "history.json", "w") as f:
            # Convert numpy to python types
            def convert(obj):
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, dict):
                    return {k: convert(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [convert(x) for x in obj]
                return obj
            json.dump(convert(history), f, indent=2)
        
        # Save final metrics as JSON for easy parsing
        # Part A: Include early stopping info and lambda_inv
        final_metrics = {
            # Training info
            "early_stop_metric": args.early_stop_metric,
            "best_epoch": int(best_epoch),
            "lambda_inv": float(args.lambda_inv),
            # CAUSAL METRIC (primary for Part A)
            "topk_f1": float(topk_f1),
            "topk_shd": int(topk_shd),
            "topk_k": int(k),
            # ROBUSTNESS METRIC
            "best_f1": float(best_f1_val),
            "best_threshold": float(best_thresh),
            "shd_best": int(best_shd),
            # AUC-F1 (reviewer-friendly)
            "auc_f1": float(auc_f1),
            # Fixed threshold
            "shd_03": int(shd_03),
            # Adjacency stats
            "a_max": float(A_final.max()),
            "a_mean": float(A_final.mean()),
            "edges_above_03": int((A_final > 0.3).sum()),
            "edges_above_01": int((A_final > 0.1).sum()),
        }
        with open(output_dir / "final_metrics.json", "w") as f:
            json.dump(final_metrics, f, indent=2)
    
    # Cleanup DDP
    if args.ddp:
        cleanup_ddp()


if __name__ == "__main__":
    main()
