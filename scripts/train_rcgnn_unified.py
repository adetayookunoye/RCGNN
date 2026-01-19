#!/usr/bin/env python3
"""
RC-GNN Unified Training Script

Consolidates best practices from all training scripts:
- train_rcgnn_v3.py: DDP, GroupDRO, 3-stage scheduling
- train_rcgnn_publication.py: 7 documented fixes
- train_stable.py: Gradient stability, LR scheduling
- train_rcgnn_v4.py: Causal priors, correlation vs causation diagnostics
- train_corruption_sweep.py: Sweep mode for ablation

Features:
1. Multi-GPU (DDP) with single-GPU/CPU fallback
2. GroupDRO for worst-case robustness across regimes
3. 3-Stage Training: discovery → pruning → refinement
4. Publication-quality fixes (temperature, loss rebalancing)
5. Gradient stability (aggressive clipping, LR scheduling)
6. Causal diagnostics (correlation vs causation)
7. Comprehensive metrics (TopK-F1, Best-F1, AUC-F1)
8. Sweep mode for ablation studies
9. Full CLI with sensible defaults

Usage:
    # Basic training
    python scripts/train_rcgnn_unified.py --data_dir data/interim/uci_air --epochs 100
    
    # Multi-GPU with DDP
    torchrun --nproc_per_node=4 scripts/train_rcgnn_unified.py --ddp --data_dir data/interim/uci_air
    
    # With GroupDRO
    python scripts/train_rcgnn_unified.py --data_dir data/interim/uci_air --use_groupdro
    
    # Sweep mode (for ablation)
    python scripts/train_rcgnn_unified.py --data_dir data/interim/uci_air --seed 42 --sweep_mode

Author: RC-GNN Team
"""

import os
import sys
import argparse
import json
import time
import math
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.rcgnn import RCGNN, notears_acyclicity

# Optional DDP imports (graceful fallback)
try:
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler
    import torch.distributed as dist
    DDP_AVAILABLE = True
except ImportError:
    DDP_AVAILABLE = False


# =============================================================================
# Configuration Defaults (Publication-Quality)
# =============================================================================

DEFAULT_CONFIG = {
    # Training
    "epochs": 100,
    "batch_size": 32,
    "patience": 20,
    "eval_frequency": 2,
    
    # Learning rate (with warm restarts - FIX 6 from publication)
    "lr": 5e-4,
    "lr_min": 1e-5,
    "weight_decay": 1e-4,
    "warmup_epochs": 5,
    "restart_every": 50,
    "grad_clip": 1.0,
    
    # Model architecture
    "latent_dim": 32,
    "hidden_dim": 64,
    
    # Loss weights (FIX 2 from publication: rebalanced)
    "lambda_recon": 100.0,      # Strong reconstruction signal
    "lambda_miss": 0.5,         # Missingness prediction
    "lambda_hsic": 0.1,         # Disentanglement
    "lambda_sparse": 1e-4,      # Sparsity
    "lambda_acyclic": 0.05,     # Acyclicity (delayed)
    "lambda_budget": 0.05,      # Edge budget
    "lambda_inv": 1.0,          # Invariance
    "lambda_causal": 0.2,       # Causal priors (V4)
    "lambda_var_penalty": 0.01, # MNAR variance penalty
    
    # Structure learning (FIX 1 + FIX 5 from publication)
    "temperature_fixed": 1.0,   # No annealing
    "sparsify_method": "topk",  # or "sigmoid" for continuous
    "topk_ratio": 0.15,
    "target_edges": 13,         # Adjust per dataset
    
    # 3-Stage scheduling (from V3)
    "stage1_end": 0.30,         # Discovery phase ends
    "stage2_end": 0.80,         # Pruning phase ends
    
    # GroupDRO (from V3)
    "dro_step_size": 0.01,
    
    # Evaluation
    "threshold_grid": list(np.arange(0.05, 0.55, 0.05)),
    
    # System
    "device": "auto",
    "seed": 42,
    "num_workers": 4,
}


# =============================================================================
# DDP Utilities
# =============================================================================

def setup_ddp():
    """Initialize Distributed Data Parallel."""
    if not DDP_AVAILABLE:
        raise RuntimeError("DDP not available. Install torch with distributed support.")
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    """Cleanup DDP."""
    if DDP_AVAILABLE and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process (rank 0)."""
    if not DDP_AVAILABLE or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_world_size():
    """Get number of distributed processes."""
    if not DDP_AVAILABLE or not dist.is_initialized():
        return 1
    return dist.get_world_size()


# =============================================================================
# Data Loading
# =============================================================================

def load_data(data_dir: str, normalize: bool = True) -> Dict[str, torch.Tensor]:
    """
    Load dataset from numpy files.
    
    Expected files:
        - X.npy: Data tensor [N, T, d] or [N, d]
        - M.npy: Missingness mask (optional, defaults to all observed)
        - e.npy: Regime labels (optional, defaults to single regime)
        - A_true.npy: Ground truth adjacency (optional, for evaluation)
        - config.json: Dataset config (optional, for intervention targets)
    """
    data_path = Path(data_dir)
    
    # Load required data
    X = np.load(data_path / "X.npy")
    
    # Load optional data with defaults
    M = np.load(data_path / "M.npy") if (data_path / "M.npy").exists() else np.ones_like(X)
    e = np.load(data_path / "e.npy") if (data_path / "e.npy").exists() else np.zeros(X.shape[0], dtype=np.int64)
    A_true = np.load(data_path / "A_true.npy") if (data_path / "A_true.npy").exists() else None
    
    # Load config if available
    config = None
    if (data_path / "config.json").exists():
        with open(data_path / "config.json") as f:
            config = json.load(f)
    
    # Handle NaN
    X = np.nan_to_num(X, nan=0.0)
    
    # Normalize (per-feature z-score)
    if normalize:
        if X.ndim == 3:
            # [N, T, d] -> flatten to [N*T, d]
            shape = X.shape
            X_flat = X.reshape(-1, X.shape[-1])
            mean = X_flat.mean(axis=0, keepdims=True)
            std = X_flat.std(axis=0, keepdims=True) + 1e-8
            X_flat = (X_flat - mean) / std
            X = X_flat.reshape(shape)
        else:
            mean = X.mean(axis=0, keepdims=True)
            std = X.std(axis=0, keepdims=True) + 1e-8
            X = (X - mean) / std
    
    data = {
        "X": torch.from_numpy(X).float(),
        "M": torch.from_numpy(M).float(),
        "e": torch.from_numpy(e).long(),
        "config": config,
    }
    
    if A_true is not None:
        data["A_true"] = torch.from_numpy(A_true).float()
    
    return data


def create_dataloaders(
    data: Dict[str, torch.Tensor],
    batch_size: int = 32,
    train_split: float = 0.8,
    ddp: bool = False,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Create train/val dataloaders with optional DDP support."""
    X, M, e = data["X"], data["M"], data["e"]
    
    N = X.shape[0]
    n_train = int(N * train_split)
    
    # Deterministic shuffle
    torch.manual_seed(seed)
    perm = torch.randperm(N)
    X, M, e = X[perm], M[perm], e[perm]
    
    # Split
    X_train, X_val = X[:n_train], X[n_train:]
    M_train, M_val = M[:n_train], M[n_train:]
    e_train, e_val = e[:n_train], e[n_train:]
    
    train_ds = TensorDataset(X_train, M_train, e_train)
    val_ds = TensorDataset(X_val, M_val, e_val)
    
    # Use DistributedSampler for DDP
    if ddp and DDP_AVAILABLE:
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        val_sampler = DistributedSampler(val_ds, shuffle=False)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, sampler=val_sampler,
            num_workers=num_workers, pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=torch.cuda.is_available()
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=torch.cuda.is_available()
        )
    
    return train_loader, val_loader


# =============================================================================
# Metrics
# =============================================================================

def compute_structure_metrics(
    A_pred: torch.Tensor,
    A_true: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute F1, SHD, precision, recall at fixed threshold."""
    A_pred_np = (A_pred.detach().cpu().numpy() > threshold).astype(int)
    A_true_np = (A_true.cpu().numpy() > 0.5).astype(int)
    
    # Zero diagonal
    np.fill_diagonal(A_pred_np, 0)
    np.fill_diagonal(A_true_np, 0)
    
    y_pred = A_pred_np.flatten()
    y_true = A_true_np.flatten()
    
    TP = ((y_pred == 1) & (y_true == 1)).sum()
    FP = ((y_pred == 1) & (y_true == 0)).sum()
    FN = ((y_pred == 0) & (y_true == 1)).sum()
    
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    shd = int(FP + FN)
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "shd": shd,
        "pred_edges": int(y_pred.sum()),
        "true_edges": int(y_true.sum()),
        "TP": int(TP),
        "FP": int(FP),
        "FN": int(FN),
        "threshold": threshold,
    }


def compute_topk_f1(
    A_pred: torch.Tensor,
    A_true: torch.Tensor,
    k: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute TopK metrics: predict top K edges vs true edges.
    K defaults to number of true edges (fair comparison).
    """
    A_pred_np = A_pred.detach().cpu().numpy().copy()
    A_true_np = (A_true.cpu().numpy() > 0.5).astype(int)
    
    # Zero diagonal
    np.fill_diagonal(A_pred_np, 0)
    np.fill_diagonal(A_true_np, 0)
    
    n_true_edges = int(A_true_np.sum())
    if k is None:
        k = n_true_edges
    
    # Get top K edges
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


def compute_best_f1(
    A_pred: torch.Tensor,
    A_true: torch.Tensor,
    thresholds: List[float] = None,
) -> Dict[str, float]:
    """Find best F1 over thresholds."""
    if thresholds is None:
        thresholds = list(np.arange(0.05, 0.55, 0.05))
    
    best_f1 = 0
    best_t = 0.1
    best_metrics = {}
    
    for t in thresholds:
        metrics = compute_structure_metrics(A_pred, A_true, threshold=t)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_t = t
            best_metrics = metrics
    
    return {
        "best_f1": best_f1,
        "best_threshold": best_t,
        "best_shd": best_metrics.get("shd", 0),
        "best_precision": best_metrics.get("precision", 0),
        "best_recall": best_metrics.get("recall", 0),
    }


def compute_auc_f1(
    A_pred: torch.Tensor,
    A_true: torch.Tensor,
    thresholds: List[float] = None,
) -> Dict[str, float]:
    """Compute AUC-F1 over thresholds (robustness metric)."""
    if thresholds is None:
        thresholds = list(np.arange(0.05, 0.55, 0.05))
    
    f1_scores = []
    for t in thresholds:
        metrics = compute_structure_metrics(A_pred, A_true, threshold=t)
        f1_scores.append(metrics["f1"])
    
    # Trapezoidal AUC
    auc = np.trapz(f1_scores, thresholds) / (thresholds[-1] - thresholds[0])
    
    return {
        "auc_f1": float(auc),
        "max_f1": float(max(f1_scores)),
        "mean_f1": float(np.mean(f1_scores)),
    }


def diagnose_correlation_vs_causation(
    A_pred: torch.Tensor,
    A_true: torch.Tensor,
    X: torch.Tensor,
) -> Dict[str, float]:
    """
    Diagnose whether model learns causation or correlation.
    From V4 causal priors.
    """
    A_pred_np = A_pred.detach().cpu().numpy().copy()
    A_true_np = A_true.cpu().numpy()
    X_np = X.cpu().numpy()
    
    # Compute correlation matrix
    if X_np.ndim == 3:
        X_flat = X_np.reshape(-1, X_np.shape[-1])
    else:
        X_flat = X_np
    corr = np.corrcoef(X_flat.T)
    corr = np.abs(corr)
    np.fill_diagonal(corr, 0)
    
    # Zero diagonals
    np.fill_diagonal(A_pred_np, 0)
    np.fill_diagonal(A_true_np, 0)
    
    # Get top K for each (K = true edges)
    k = int(A_true_np.sum())
    
    def top_k_indices(arr, k):
        flat = arr.flatten()
        return set(np.argsort(flat)[-k:].tolist()) if k > 0 else set()
    
    pred_topk = top_k_indices(A_pred_np, k)
    corr_topk = top_k_indices(corr, k)
    true_edges = set(np.argwhere(A_true_np.flatten() > 0.5).flatten().tolist())
    
    # Overlaps
    pred_corr_overlap = len(pred_topk & corr_topk) / max(len(pred_topk), 1)
    pred_true_overlap = len(pred_topk & true_edges) / max(len(pred_topk), 1)
    corr_true_overlap = len(corr_topk & true_edges) / max(len(corr_topk), 1)
    
    # Average edge correlations
    def avg_edge_corr(edge_set):
        d = corr.shape[0]
        vals = [corr.flatten()[i] for i in edge_set if i < d*d]
        return np.mean(vals) if vals else 0.0
    
    avg_pred_edge_corr = avg_edge_corr(pred_topk)
    avg_true_edge_corr = avg_edge_corr(true_edges)
    
    # Diagnosis
    if pred_true_overlap > pred_corr_overlap + 0.1:
        diagnosis = "causation"
    elif pred_corr_overlap > pred_true_overlap + 0.1:
        diagnosis = "correlation"
    else:
        diagnosis = "mixed"
    
    return {
        "pred_corr_overlap": pred_corr_overlap,
        "pred_true_overlap": pred_true_overlap,
        "corr_true_overlap": corr_true_overlap,
        "avg_pred_edge_corr": avg_pred_edge_corr,
        "avg_true_edge_corr": avg_true_edge_corr,
        "diagnosis": diagnosis,
    }


# =============================================================================
# GroupDRO
# =============================================================================

def groupdro_reweight(
    regime_losses: Dict[int, torch.Tensor],
    weights: Dict[int, float],
    step_size: float = 0.01,
) -> Dict[int, float]:
    """Update GroupDRO weights based on regime losses."""
    new_weights = {}
    total = 0.0
    
    for r, loss in regime_losses.items():
        w = weights.get(r, 1.0)
        # Exponentiated gradient ascent
        w_new = w * torch.exp(step_size * loss.detach()).item()
        new_weights[r] = w_new
        total += w_new
    
    # Normalize
    for r in new_weights:
        new_weights[r] /= (total + 1e-8)
    
    return new_weights


# =============================================================================
# Training Loop
# =============================================================================

def train_epoch(
    model: RCGNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    config: Dict,
    ddp: bool = False,
    dro_weights: Optional[Dict[int, float]] = None,
) -> Tuple[Dict[str, float], Optional[Dict[int, float]]]:
    """Train one epoch with all bells and whistles."""
    model.train()
    base_model = model.module if ddp else model
    
    total_loss = 0.0
    metrics_sum = {}
    n_batches = 0
    
    # For GroupDRO
    regime_losses = {}
    regime_counts = {}
    
    for batch_idx, (X, M, e) in enumerate(loader):
        X = X.to(device)
        M = M.to(device)
        e = e.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        outputs = model(X, M, regime=e)
        
        # Compute loss with epoch info for scheduling
        loss, batch_metrics = base_model.compute_loss(
            outputs, X, M, regime=e,
            epoch=epoch,
            total_epochs=total_epochs,
        )
        
        # GroupDRO: reweight by regime
        if dro_weights is not None:
            unique_regimes = torch.unique(e)
            weighted_loss = torch.tensor(0.0, device=device)
            
            for r in unique_regimes:
                r_int = r.item()
                mask = (e == r)
                if mask.sum() == 0:
                    continue
                
                # Track regime loss
                if r_int not in regime_losses:
                    regime_losses[r_int] = 0.0
                    regime_counts[r_int] = 0
                regime_losses[r_int] += loss.item()
                regime_counts[r_int] += 1
                
                # Weight
                w = dro_weights.get(r_int, 1.0)
                weighted_loss = weighted_loss + w * loss
            
            loss = weighted_loss
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=config["grad_clip"]
        )
        
        optimizer.step()
        base_model.graph_learner.increment_step()
        
        # Track metrics
        total_loss += loss.item()
        for k, v in batch_metrics.items():
            if isinstance(v, (int, float)):
                metrics_sum[k] = metrics_sum.get(k, 0.0) + v
        metrics_sum["grad_norm"] = metrics_sum.get("grad_norm", 0.0) + grad_norm.item()
        n_batches += 1
    
    # Average metrics
    avg_metrics = {k: v / n_batches for k, v in metrics_sum.items()}
    avg_metrics["loss"] = total_loss / n_batches
    
    # Update DRO weights
    if dro_weights is not None and regime_losses:
        avg_regime_losses = {
            r: torch.tensor(regime_losses[r] / regime_counts[r])
            for r in regime_losses
        }
        dro_weights = groupdro_reweight(
            avg_regime_losses, dro_weights, step_size=config["dro_step_size"]
        )
    
    return avg_metrics, dro_weights


def validate(
    model: RCGNN,
    loader: DataLoader,
    A_true: Optional[torch.Tensor],
    X_all: torch.Tensor,
    device: torch.device,
    config: Dict,
    ddp: bool = False,
) -> Dict[str, float]:
    """Validation with comprehensive metrics."""
    model.eval()
    base_model = model.module if ddp else model
    
    with torch.no_grad():
        # Get predicted adjacency
        A_pred = base_model.graph_learner.get_mean_adjacency()
        
        metrics = {
            "A_mean": A_pred.mean().item(),
            "A_max": A_pred.max().item(),
            "A_min": A_pred.min().item(),
            "temperature": base_model.graph_learner.get_temperature(),
        }
        
        if A_true is not None:
            # TopK-F1 (primary metric)
            topk = compute_topk_f1(A_pred, A_true)
            metrics.update(topk)
            
            # Best-F1 (optimal threshold)
            best = compute_best_f1(A_pred, A_true, config["threshold_grid"])
            metrics.update(best)
            
            # AUC-F1 (robustness)
            auc = compute_auc_f1(A_pred, A_true, config["threshold_grid"])
            metrics.update(auc)
            
            # Fixed threshold metrics
            fixed = compute_structure_metrics(A_pred, A_true, threshold=0.2)
            metrics["fixed02_f1"] = fixed["f1"]
            metrics["fixed02_pred_edges"] = fixed["pred_edges"]
        
        # Correlation vs causation diagnosis (every 10 epochs)
        # Done in main loop
        
    return metrics


# =============================================================================
# Main Training Function
# =============================================================================

def train(
    data_dir: str,
    output_dir: str = "artifacts/unified",
    config: Dict = None,
    ddp: bool = False,
    use_groupdro: bool = False,
    sweep_mode: bool = False,
) -> Dict[str, float]:
    """
    Main training function with all features.
    
    Args:
        data_dir: Path to data directory
        output_dir: Path to output directory
        config: Configuration dict (merged with defaults)
        ddp: Enable Distributed Data Parallel
        use_groupdro: Enable GroupDRO
        sweep_mode: Enable sweep mode (minimal output)
    
    Returns:
        Dictionary of final metrics
    """
    # Merge config with defaults
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)
    
    # Setup DDP
    local_rank = 0
    if ddp:
        local_rank = setup_ddp()
        cfg["device"] = f"cuda:{local_rank}"
    
    # Device
    if cfg["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg["device"])
    
    # Seed
    torch.manual_seed(cfg["seed"] + local_rank)
    np.random.seed(cfg["seed"] + local_rank)
    
    # Load data
    data = load_data(data_dir, normalize=True)
    d = data["X"].shape[-1]
    n_regimes = len(torch.unique(data["e"]))
    A_true = data.get("A_true", None)
    
    # Adjust target edges if A_true available
    if A_true is not None and cfg["target_edges"] == 13:
        cfg["target_edges"] = int(A_true.sum())
    
    train_loader, val_loader = create_dataloaders(
        data,
        batch_size=cfg["batch_size"],
        ddp=ddp,
        num_workers=cfg["num_workers"],
        seed=cfg["seed"],
    )
    
    # Print info
    if is_main_process() and not sweep_mode:
        print("\n" + "=" * 70)
        print("RC-GNN UNIFIED TRAINING")
        print("=" * 70)
        print(f"Device: {device}")
        if ddp:
            print(f"DDP: {get_world_size()} GPUs")
        print(f"Data: {data_dir}")
        print(f"  X shape: {data['X'].shape}")
        print(f"  Regimes: {n_regimes}")
        print(f"  True edges: {int(A_true.sum()) if A_true is not None else 'N/A'}")
        print(f"Epochs: {cfg['epochs']}, Batch: {cfg['batch_size']}, LR: {cfg['lr']}")
        print(f"GroupDRO: {use_groupdro}")
        print(f"3-Stage: discovery→{int(cfg['stage1_end']*100)}%, "
              f"pruning→{int(cfg['stage2_end']*100)}%, refinement→100%")
        print("=" * 70 + "\n")
    
    # Create model
    model = RCGNN(
        d=d,
        latent_dim=cfg["latent_dim"],
        hidden_dim=cfg["hidden_dim"],
        n_regimes=n_regimes,
        target_edges=cfg["target_edges"],
        lambda_recon=cfg["lambda_recon"],
        lambda_miss=cfg["lambda_miss"],
        lambda_hsic=cfg["lambda_hsic"],
        lambda_sparse=cfg["lambda_sparse"],
        lambda_inv=cfg["lambda_inv"],
        lambda_causal=cfg["lambda_causal"],
        lambda_var_penalty=cfg["lambda_var_penalty"],
    ).to(device)
    
    # Wrap with DDP
    if ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    base_model = model.module if ddp else model
    
    if is_main_process() and not sweep_mode:
        print(f"[Model] d={d}, latent={cfg['latent_dim']}, regimes={n_regimes}")
        print(f"[Model] Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer with warm restarts
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"]
    )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cfg["restart_every"],
        T_mult=1,
        eta_min=cfg["lr_min"]
    )
    
    # GroupDRO weights
    dro_weights = None
    if use_groupdro and n_regimes > 1:
        dro_weights = {r: 1.0 / n_regimes for r in range(n_regimes)}
    
    # Output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Training tracking
    best_metric = 0
    best_epoch = 0
    patience_counter = 0
    history = []
    stage1_healthy = True
    
    # Training loop
    if is_main_process() and not sweep_mode:
        print("--- Training ---\n")
    
    for epoch in range(1, cfg["epochs"] + 1):
        t_start = time.time()
        
        # Set sampler epoch for DDP
        if ddp and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        
        # Train epoch
        train_metrics, dro_weights = train_epoch(
            model, train_loader, optimizer, device,
            epoch, cfg["epochs"], cfg,
            ddp=ddp, dro_weights=dro_weights,
        )
        
        scheduler.step()
        t_elapsed = time.time() - t_start
        
        # Validate
        val_metrics = {}
        if epoch % cfg["eval_frequency"] == 0 or epoch == cfg["epochs"]:
            val_metrics = validate(
                model, val_loader, A_true, data["X"], device, cfg, ddp=ddp
            )
            
            # Correlation vs causation diagnosis (every 10 epochs)
            if epoch % 10 == 0 and A_true is not None:
                X_sample = data["X"][:min(2000, len(data["X"]))].to(device)
                diag = diagnose_correlation_vs_causation(
                    base_model.graph_learner.get_mean_adjacency(),
                    A_true.to(device),
                    X_sample,
                )
                val_metrics.update({f"diag_{k}": v for k, v in diag.items() if isinstance(v, (int, float))})
                val_metrics["diagnosis"] = diag["diagnosis"]
        
        # Log
        if is_main_process():
            entry = {
                "epoch": epoch,
                "time": t_elapsed,
                **train_metrics,
                **val_metrics,
            }
            history.append(entry)
            
            if not sweep_mode:
                # Stage info
                stage = "1:DISC" if epoch <= cfg["stage1_end"] * cfg["epochs"] else \
                        "2:PRUNE" if epoch <= cfg["stage2_end"] * cfg["epochs"] else "3:REFINE"
                
                # Compact log
                topk_f1 = val_metrics.get("topk_f1", 0)
                best_f1 = val_metrics.get("best_f1", 0)
                A_max = val_metrics.get("A_max", train_metrics.get("A_max", 0))
                
                print(f"[{stage}] Epoch {epoch:3d}/{cfg['epochs']} | "
                      f"loss={train_metrics['loss']:.4f} | "
                      f"L_recon={train_metrics.get('L_recon', 0):.4f} | "
                      f"A_max={A_max:.3f} | "
                      f"TopK-F1={topk_f1:.4f} Best-F1={best_f1:.4f} | "
                      f"{t_elapsed:.1f}s")
                
                # Diagnosis
                if "diagnosis" in val_metrics:
                    print(f"         | 🔍 Learning: {val_metrics['diagnosis'].upper()}")
        
        # Stage 1 health check
        stage1_end = int(cfg["stage1_end"] * cfg["epochs"])
        if epoch == stage1_end and is_main_process():
            A_max_val = val_metrics.get("A_max", train_metrics.get("A_max", 0))
            edges_02 = val_metrics.get("fixed02_pred_edges", 0)
            
            if A_max_val < 0.4 or edges_02 == 0:
                stage1_healthy = False
                if not sweep_mode:
                    print(f"         | ⚠️ Stage 1 UNHEALTHY: A_max={A_max_val:.3f}, edges@0.2={edges_02}")
            else:
                if not sweep_mode:
                    print(f"         | ✓ Stage 1 HEALTHY: A_max={A_max_val:.3f}, edges@0.2={edges_02}")
        
        # Stage 3 LR reduction
        stage3_start = int(cfg["stage2_end"] * cfg["epochs"])
        if epoch == stage3_start:
            for param_group in optimizer.param_groups:
                param_group["lr"] *= 0.5
            if is_main_process() and not sweep_mode:
                print(f"         | 📉 Stage 3: LR reduced to {optimizer.param_groups[0]['lr']:.1e}")
        
        # Model selection (TopK-F1 primary)
        if "topk_f1" in val_metrics:
            current_metric = val_metrics["topk_f1"]
            
            if current_metric > best_metric:
                best_metric = current_metric
                best_epoch = epoch
                patience_counter = 0
                
                if is_main_process():
                    # Save best model
                    model_to_save = model.module if ddp else model
                    torch.save({
                        "epoch": epoch,
                        "model_state": model_to_save.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "metrics": val_metrics,
                        "config": cfg,
                    }, output_path / "best_model.pt")
                    
                    # Save adjacency
                    A_pred = base_model.graph_learner.get_mean_adjacency()
                    np.save(output_path / "A_best.npy", A_pred.cpu().numpy())
                    
                    if not sweep_mode:
                        print(f"         | 🏆 New best! TopK-F1={best_metric:.4f}")
            else:
                patience_counter += 1
        
        # Early stopping
        if patience_counter >= cfg["patience"]:
            if is_main_process() and not sweep_mode:
                print(f"\n⏹️ Early stopping at epoch {epoch} (patience={cfg['patience']})")
            break
    
    # Final summary
    if is_main_process():
        # Save history
        with open(output_path / "training_history.json", "w") as f:
            json.dump(history, f, indent=2, default=str)
        
        # Save final adjacency
        A_final = base_model.graph_learner.get_mean_adjacency()
        np.save(output_path / "A_final.npy", A_final.cpu().numpy())
        
        if not sweep_mode:
            print("\n" + "=" * 70)
            print("TRAINING COMPLETE")
            print("=" * 70)
            print(f"Best TopK-F1: {best_metric:.4f} @ epoch {best_epoch}")
            print(f"Output: {output_path}")
            print("=" * 70)
    
    # Cleanup DDP
    if ddp:
        cleanup_ddp()
    
    # Return final metrics for sweep mode
    return {
        "best_topk_f1": best_metric,
        "best_epoch": best_epoch,
        "final_epoch": epoch,
        "stage1_healthy": stage1_healthy,
    }


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="RC-GNN Unified Training Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Required
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to data directory")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="artifacts/unified",
                        help="Output directory")
    
    # Training
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_CONFIG["weight_decay"])
    parser.add_argument("--grad_clip", type=float, default=DEFAULT_CONFIG["grad_clip"])
    parser.add_argument("--patience", type=int, default=DEFAULT_CONFIG["patience"])
    
    # Model
    parser.add_argument("--latent_dim", type=int, default=DEFAULT_CONFIG["latent_dim"])
    parser.add_argument("--hidden_dim", type=int, default=DEFAULT_CONFIG["hidden_dim"])
    parser.add_argument("--target_edges", type=int, default=DEFAULT_CONFIG["target_edges"])
    
    # Loss weights
    parser.add_argument("--lambda_recon", type=float, default=DEFAULT_CONFIG["lambda_recon"])
    parser.add_argument("--lambda_hsic", type=float, default=DEFAULT_CONFIG["lambda_hsic"])
    parser.add_argument("--lambda_sparse", type=float, default=DEFAULT_CONFIG["lambda_sparse"])
    parser.add_argument("--lambda_inv", type=float, default=DEFAULT_CONFIG["lambda_inv"])
    parser.add_argument("--lambda_causal", type=float, default=DEFAULT_CONFIG["lambda_causal"])
    
    # Features
    parser.add_argument("--use_groupdro", action="store_true",
                        help="Enable GroupDRO for worst-case robustness")
    parser.add_argument("--ddp", action="store_true",
                        help="Enable Distributed Data Parallel")
    parser.add_argument("--sweep_mode", action="store_true",
                        help="Minimal output for ablation sweeps")
    
    # System
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "cuda:0", "cuda:1"])
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--num_workers", type=int, default=DEFAULT_CONFIG["num_workers"])
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Build config from args
    config = {k: v for k, v in vars(args).items() 
              if k not in ["data_dir", "output_dir", "ddp", "use_groupdro", "sweep_mode"]}
    
    # Train
    results = train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        config=config,
        ddp=args.ddp,
        use_groupdro=args.use_groupdro,
        sweep_mode=args.sweep_mode,
    )
    
    # Print results in sweep mode
    if args.sweep_mode:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
