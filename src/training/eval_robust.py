"""Robust evaluation utilities for RC-GNN - handles edge cases in SHD computation."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


def evaluate_adj(
    A_pred,             # float scores or logits in [N,N]
    A_true,             # binary ground truth in [N,N]
    threshold=None,     # if None -> tune by F1 on val grid
    directed=True,      # evaluate direction; if False -> skeleton only
    mask=None,          # boolean [N,N] of valid edge positions
    tune_grid=None,     # thresholds to search for best F1
):
    """
    Evaluate predicted adjacency against ground truth with proper SHD computation.
    
    Key features:
    - Threshold tuning by F1 on validation grid
    - Skeleton vs. directed evaluation modes
    - Proper SHD: skeleton error + orientation error
    - Masking for invalid edge positions
    - No self-loops, consistent metric computation
    
    Args:
        A_pred: Predicted adjacency [N,N], float scores or logits
        A_true: Ground truth adjacency [N,N], binary
        threshold: Binarization threshold; if None, tuned by F1
        directed: If True, evaluate directions; if False, skeleton only
        mask: Valid edge positions [N,N] boolean (excludes self-loops by default)
        tune_grid: Thresholds to search for best F1 (default: linspace(0, 0.9, 10))
        
    Returns:
        dict with metrics: precision, recall, f1, auc, shd, edges_pred, edges_true, 
                          threshold_used, directed_eval
    """
    N = A_true.shape[0]

    # ----- 1) Build a clean mask: no self-loops; (optionally) only one triangle for skeleton -----
    if mask is None:
        mask = np.ones((N, N), dtype=bool)
    mask &= ~np.eye(N, dtype=bool)  # drop diagonal

    # If evaluating undirected structure quality (skeleton), only keep upper triangle to avoid double counting
    if not directed:
        tri_mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        mask &= tri_mask

    # ----- 2) Prepare ground truth & scores under the mask -----
    A_true_bin = (A_true > 0.5).astype(np.int32)
    y_true = A_true_bin[mask].reshape(-1)
    y_score = A_pred[mask].reshape(-1)

    # Degenerate GT guard
    has_pos = (y_true.sum() > 0) and (y_true.sum() < y_true.size)

    # ----- 3) Threshold selection (tune once, reuse everywhere) -----
    if threshold is None:
        if tune_grid is None:
            tune_grid = np.linspace(0.0, 0.9, 10)
        best_f1, best_thr = -1.0, 0.5
        for thr in tune_grid:
            y_pred_try = (y_score > thr).astype(np.int32)
            p, r, f1, _ = precision_recall_fscore_support(
                y_true, y_pred_try, average="binary", zero_division=0
            )
            if f1 > best_f1:
                best_f1, best_thr = f1, thr
        threshold = best_thr

    # Final binarization
    y_pred = (y_score > threshold).astype(np.int32)

    # ----- 4) Precision / Recall / F1 / AUC -----
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    try:
        auc = roc_auc_score(y_true, y_score) if has_pos else 0.0
    except Exception:
        auc = 0.0

    # ----- 5) SHD (skeleton + orientation aware) -----
    # Reconstruct masked matrices for SHD
    A_pred_bin = np.zeros_like(A_true_bin)
    A_pred_bin[mask] = y_pred
    A_true_m = A_true_bin * mask
    A_pred_m = A_pred_bin * mask

    # Skeleton SHD: treat edges as undirected (orientation ignored)
    Sk_true = ((A_true_m + A_true_m.T) > 0).astype(np.int32)
    Sk_pred = ((A_pred_m + A_pred_m.T) > 0).astype(np.int32)
    shd_skeleton = np.sum(np.abs(Sk_true - Sk_pred)) // 2  # each undirected edge counted once

    # Orientation error: count where skeletons agree but directions differ
    # For each unordered pair (i<j) with an edge in both skeletons, add 1 if directions disagree
    orient_err = 0
    iu = np.triu_indices(N, k=1)
    for i, j in zip(iu[0], iu[1]):
        if Sk_true[i, j] == 1 and Sk_pred[i, j] == 1:
            # true directions
            t_ij, t_ji = A_true_m[i, j], A_true_m[j, i]
            p_ij, p_ji = A_pred_m[i, j], A_pred_m[j, i]
            if (t_ij != p_ij) or (t_ji != p_ji):
                # directions differ (reversal or bidirection mismatch) -> count 1
                orient_err += 1

    # Total SHD (common convention): additions+deletions (skeleton) + orientation errors
    shd = int(shd_skeleton + orient_err)

    # ----- 6) Edge counts (use SAME threshold!) -----
    edges_true = int(A_true_m.sum())
    edges_pred = int(A_pred_m.sum())

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
        "shd": shd,
        "edges_pred": edges_pred,
        "edges_true": edges_true,
        "threshold_used": float(threshold),
        "directed_eval": bool(directed),
    }


def compute_metrics_robust(A_pred, A_true, threshold=0.5, verbose=False):
    """
    Wrapper around evaluate_adj() with backward compatibility.
    
    Converts tensors to numpy and handles batch dimensions.
    Falls back to evaluate_adj() for robust computation.
    """
    try:
        # Convert to numpy
        if isinstance(A_pred, torch.Tensor):
            A_pred = A_pred.detach().cpu().numpy()
        if isinstance(A_true, torch.Tensor):
            A_true = A_true.cpu().numpy()
        
        # Handle batch dimension (take mean)
        if len(A_pred.shape) == 3:
            A_pred = A_pred.mean(axis=0)
        if len(A_true.shape) == 3:
            A_true = A_true.mean(axis=0)
        
        # Validate shapes match
        if A_pred.shape != A_true.shape:
            return {
                "precision": 0.0, "recall": 0.0, "f1": 0.0, "shd": 1e9, "auc": 0.0,
                "edge_count": 0, "edges_pred": 0, "edges_true": 0,
                "error_msg": f"Shape mismatch: A_pred {A_pred.shape} vs A_true {A_true.shape}",
                "warning": None
            }
        
        # Finiteness check
        if not np.all(np.isfinite(A_pred)):
            return {
                "precision": 0.0, "recall": 0.0, "f1": 0.0, "shd": 1e9, "auc": 0.0,
                "edge_count": 0, "edges_pred": 0, "edges_true": 0,
                "error_msg": f"Non-finite values in A_pred",
                "warning": None
            }
        
        # Auto-sigmoid if raw logits
        A_pred_range = A_pred.max() - A_pred.min()
        if A_pred_range > 10:
            A_pred = 1.0 / (1.0 + np.exp(-A_pred))
        
        # Clamp to [0, 1]
        A_pred = np.clip(A_pred, 0, 1)
        
        # Use evaluate_adj with fixed threshold
        result = evaluate_adj(A_pred, A_true, threshold=threshold, directed=True)
        result["error_msg"] = None
        result["edge_count"] = result["edges_pred"]  # Backward compat
        
        if verbose:
            print(f"   ✓ Edges predicted: {result['edges_pred']}/{result['edges_true']} | "
                  f"F1: {result['f1']:.4f} | SHD: {result['shd']:.0f}")
        
        return result
        
    except Exception as e:
        return {
            "precision": 0.0, "recall": 0.0, "f1": 0.0, "shd": 1e9, "auc": 0.0,
            "edge_count": 0, "edges_pred": 0, "edges_true": 0,
            "error_msg": f"Unexpected error: {str(e)}",
            "warning": None
        }


def eval_epoch_robust(model, eval_loader, A_true=None, device="cpu", threshold=0.5, verbose=False):
    """
    Evaluation epoch with robust error handling.
    
    Args:
        model: RC-GNN model
        eval_loader: DataLoader for evaluation
        A_true: Ground truth adjacency [d, d] (optional)
        device: Device to evaluate on
        threshold: Threshold for binarization
        verbose: Print diagnostics
        
    Returns:
        dict with evaluation metrics
    """
    model.eval()
    
    all_A_pred = []
    total_recon_loss = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for batch in eval_loader:
            X = batch["X"].to(device)
            M = batch.get("M", None)
            if M is not None:
                M = M.to(device)
            e = batch.get("e", None)
            if e is not None:
                e = e.to(device)
            
            # Forward pass
            output = model(X, M, e)
            
            # Store adjacency
            A_pred = output["A"].detach()
            if len(A_pred.shape) == 3:
                A_pred = A_pred.mean(dim=0)
            all_A_pred.append(A_pred)
            
            # Reconstruction loss
            if M is not None:
                recon_loss = ((output["X_recon"] - X) ** 2 * M).mean()
            else:
                recon_loss = ((output["X_recon"] - X) ** 2).mean()
            total_recon_loss += recon_loss.item()
            n_batches += 1
    
    # Average adjacency
    A_pred_avg = torch.stack(all_A_pred).mean(dim=0)
    avg_recon_loss = total_recon_loss / max(1, n_batches)
    
    metrics = {
        "recon_loss": avg_recon_loss,
        "A_mean": A_pred_avg.cpu().numpy(),
    }
    
    # Compute structural metrics with robust error handling
    if A_true is not None:
        struct_metrics = compute_metrics_robust(A_pred_avg, A_true, threshold, verbose=verbose)
        metrics.update(struct_metrics)
    else:
        # No ground truth, fill in defaults
        metrics.update({
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "shd": 1e9,
            "auc": 0.0,
            "edge_count": 0,
            "error_msg": "No ground truth available",
        })
    
    return metrics
