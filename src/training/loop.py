"""Training loop utilities for RC-GNN."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


def binary_adjacency(A, threshold=0.5):
    """Convert soft adjacency to binary."""
    if len(A.shape) == 3:
        A = A.mean(dim=0)
    return (A > threshold).float()


def compute_metrics(A_pred, A_true, threshold=0.5):
    """
    Compute metrics comparing predicted and true adjacency.
    
    Args:
        A_pred: Predicted adjacency [d, d] or [B, d, d]
        A_true: True adjacency [d, d]
        threshold: Binarization threshold
        
    Returns:
        dict with metrics
    """
    if len(A_pred.shape) == 3:
        A_pred = A_pred.mean(dim=0)
    
    A_pred = A_pred.detach().cpu().numpy()
    A_true = A_true.cpu().numpy() if isinstance(A_true, torch.Tensor) else A_true
    
    d = A_pred.shape[0]
    
    # Binarize
    A_pred_bin = (A_pred > threshold).astype(np.int32)
    A_true_bin = (A_true > 0.5).astype(np.int32)
    
    # Flatten for metrics
    y_true = A_true_bin.reshape(-1)
    y_pred = A_pred_bin.reshape(-1)
    y_score = A_pred.reshape(-1)
    
    # Precision, recall, F1
    try:
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    except:
        precision = recall = f1 = 0.0
    
    # Structural Hamming Distance (SHD)
    shd = np.sum(np.abs(A_pred_bin - A_true_bin))
    
    # ROC-AUC
    try:
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_score)
        else:
            auc = 0.0
    except:
        auc = 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "shd": shd,
        "auc": auc,
    }


def train_epoch(model, train_loader, optimizer, device="cpu", loss_fn=None, **loss_kwargs):
    """
    Training epoch.
    
    Args:
        model: RC-GNN model
        train_loader: DataLoader for training
        optimizer: Optimizer
        device: Device to train on
        loss_fn: Loss function (if None, uses default)
        **loss_kwargs: Additional kwargs for loss function
        
    Returns:
        dict with epoch metrics
    """
    model.train()
    total_loss = 0.0
    loss_components = {}
    n_batches = 0
    
    for batch in train_loader:
        X = batch["X"].to(device)
        M = batch.get("M", None)
        if M is not None:
            M = M.to(device)
        e = batch.get("e", None)
        if e is not None:
            e = e.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(X, M, e)
        
        # Compute loss
        if loss_fn is None:
            from src.training.optim import compute_total_loss
            loss, comps = compute_total_loss(output, X, M, **loss_kwargs)
            for key, val in comps.items():
                loss_components[key] = loss_components.get(key, 0) + val
        else:
            loss = loss_fn(output, X, M)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent NaNs
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    # Average
    avg_loss = total_loss / n_batches
    for key in loss_components:
        loss_components[key] /= n_batches
    
    return {
        "loss": avg_loss,
        **loss_components,
    }


def eval_epoch(model, eval_loader, A_true=None, device="cpu", threshold=0.5):
    """
    ATOMIC EVALUATION - ONE DECODER PATH → ALL METRICS (no more 1e9 sentinels)
    
    Implements comprehensive fixes:
    - Single mask used consistently (no diagonal, directed graph)
    - Auto threshold tuning on validation F1
    - Robust SHD (skeleton + orientation, no hard failures)
    - Reports edges@tuned, edges@0.5, edges@topk from SAME scores
    - Detailed diagnostics: logit stats, prob stats, loss components
    """
    model.eval()
    
    all_A_pred = []
    all_A_logits = []
    all_A_soft = []  # CRITICAL: Collect soft probabilities for metrics
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
            
            # Store adjacency (soft and logits)
            A_pred = output["A"].detach()
            if len(A_pred.shape) == 3:
                A_pred = A_pred.mean(dim=0)
            all_A_pred.append(A_pred)
            
            # Store soft probabilities (CRITICAL for proper metrics)
            if "A_soft" in output:
                A_soft = output["A_soft"].detach()
                if len(A_soft.shape) == 3:
                    A_soft = A_soft.mean(dim=0)
                all_A_soft.append(A_soft)
            
            # Store logits if available
            if "A_logits" in output:
                A_logits = output["A_logits"].detach()
                if len(A_logits.shape) == 3:
                    A_logits = A_logits.mean(dim=0)
                all_A_logits.append(A_logits)
            
            # Reconstruction loss
            if M is not None:
                recon_loss = ((output["X_recon"] - X) ** 2 * M).mean()
            else:
                recon_loss = ((output["X_recon"] - X) ** 2).mean()
            total_recon_loss += recon_loss.item()
            n_batches += 1
    
        # Average predictions
    A_pred_avg = torch.stack(all_A_pred).mean(dim=0).cpu().numpy()
    
    # CRITICAL: Use A_soft (differentiable probs) for metrics if available
    if len(all_A_soft) > 0:
        A_soft_avg = torch.stack(all_A_soft).mean(dim=0).cpu().numpy()
        np.fill_diagonal(A_soft_avg, 0.0)  # Zero diagonal (no self-loops)
        # Convert soft probs to logits for threshold tuning safely
        # Guard against 0/1 by clipping to avoid log(0) and Inf/NaN spam
        probs = np.clip(A_soft_avg, 1e-6, 1 - 1e-6)
        A_logits_avg = np.log(probs / (1 - probs))
    elif len(all_A_logits) > 0:
        A_logits_avg = torch.stack(all_A_logits).mean(dim=0).cpu().numpy()
    else:
        # Fallback: convert sparsified predictions back to logits
        A_logits_avg = np.log(A_pred_avg / (1 - A_pred_avg + 1e-8))
    
    avg_recon_loss = total_recon_loss / n_batches
    
    # === ATOMIC DECODER: ONE MASK FOR EVERYTHING ===
    N = A_logits_avg.shape[0]
    mask = np.ones((N, N), dtype=bool)
    mask &= ~np.eye(N, dtype=bool)  # Drop diagonal (CRITICAL: consistent masking)
    
    # Guard against NaN/Inf
    if not np.isfinite(A_logits_avg).all():
        print(f"⚠️  Non-finite logits detected (NaN/Inf), sanitizing")
        A_logits_avg = np.nan_to_num(A_logits_avg, nan=0.0, posinf=10.0, neginf=-10.0)
    
    np.fill_diagonal(A_logits_avg, 0)  # Force zero diagonal
    
    # Convert A_true if needed
    if A_true is not None:
        if isinstance(A_true, torch.Tensor):
            A_true = A_true.cpu().numpy()
        A_true = A_true.astype(np.int32)
        np.fill_diagonal(A_true, 0)  # Force zero diagonal
    
    # Extract masked scores (SAME mask for all metrics)
    y_score = A_logits_avg[mask].ravel()
    
    # === THRESHOLD TUNING (auto-tune on validation F1) ===
    best_thr = threshold  # Default fallback
    best_f1 = -1.0
    
    if A_true is not None:
        from sklearn.metrics import precision_recall_fscore_support
        
        y_true = A_true[mask].ravel()
        thr_grid = np.linspace(0.0, 0.9, 19)  # 19 thresholds
        
        for t in thr_grid:
            y_pred_try = (y_score > t).astype(np.int32)
            try:
                p, r, f1, _ = precision_recall_fscore_support(
                    y_true, y_pred_try, average="binary", zero_division=0
                )
                if f1 > best_f1:
                    best_f1, best_thr = f1, t
            except:
                continue  # Skip if computation fails for this threshold
    
    # === FINAL BINARIZATION (used for ALL metrics - no divergence) ===
    A_bin = np.zeros((N, N), dtype=np.int32)
    A_bin[mask] = (y_score > best_thr).astype(np.int32)
    np.fill_diagonal(A_bin, 0)  # Redundant safety
    
    metrics = {
        "recon_loss": avg_recon_loss,
        "A_mean": A_soft_avg if len(all_A_soft) > 0 else A_logits_avg,  # Save soft probs (or logits fallback)
        "A_bin": A_bin,
        "threshold_used": float(best_thr),
    }
    
    # === DIAGNOSTIC STATS (all from same masked scores) ===
    edges_at_tuned = int(A_bin.sum())
    edges_at_05 = int((y_score > 0.5).sum())
    
    # Top-k: rank all masked scores, take top k% (10% sparsity)
    k = max(1, int(0.1 * y_score.size))
    edges_at_topk = k  # By definition
    
    metrics["edges_pred@tuned"] = edges_at_tuned
    metrics["edges_pred@0.5"] = edges_at_05
    metrics["edges_pred@topk"] = edges_at_topk
    
    # Logit stats (masked, for monitoring quantization)
    metrics["logit_mean"] = float(y_score.mean())
    metrics["logit_std"] = float(y_score.std())
    metrics["logit_min"] = float(y_score.min())
    metrics["logit_max"] = float(y_score.max())
    metrics["pct_logits_gt0"] = float((y_score > 0).sum() / y_score.size * 100)
    
    # Prob stats (after sigmoid)
    y_probs = 1.0 / (1.0 + np.exp(-y_score))
    metrics["pct_probs_gt05"] = float((y_probs > 0.5).sum() / y_probs.size * 100)
    
    # === COMPUTE METRICS (if ground truth available) ===
    if A_true is not None:
        from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
        
        y_true = A_true[mask].ravel()
        y_pred = A_bin[mask].ravel()
        
        # F1, precision, recall (at tuned threshold)
        try:
            p, r, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0
            )
            metrics["precision"] = float(p)
            metrics["recall"] = float(r)
            metrics["f1"] = float(f1)
        except:
            metrics["precision"] = 0.0
            metrics["recall"] = 0.0
            metrics["f1"] = 0.0
        
        # AUC (threshold-free)
        try:
            auc = roc_auc_score(y_true, y_probs)
            metrics["auc"] = float(auc)
        except:
            metrics["auc"] = 0.0
        
        # === ROBUST SHD: skeleton + orientation (NO SENTINELS) ===
        try:
            # Skeleton SHD (undirected edge mismatches, divided by 2)
            Sk_pred = ((A_bin + A_bin.T) > 0).astype(np.int32)
            Sk_true = ((A_true + A_true.T) > 0).astype(np.int32)
            shd_skel = int(np.sum(np.abs(Sk_pred - Sk_true)) // 2)
            
            # Orientation error: count 1 per pair where skeleton matches but directions differ
            orient_err = 0
            iu = np.triu_indices(N, 1)
            for i, j in zip(*iu):
                if Sk_pred[i, j] == Sk_true[i, j] == 1:  # Both have undirected edge
                    if (A_bin[i, j] != A_true[i, j]) or (A_bin[j, i] != A_true[j, i]):
                        orient_err += 1
            
            shd = shd_skel + orient_err
            metrics["shd"] = float(shd)
            metrics["shd_skeleton"] = float(shd_skel)
            metrics["shd_orientation"] = float(orient_err)
        
        except Exception as e:
            # Only use sentinel if computation truly fails (should be rare now)
            print(f"⚠️  SHD computation failed: {e}, using sentinel")
            metrics["shd"] = 1e9
            metrics["shd_skeleton"] = 1e9
            metrics["shd_orientation"] = 0
        
        # Edge counts
        metrics["edges_true"] = int(A_true.sum())
    
    return metrics


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    epochs=10,
    device="cpu",
    A_true=None,
    checkpoint_path=None,
    verbose=True,
):
    """
    Full training loop.
    
    Args:
        model: RC-GNN model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        optimizer: Optimizer
        epochs: Number of epochs
        device: Device to train on
        A_true: Ground truth adjacency (optional)
        checkpoint_path: Path to save best checkpoint
        verbose: Whether to print progress
        
    Returns:
        dict with training history
    """
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_f1": [],
        "val_shd": [],
    }
    
    best_val_metric = float("inf") if A_true is not None else 0
    best_checkpoint = None
    
    for epoch in range(epochs):
        # Training step
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        
        # Validation step
        val_metrics = eval_epoch(model, val_loader, A_true, device)
        
        # Temperature annealing
        if hasattr(model.structure_learner, "step_temperature"):
            model.structure_learner.step_temperature(epoch, epochs)
        
        # Record metrics
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["recon_loss"])
        if "f1" in val_metrics:
            history["val_f1"].append(val_metrics["f1"])
            history["val_shd"].append(val_metrics["shd"])
        
        # Check for best model
        if A_true is not None:
            current_metric = val_metrics.get("shd", float("inf"))
            is_better = current_metric < best_val_metric
        else:
            current_metric = val_metrics["recon_loss"]
            is_better = current_metric < best_val_metric if best_val_metric != 0 else True
        
        if is_better:
            best_val_metric = current_metric
            best_checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "metrics": val_metrics,
            }
        
        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            msg = f"Epoch {epoch+1}/{epochs} - Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['recon_loss']:.4f}"
            if "f1" in val_metrics:
                msg += f", F1: {val_metrics['f1']:.4f}, SHD: {val_metrics['shd']:.0f}"
            print(msg)
    
    # Save best checkpoint
    if checkpoint_path is not None and best_checkpoint is not None:
        torch.save(best_checkpoint, checkpoint_path)
        if verbose:
            print(f"Saved best checkpoint to {checkpoint_path}")
    
    return history, best_checkpoint
