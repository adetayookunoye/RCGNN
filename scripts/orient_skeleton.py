#!/usr/bin/env python3
"""
orient_skeleton.py — Post-hoc orientation of RC-GNN learned skeleton.
================================================================================
V9.2.11 INSIGHT: RC-GNN learns good undirected skeleton (Skel-F1=0.53-0.57) but
direction is fundamentally limited by L_recon preferring bidirectional flow.
The decoder (A.T @ z_signal) maximizes reconstruction with dir_probs≈0.5.

SOLUTION: Decouple skeleton learning from orientation.
  1. Extract undirected skeleton from RC-GNN's learned adjacency (top-K by magnitude)
  2. Orient edges using methods that don't fight reconstruction:
     a) Variance ordering: Var(child) > Var(parent) in linear SEMs
     b) Lead-lag: Cross-correlation temporal ordering
     c) R²-residual: Regress each direction, lower residual variance = parent→child
     d) NOTEARS-constrained: Run NOTEARS but restrict to skeleton edges only
     e) Multi-environment invariance: True causal edges have invariant coefficients
  3. Evaluate each orientation against ground truth

Usage:
  python scripts/orient_skeleton.py \
    --artifacts-dir artifacts/table2a/h1_easy/seed_0 \
    --data-dir data/interim/table2a/h1_easy/seed_0 \
    --output artifacts/table2a/h1_easy/seed_0/orientation_results.json

The script outputs:
  - Comparison table of all orienters
  - Best-oriented A saved as A_posthoc_best.npy
  - JSON with all metrics
================================================================================
"""

import argparse
import json
import sys
import os
from pathlib import Path

import numpy as np

# --- path_helper: ensure project root on sys.path ---
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


# ============================================================================
# SKELETON EXTRACTION
# ============================================================================

def extract_undirected_skeleton(A: np.ndarray, K: int) -> np.ndarray:
    """
    Extract top-K undirected edges from adjacency matrix.
    
    For each pair (i,j), the undirected magnitude is max(A[i,j], A[j,i]).
    Select top-K pairs by magnitude → binary undirected skeleton.
    
    Args:
        A: [d, d] weighted adjacency (from RC-GNN)
        K: Number of edges to select (= true edge count)
    
    Returns:
        skel: [d, d] binary symmetric skeleton (skel[i,j] = skel[j,i] = 1 for edges)
    """
    d = A.shape[0]
    # Undirected magnitude: max(A[i,j], A[j,i])
    A_undir = np.maximum(A, A.T)
    # Zero diagonal
    np.fill_diagonal(A_undir, 0)
    
    # Extract upper triangle pairs
    rows, cols = np.triu_indices(d, k=1)
    pair_mags = A_undir[rows, cols]
    
    # Select top-K pairs
    K_pairs = min(K, len(pair_mags))
    topk_idx = np.argsort(pair_mags)[::-1][:K_pairs]
    
    skel = np.zeros((d, d))
    for idx in topk_idx:
        i, j = rows[idx], cols[idx]
        skel[i, j] = 1
        skel[j, i] = 1
    
    return skel


# ============================================================================
# ORIENTATION METHODS
# ============================================================================

def orient_by_variance(X: np.ndarray, M: np.ndarray, skel: np.ndarray) -> np.ndarray:
    """
    Orient by variance ordering: parent has lower variance than child.
    
    In linear SEMs: X_j = sum_i(w_ij * X_i) + noise_j
    Children accumulate variance from parents → Var(child) > Var(parent).
    
    For each edge (i,j) in skeleton:
      If Var(X_j) > Var(X_i): orient i→j  (i is parent)
      Else: orient j→i
    """
    d = X.shape[-1]
    
    # Compute masked variance per variable
    if M is not None:
        X_masked = X * M
        n_obs = M.sum(axis=tuple(range(M.ndim - 1)))  # [d]
        n_obs = np.maximum(n_obs, 1)
        mean_x = (X_masked).reshape(-1, d).sum(axis=0) / n_obs
        sq_dev = ((X_masked.reshape(-1, d) - mean_x) ** 2) * M.reshape(-1, d)
        var_x = sq_dev.sum(axis=0) / n_obs
    else:
        X_flat = X.reshape(-1, d)
        var_x = np.var(X_flat, axis=0)
    
    A_dir = np.zeros((d, d))
    rows, cols = np.where(np.triu(skel, k=1) > 0)
    
    for i, j in zip(rows, cols):
        if var_x[j] > var_x[i]:
            A_dir[i, j] = 1  # i→j (j has higher variance = child)
        elif var_x[i] > var_x[j]:
            A_dir[j, i] = 1  # j→i (i has higher variance = child)
        else:
            # Tie: arbitrarily orient i→j
            A_dir[i, j] = 1
    
    return A_dir


def orient_by_lead_lag(X: np.ndarray, M: np.ndarray, skel: np.ndarray,
                       max_lag: int = 5) -> np.ndarray:
    """
    Orient by lead-lag: if X_i leads X_j (corr(X_i[t], X_j[t+lag]) peaks at lag>0),
    then i→j (i is the cause).
    
    For temporal data [N, T, d], compute cross-correlation at various lags.
    """
    d = X.shape[-1]
    
    # Need temporal dimension
    if X.ndim == 2:
        # IID data — can't do lead-lag, fall back to variance
        print("  [lead-lag] No temporal structure, falling back to variance ordering")
        return orient_by_variance(X, M, skel)
    
    # X is [N, T, d]
    N, T, _ = X.shape
    if T < 3:
        print("  [lead-lag] T < 3, falling back to variance ordering")
        return orient_by_variance(X, M, skel)
    
    A_dir = np.zeros((d, d))
    rows, cols = np.where(np.triu(skel, k=1) > 0)
    
    for i, j in zip(rows, cols):
        # Compute cross-correlation: corr(X_i[t], X_j[t+lag]) for lag in [-max_lag, max_lag]
        best_lag = 0
        best_corr = 0
        
        for lag in range(-min(max_lag, T-1), min(max_lag, T-1) + 1):
            if lag == 0:
                continue
            if lag > 0:
                xi = X[:, :T-lag, i].flatten()
                xj = X[:, lag:, j].flatten()
            else:
                xi = X[:, -lag:, i].flatten()
                xj = X[:, :T+lag, j].flatten()
            
            if M is not None:
                if lag > 0:
                    mi = M[:, :T-lag, i].flatten()
                    mj = M[:, lag:, j].flatten()
                else:
                    mi = M[:, -lag:, i].flatten()
                    mj = M[:, :T+lag, j].flatten()
                valid = (mi > 0) & (mj > 0)
                xi, xj = xi[valid], xj[valid]
            
            if len(xi) < 10:
                continue
            
            # Pearson correlation
            xi_c = xi - xi.mean()
            xj_c = xj - xj.mean()
            denom = (np.sqrt(np.sum(xi_c**2)) * np.sqrt(np.sum(xj_c**2))) + 1e-8
            corr = np.sum(xi_c * xj_c) / denom
            
            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag
        
        # best_lag > 0 means X_i leads X_j → i→j
        # best_lag < 0 means X_j leads X_i → j→i
        if best_lag > 0:
            A_dir[i, j] = 1
        elif best_lag < 0:
            A_dir[j, i] = 1
        else:
            # No clear lead-lag, fall back to variance
            var_i = np.var(X[:, :, i])
            var_j = np.var(X[:, :, j])
            if var_j > var_i:
                A_dir[i, j] = 1
            else:
                A_dir[j, i] = 1
    
    return A_dir


def orient_by_residual(X: np.ndarray, M: np.ndarray, skel: np.ndarray) -> np.ndarray:
    """
    Orient by residual variance: for each edge (i,j), fit:
      X_j = β * X_i + ε_fwd  (i→j model)
      X_i = β * X_j + ε_rev  (j→i model)
    
    If Var(ε_fwd) < Var(ε_rev): i→j is more likely (i explains j better)
    
    This leverages the SEM structure: regressing on the true parent gives
    lower residual variance than regressing on the child.
    """
    d = X.shape[-1]
    X_flat = X.reshape(-1, d)
    
    if M is not None:
        M_flat = M.reshape(-1, d)
    else:
        M_flat = np.ones_like(X_flat)
    
    A_dir = np.zeros((d, d))
    rows, cols = np.where(np.triu(skel, k=1) > 0)
    
    for i, j in zip(rows, cols):
        # Get valid observations for both variables
        valid = (M_flat[:, i] > 0) & (M_flat[:, j] > 0)
        xi = X_flat[valid, i]
        xj = X_flat[valid, j]
        
        if len(xi) < 10:
            A_dir[i, j] = 1  # Default
            continue
        
        # Forward: X_j = β * X_i + ε
        xi_c = xi - xi.mean()
        xj_c = xj - xj.mean()
        beta_fwd = np.sum(xi_c * xj_c) / (np.sum(xi_c**2) + 1e-8)
        resid_fwd = xj_c - beta_fwd * xi_c
        var_fwd = np.var(resid_fwd)
        
        # Reverse: X_i = β * X_j + ε
        beta_rev = np.sum(xj_c * xi_c) / (np.sum(xj_c**2) + 1e-8)
        resid_rev = xi_c - beta_rev * xj_c
        var_rev = np.var(resid_rev)
        
        if var_fwd < var_rev:
            A_dir[i, j] = 1  # i→j (i explains j better)
        else:
            A_dir[j, i] = 1  # j→i (j explains i better)
    
    return A_dir


def orient_by_multi_parent_residual(X: np.ndarray, M: np.ndarray, 
                                      skel: np.ndarray) -> np.ndarray:
    """
    Orient by multi-parent residual: for each node j, try ALL skeleton neighbors
    as parents simultaneously. Compare models:
      X_j = Σ(β_k * X_k) + ε  for k ∈ neighbors(j) in skeleton
    
    For each edge (i,j), compare:
      R²(j|neighbors_j) vs R²(i|neighbors_i)
    
    The node with higher R² (better explained by its neighbors) is more likely
    the child. This is more powerful than pairwise residual because it uses
    the full parent set.
    """
    d = X.shape[-1]
    X_flat = X.reshape(-1, d)
    
    if M is not None:
        M_flat = M.reshape(-1, d)
    else:
        M_flat = np.ones_like(X_flat)
    
    # Get neighbors for each node
    neighbors = {i: list(np.where(skel[i] > 0)[0]) for i in range(d)}
    
    # Compute R² for each node given its skeleton neighbors
    r2 = np.zeros(d)
    for j in range(d):
        nb = neighbors[j]
        if len(nb) == 0:
            r2[j] = 0
            continue
        
        # Valid observations where j and all neighbors are observed
        valid = M_flat[:, j] > 0
        for k in nb:
            valid = valid & (M_flat[:, k] > 0)
        
        n_valid = valid.sum()
        if n_valid < max(10, len(nb) + 2):
            r2[j] = 0
            continue
        
        # Regress X_j on X_{neighbors}
        y = X_flat[valid, j]
        X_nb = X_flat[valid][:, nb]
        
        # Add intercept
        X_des = np.column_stack([X_nb, np.ones(n_valid)])
        
        # OLS: β = (X'X)^{-1} X'y
        try:
            beta = np.linalg.lstsq(X_des, y, rcond=None)[0]
            y_hat = X_des @ beta
            ss_res = np.sum((y - y_hat) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2[j] = 1 - ss_res / (ss_tot + 1e-8)
        except np.linalg.LinAlgError:
            r2[j] = 0
    
    # Orient: higher R² = child
    A_dir = np.zeros((d, d))
    rows, cols = np.where(np.triu(skel, k=1) > 0)
    
    for i, j in zip(rows, cols):
        if r2[j] > r2[i]:
            A_dir[i, j] = 1  # j has higher R² → j is child → i→j
        elif r2[i] > r2[j]:
            A_dir[j, i] = 1  # i has higher R² → i is child → j→i
        else:
            # Tie: variance fallback
            var_i = np.var(X_flat[:, i])
            var_j = np.var(X_flat[:, j])
            if var_j > var_i:
                A_dir[i, j] = 1
            else:
                A_dir[j, i] = 1
    
    return A_dir


def orient_by_invariance(X: np.ndarray, M: np.ndarray, e: np.ndarray,
                          skel: np.ndarray) -> np.ndarray:
    """
    Orient by multi-environment invariance (ICP-inspired).
    
    For each edge (i,j), fit X_j = β * X_i + ε in each environment.
    True causal direction i→j has INVARIANT β across environments.
    Wrong direction j→i has environment-dependent β (confounded).
    
    Compare: CoV(β_fwd across envs) vs CoV(β_rev across envs)
    Lower CoV = more invariant = more likely the causal direction.
    """
    d = X.shape[-1]
    X_flat = X.reshape(-1, d)
    
    if M is not None:
        M_flat = M.reshape(-1, d)
    else:
        M_flat = np.ones_like(X_flat)
    
    if e is not None:
        e_flat = e.reshape(-1) if e.ndim > 1 else np.repeat(e, X.shape[1] if X.ndim == 3 else 1)
        # For [N, T, d] data, expand e to match flattened X
        if X.ndim == 3 and len(e_flat) == X.shape[0]:
            e_flat = np.repeat(e_flat, X.shape[1])
    else:
        print("  [invariance] No environment labels, falling back to variance ordering")
        return orient_by_variance(X, M, skel)
    
    env_ids = np.unique(e_flat)
    if len(env_ids) < 2:
        print("  [invariance] Only 1 environment, falling back to variance ordering")
        return orient_by_variance(X, M, skel)
    
    A_dir = np.zeros((d, d))
    rows, cols = np.where(np.triu(skel, k=1) > 0)
    
    for i, j in zip(rows, cols):
        betas_fwd = []
        betas_rev = []
        
        for env in env_ids:
            env_mask = e_flat == env
            valid = env_mask & (M_flat[:, i] > 0) & (M_flat[:, j] > 0)
            n_valid = valid.sum()
            
            if n_valid < 10:
                continue
            
            xi = X_flat[valid, i]
            xj = X_flat[valid, j]
            xi_c = xi - xi.mean()
            xj_c = xj - xj.mean()
            
            # Forward: X_j = β * X_i
            beta_fwd = np.sum(xi_c * xj_c) / (np.sum(xi_c**2) + 1e-8)
            betas_fwd.append(beta_fwd)
            
            # Reverse: X_i = β * X_j
            beta_rev = np.sum(xj_c * xi_c) / (np.sum(xj_c**2) + 1e-8)
            betas_rev.append(beta_rev)
        
        if len(betas_fwd) < 2:
            # Not enough environments, fall back to variance
            var_i = np.var(X_flat[:, i])
            var_j = np.var(X_flat[:, j])
            if var_j > var_i:
                A_dir[i, j] = 1
            else:
                A_dir[j, i] = 1
            continue
        
        # Coefficient of variation: std/|mean|
        betas_fwd = np.array(betas_fwd)
        betas_rev = np.array(betas_rev)
        
        cov_fwd = np.std(betas_fwd) / (np.abs(np.mean(betas_fwd)) + 1e-8)
        cov_rev = np.std(betas_rev) / (np.abs(np.mean(betas_rev)) + 1e-8)
        
        if cov_fwd < cov_rev:
            A_dir[i, j] = 1  # i→j is more invariant
        else:
            A_dir[j, i] = 1  # j→i is more invariant
    
    return A_dir


def orient_by_notears_constrained(X: np.ndarray, M: np.ndarray,
                                    skel: np.ndarray,
                                    lambda1: float = 0.1) -> np.ndarray:
    """
    Run NOTEARS-linear but constrained to skeleton edges only.
    
    Mask out all non-skeleton entries so NOTEARS only searches over
    edge directions within the learned skeleton.
    """
    d = X.shape[-1]
    X_flat = X.reshape(-1, d)
    
    if M is not None:
        M_flat = M.reshape(-1, d)
        # Simple mean imputation for NOTEARS
        col_means = np.nanmean(np.where(M_flat > 0, X_flat, np.nan), axis=0)
        col_means = np.nan_to_num(col_means, nan=0.0)
        X_imp = X_flat.copy()
        for col in range(d):
            mask_col = M_flat[:, col] == 0
            X_imp[mask_col, col] = col_means[col]
    else:
        X_imp = X_flat
    
    # Standardize
    X_std = (X_imp - X_imp.mean(axis=0)) / (X_imp.std(axis=0) + 1e-8)
    
    n = X_std.shape[0]
    
    # NOTEARS with skeleton constraint
    # W is only allowed on skeleton edges
    W = np.zeros((d, d))
    
    # Initialize W on skeleton edges only
    for i in range(d):
        for j in range(d):
            if skel[i, j] > 0 and i != j:
                # OLS estimate: W[i,j] = how much X_i contributes to X_j
                xi = X_std[:, i]
                xj = X_std[:, j]
                W[i, j] = np.sum(xi * xj) / (np.sum(xi**2) + 1e-8) * 0.5
    
    # Simplified NOTEARS: gradient descent on L = ||X - XW||² + λ₁||W||₁
    # with acyclicity constraint h(W) = tr(e^{W∘W}) - d = 0
    # Constrained to skeleton mask
    
    lr = 0.01
    mu = 1.0  # Augmented Lagrangian penalty
    alpha_al = 0.0  # Lagrange multiplier for acyclicity
    
    for outer in range(10):  # Augmented Lagrangian outer iterations
        for step in range(200):
            # Gradient of ||X - XW||²/n
            R = X_std - X_std @ W
            grad_ls = -2.0 / n * (X_std.T @ R)
            
            # Gradient of acyclicity: h(W) = tr(e^{W∘W}) - d
            W_sq = W * W
            try:
                E = np.linalg.matrix_power(np.eye(d) + W_sq / d, d)
            except np.linalg.LinAlgError:
                E = np.eye(d)
            grad_h = 2 * W * E
            
            h_val = np.trace(E) - d
            
            # L1 subgradient
            grad_l1 = lambda1 * np.sign(W)
            
            # Total gradient
            grad = grad_ls + (alpha_al + mu * h_val) * grad_h + grad_l1
            
            # Zero gradient on non-skeleton entries (constraint)
            grad = grad * skel
            
            W = W - lr * grad
            
            # Project: zero non-skeleton entries
            W = W * skel
            np.fill_diagonal(W, 0)
        
        # Update Lagrangian
        W_sq = W * W
        try:
            E = np.linalg.matrix_power(np.eye(d) + W_sq / d, d)
        except np.linalg.LinAlgError:
            E = np.eye(d)
        h_val = np.trace(E) - d
        alpha_al += mu * h_val
        mu = min(mu * 2, 1e6)
        
        if h_val < 1e-8:
            break
    
    # Convert continuous W to binary DAG
    # For each pair (i,j), keep the stronger direction
    A_dir = np.zeros((d, d))
    rows, cols = np.where(np.triu(skel, k=1) > 0)
    
    for i, j in zip(rows, cols):
        if abs(W[i, j]) > abs(W[j, i]):
            A_dir[i, j] = 1
        elif abs(W[j, i]) > abs(W[i, j]):
            A_dir[j, i] = 1
        else:
            # Tie: variance fallback
            var_i = np.var(X_flat[:, i])
            var_j = np.var(X_flat[:, j])
            if var_j > var_i:
                A_dir[i, j] = 1
            else:
                A_dir[j, i] = 1
    
    return A_dir


def orient_ensemble(orientations: dict, skel: np.ndarray) -> np.ndarray:
    """
    Majority-vote ensemble: for each edge, take the direction that most
    orienters agree on.
    """
    d = skel.shape[0]
    A_dir = np.zeros((d, d))
    rows, cols = np.where(np.triu(skel, k=1) > 0)
    
    for i, j in zip(rows, cols):
        votes_ij = 0  # i→j
        votes_ji = 0  # j→i
        
        for name, A in orientations.items():
            if A[i, j] > 0:
                votes_ij += 1
            elif A[j, i] > 0:
                votes_ji += 1
        
        if votes_ij >= votes_ji:
            A_dir[i, j] = 1
        else:
            A_dir[j, i] = 1
    
    return A_dir


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def compute_metrics(A_pred: np.ndarray, A_true: np.ndarray, K: int) -> dict:
    """Compute directed and skeleton metrics."""
    d = A_true.shape[0]
    A_true_bin = (A_true > 0).astype(float)
    A_pred_bin = (A_pred > 0).astype(float)
    
    # Directed F1 (main metric)
    tp = np.sum((A_pred_bin > 0) & (A_true_bin > 0))
    fp = np.sum((A_pred_bin > 0) & (A_true_bin == 0))
    fn = np.sum((A_pred_bin == 0) & (A_true_bin > 0))
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    
    # Skeleton F1
    skel_pred = np.maximum(A_pred_bin, A_pred_bin.T)
    skel_true = np.maximum(A_true_bin, A_true_bin.T)
    # Upper triangle only
    upper = np.triu(np.ones((d, d)), k=1)
    sp = skel_pred * upper
    st = skel_true * upper
    tp_skel = np.sum((sp > 0) & (st > 0))
    fp_skel = np.sum((sp > 0) & (st == 0))
    fn_skel = np.sum((sp == 0) & (st > 0))
    prec_skel = tp_skel / (tp_skel + fp_skel) if (tp_skel + fp_skel) > 0 else 0
    rec_skel = tp_skel / (tp_skel + fn_skel) if (tp_skel + fn_skel) > 0 else 0
    f1_skel = 2 * prec_skel * rec_skel / (prec_skel + rec_skel) if (prec_skel + rec_skel) > 0 else 0
    
    # SHD
    shd = int(np.sum(np.abs(A_pred_bin - A_true_bin)))
    
    # Bidirectional edges (should be 0 for post-hoc orientation)
    bidir = 0
    for i in range(d):
        for j in range(i+1, d):
            if A_pred_bin[i, j] > 0 and A_pred_bin[j, i] > 0:
                bidir += 1
    
    # Orientation accuracy: among correctly identified skeleton edges,
    # how many have the right direction?
    n_correct_dir = 0
    n_skel_match = 0
    for i in range(d):
        for j in range(i+1, d):
            # Check if this undirected edge is in both skeleton pred and true
            pred_edge = (A_pred_bin[i,j] > 0) or (A_pred_bin[j,i] > 0)
            true_edge = (A_true_bin[i,j] > 0) or (A_true_bin[j,i] > 0)
            if pred_edge and true_edge:
                n_skel_match += 1
                # Check direction matches
                if A_pred_bin[i,j] == A_true_bin[i,j] and A_pred_bin[j,i] == A_true_bin[j,i]:
                    n_correct_dir += 1
    
    orient_acc = n_correct_dir / n_skel_match if n_skel_match > 0 else 0
    
    return {
        'TopK_F1': round(f1, 4),
        'Skel_F1': round(f1_skel, 4),
        'SHD': shd,
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'bidir': bidir,
        'orient_acc': round(orient_acc, 4),
        'skel_matches': n_skel_match,
        'dir_correct': n_correct_dir,
        'n_edges_pred': int(A_pred_bin.sum()),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Post-hoc skeleton orientation")
    parser.add_argument("--artifacts-dir", required=True, help="RC-GNN artifacts directory")
    parser.add_argument("--data-dir", required=True, help="Dataset directory")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--adjacency", default="A_best_score.npy",
                        help="Which adjacency to load (default: A_best_score.npy)")
    parser.add_argument("--K", type=int, default=None,
                        help="Number of edges (default: from A_true)")
    args = parser.parse_args()
    
    art_dir = Path(args.artifacts_dir)
    data_dir = Path(args.data_dir)
    
    print("=" * 72)
    print("POST-HOC SKELETON ORIENTATION")
    print("=" * 72)
    print(f"  Artifacts: {art_dir}")
    print(f"  Data:      {data_dir}")
    print(f"  Adjacency: {args.adjacency}")
    
    # Load data
    A_rcgnn = np.load(art_dir / args.adjacency)
    A_true = np.load(data_dir / "A_true.npy")
    X = np.load(data_dir / "X.npy")
    M = np.load(data_dir / "M.npy")
    
    e = None
    e_path = data_dir / "e.npy"
    if e_path.exists():
        e = np.load(e_path)
    
    d = A_true.shape[0]
    K = args.K or int(A_true.sum())
    
    print(f"  d={d}, K={K} (true edges), X shape={X.shape}")
    print("=" * 72)
    
    # Step 1: Extract undirected skeleton
    skel = extract_undirected_skeleton(A_rcgnn, K)
    n_skel_edges = int(np.triu(skel, k=1).sum())
    print(f"\n[1] Extracted undirected skeleton: {n_skel_edges} edges (target K={K})")
    
    # Evaluate skeleton quality
    skel_true = np.maximum((A_true > 0).astype(float), (A_true > 0).astype(float).T)
    upper = np.triu(np.ones((d, d)), k=1)
    tp_skel = np.sum((skel * upper > 0) & (skel_true * upper > 0))
    print(f"    Skeleton TP: {int(tp_skel)}/{K} edges correctly identified")
    
    # Step 2: Run all orienters
    print(f"\n[2] Running orientation methods...")
    
    orientations = {}
    
    # RC-GNN native (baseline — what we're trying to beat)
    A_rcgnn_topk = np.zeros((d, d))
    A_flat = A_rcgnn.copy()
    np.fill_diagonal(A_flat, 0)
    topk_idx = np.argsort(A_flat.flatten())[::-1][:K]
    for idx in topk_idx:
        i, j = idx // d, idx % d
        A_rcgnn_topk[i, j] = 1
    orientations['rcgnn_native'] = A_rcgnn_topk
    print(f"  ✓ RC-GNN native direction (baseline)")
    
    # a) Variance ordering
    A_var = orient_by_variance(X, M, skel)
    orientations['variance'] = A_var
    print(f"  ✓ Variance ordering")
    
    # b) Lead-lag
    A_ll = orient_by_lead_lag(X, M, skel)
    orientations['lead_lag'] = A_ll
    print(f"  ✓ Lead-lag")
    
    # c) Pairwise residual
    A_resid = orient_by_residual(X, M, skel)
    orientations['residual'] = A_resid
    print(f"  ✓ Pairwise residual")
    
    # d) Multi-parent residual (R²)
    A_mpr = orient_by_multi_parent_residual(X, M, skel)
    orientations['multi_parent_R2'] = A_mpr
    print(f"  ✓ Multi-parent R²")
    
    # e) Invariance (if environments available)
    if e is not None:
        A_inv = orient_by_invariance(X, M, e, skel)
        orientations['invariance'] = A_inv
        print(f"  ✓ Multi-environment invariance")
    
    # f) NOTEARS-constrained
    A_nt = orient_by_notears_constrained(X, M, skel)
    orientations['notears_constrained'] = A_nt
    print(f"  ✓ NOTEARS-constrained")
    
    # g) Ensemble (majority vote, excluding RC-GNN native)
    posthoc_orientations = {k: v for k, v in orientations.items() if k != 'rcgnn_native'}
    A_ens = orient_ensemble(posthoc_orientations, skel)
    orientations['ensemble'] = A_ens
    print(f"  ✓ Ensemble (majority vote of {len(posthoc_orientations)} methods)")
    
    # Step 3: Evaluate all
    print(f"\n[3] Evaluation against ground truth:")
    print(f"{'Method':<25} {'TopK-F1':>8} {'Skel-F1':>8} {'SHD':>5} {'bidir':>6} "
          f"{'OrientAcc':>10} {'DirCorr':>8}")
    print("-" * 72)
    
    results = {}
    best_f1 = 0
    best_method = None
    
    for name, A_pred in orientations.items():
        metrics = compute_metrics(A_pred, A_true, K)
        results[name] = metrics
        
        print(f"  {name:<23} {metrics['TopK_F1']:>8.4f} {metrics['Skel_F1']:>8.4f} "
              f"{metrics['SHD']:>5d} {metrics['bidir']:>6d} "
              f"{metrics['orient_acc']:>10.4f} {metrics['dir_correct']:>5d}/{metrics['skel_matches']}")
        
        if name != 'rcgnn_native' and metrics['TopK_F1'] > best_f1:
            best_f1 = metrics['TopK_F1']
            best_method = name
    
    print("-" * 72)
    
    # Report improvement
    native_f1 = results['rcgnn_native']['TopK_F1']
    if best_method:
        improvement = best_f1 - native_f1
        print(f"\n  BEST post-hoc: {best_method} (F1={best_f1:.4f}, "
              f"{'+'if improvement>=0 else ''}{improvement:.4f} vs native)")
        
        # Save best oriented adjacency
        A_best = orientations[best_method]
        out_path = art_dir / "A_posthoc_best.npy"
        np.save(out_path, A_best)
        print(f"  Saved: {out_path}")
        
        # Also save ensemble
        np.save(art_dir / "A_posthoc_ensemble.npy", A_ens)
        print(f"  Saved: {art_dir / 'A_posthoc_ensemble.npy'}")
    
    # Save results JSON
    output_path = args.output or str(art_dir / "orientation_results.json")
    output = {
        'skeleton_edges': n_skel_edges,
        'skeleton_tp': int(tp_skel),
        'K': K,
        'd': d,
        'best_method': best_method,
        'best_f1': best_f1,
        'native_f1': native_f1,
        'improvement': best_f1 - native_f1 if best_method else 0,
        'methods': results,
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\n  Results → {output_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
