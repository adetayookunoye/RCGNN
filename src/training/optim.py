"""Optimization utilities for RC-GNN."""

import torch
from typing import Dict, List


def make_optimizer(model, lr=0.001, weight_decay=1e-5):
    """
    Create optimizer for RC-GNN.
    
    Args:
        model: RC-GNN model
        lr: Learning rate
        weight_decay: Weight decay (L2 regularization)
        
    Returns:
        torch.optim.Optimizer
    """
    # Ensure numeric types
    lr = float(lr)
    weight_decay = float(weight_decay)
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def sparsity_loss(A, target_sparsity=0.1):
    """
    Sparsity regularization for adjacency matrix.
    
    Args:
        A: Adjacency matrix [B, d, d] or [d, d]
        target_sparsity: Target sparsity ratio
        
    Returns:
        Scalar loss
    """
    if len(A.shape) == 3:
        A = A.mean(dim=0)
    
    # L1 regularization encourages sparsity
    return A.abs().mean()


def acyclicity_loss(A, n_power=3):
    """
    Acyclicity penalty for DAG constraint (numerically stable).
    
    Uses the trace-based penalty: tr(I + A/d)^n - d
    which encourages acyclicity while avoiding gradient explosion.
    
    Args:
        A: Adjacency matrix [d, d] or [B, d, d]  
        n_power: Power for penalty computation
        
    Returns:
        Scalar loss (numerically stable)
    """
    if len(A.shape) == 3:
        A = A.mean(dim=0)
    
    d = A.shape[0]
    device = A.device
    dtype = A.dtype
    
    # Normalize A to prevent explosion: use A/d scaled form
    # This ensures eigenvalues stay bounded
    A_norm = A / float(d)
    
    # Compute (I + A/d)^n using binary exponentiation for stability
    I = torch.eye(d, device=device, dtype=dtype)
    result = I.clone()
    base = I + A_norm
    
    # Efficient exponentiation
    exp = n_power
    while exp > 0:
        if exp % 2 == 1:
            result = torch.matmul(result, base)
        base = torch.matmul(base, base)
        exp //= 2
        
        # Keep norms under control
        result_norm = torch.norm(result)
        if result_norm > 100:  # If growing too large, scale down
            result = result / (result_norm / 50.0)
    
    # tr((I + A/d)^n) should be close to d for a DAG
    trace_val = torch.trace(result)
    
    # Penalty: ReLU(trace - d) encourages acyclicity
    penalty = torch.nn.functional.relu(trace_val - d)
    
    return penalty


def reconstruction_loss(X_recon, X, M=None):
    """
    Reconstruction loss (MSE).
    
    Args:
        X_recon: Reconstructed data
        X: Original data
        M: Missingness mask (optional) - only measure loss on observed values
        
    Returns:
        Scalar loss
    """
    if M is not None:
        # Only compute loss on observed values
        diff = (X_recon - X) * M
        return (diff ** 2).mean()
    else:
        return ((X_recon - X) ** 2).mean()


def disentanglement_loss(z_s, z_n, z_b):
    """
    Disentanglement loss for latent factors (minimize correlation).
    
    Args:
        z_s: Signal latent [B, T, latent_dim]
        z_n: Noise latent [B, T, latent_dim]
        z_b: Bias latent [B, T, latent_dim]
        
    Returns:
        Scalar loss
    """
    B, T, latent_dim = z_s.shape
    
    # Flatten time dimension
    z_s_flat = z_s.reshape(B * T, latent_dim)
    z_n_flat = z_n.reshape(B * T, latent_dim)
    z_b_flat = z_b.reshape(B * T, latent_dim)
    
    # Normalize
    z_s_norm = (z_s_flat - z_s_flat.mean(dim=0)) / (z_s_flat.std(dim=0) + 1e-8)
    z_n_norm = (z_n_flat - z_n_flat.mean(dim=0)) / (z_n_flat.std(dim=0) + 1e-8)
    z_b_norm = (z_b_flat - z_b_flat.mean(dim=0)) / (z_b_flat.std(dim=0) + 1e-8)
    
    # Correlation-based loss
    corr_sn = torch.abs(torch.corrcoef(torch.cat([z_s_norm.T, z_n_norm.T], dim=0))).mean()
    corr_sb = torch.abs(torch.corrcoef(torch.cat([z_s_norm.T, z_b_norm.T], dim=0))).mean()
    corr_nb = torch.abs(torch.corrcoef(torch.cat([z_n_norm.T, z_b_norm.T], dim=0))).mean()
    
    return corr_sn + corr_sb + corr_nb


def compute_total_loss(
    output: Dict,
    X: torch.Tensor,
    M: torch.Tensor = None,
    A_true: torch.Tensor = None,
    lambda_recon: float = 1.0,
    lambda_sparse: float = 0.01,
    lambda_acyclic: float = 0.1,
    lambda_disen: float = 0.01,
    target_sparsity: float = 0.1,
):
    """
    Compute total training loss.
    
    Args:
        output: Model forward output dictionary
        X: Original data
        M: Missingness mask (optional)
        A_true: Ground truth adjacency (optional, for auxiliary loss)
        lambda_*: Loss weights
        
    Returns:
        Total scalar loss, dict of component losses
    """
    losses = {}
    
    # Reconstruction loss
    l_recon = reconstruction_loss(output["X_recon"], X, M)
    losses["recon"] = l_recon.item()
    
    # CRITICAL: Use A_soft (differentiable sigmoid probs) for all losses, not sparsified A
    # This enables gradient flow to structure learner
    A_for_loss = output.get("A_soft", output["A"])
    
    # Zero diagonal before any loss computation (no self-loops)
    if len(A_for_loss.shape) == 3:
        A_for_loss_clean = A_for_loss.clone()
        for i in range(A_for_loss.shape[0]):
            A_for_loss_clean[i].fill_diagonal_(0.0)
    else:
        A_for_loss_clean = A_for_loss.clone()
        A_for_loss_clean.fill_diagonal_(0.0)
    
    # Sparsity loss (on soft probs)
    l_sparse = sparsity_loss(A_for_loss_clean, target_sparsity)
    losses["sparse"] = l_sparse.item()
    
    # Acyclicity loss (on soft probs with zero diagonal)
    A_mean = A_for_loss_clean.mean(dim=0) if len(A_for_loss_clean.shape) == 3 else A_for_loss_clean
    l_acyclic = acyclicity_loss(A_mean)
    losses["acyclic"] = l_acyclic.item()
    
    # Disentanglement loss
    l_disen = disentanglement_loss(output["z_s"], output["z_n"], output["z_b"])
    losses["disen"] = l_disen.item()
    
    # Total loss
    total = (
        lambda_recon * l_recon +
        lambda_sparse * l_sparse +
        lambda_acyclic * l_acyclic +
        lambda_disen * l_disen
    )
    
    return total, losses
