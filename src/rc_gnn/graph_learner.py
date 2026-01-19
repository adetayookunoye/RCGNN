"""
Causal Graph Learner for RC-GNN v2

Learns a weighted adjacency matrix A with:
- TRUE NOTEARS acyclicity constraint: h(A) = tr(exp(A∘A)) - d
- L1 sparsity regularization
- Per-environment structure support
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List


def notears_h(A: torch.Tensor) -> torch.Tensor:
    """
    Compute the NOTEARS acyclicity constraint.
    
    h(A) = tr(exp(A ∘ A)) - d
    
    Where:
    - A ∘ A is element-wise product
    - exp is matrix exponential
    - For a DAG, h(A) = 0
    
    Args:
        A: Adjacency matrix [d, d] with values in [0, 1]
        
    Returns:
        h: Acyclicity penalty (0 for DAG, > 0 for cyclic)
    """
    d = A.shape[0]
    
    # Element-wise square
    A_sq = A * A
    
    # Matrix exponential (numerically stable via eigendecomposition for small d)
    # For larger d, use truncated series
    if d <= 50:
        exp_A = torch.matrix_exp(A_sq)
    else:
        # Truncated series: exp(A) ≈ Σ_{k=0}^{K} A^k / k!
        exp_A = torch.eye(d, device=A.device, dtype=A.dtype)
        A_power = torch.eye(d, device=A.device, dtype=A.dtype)
        for k in range(1, 10):  # 10 terms is usually sufficient
            A_power = A_power @ A_sq / k
            exp_A = exp_A + A_power
    
    h = torch.trace(exp_A) - d
    return h


def notears_h_poly(A: torch.Tensor, power: int = 10) -> torch.Tensor:
    """
    Polynomial approximation of NOTEARS constraint.
    
    h_poly(A) = tr((I + A∘A/d)^d) - d
    
    More numerically stable for larger graphs.
    """
    d = A.shape[0]
    A_sq = A * A / d
    
    M = torch.eye(d, device=A.device, dtype=A.dtype) + A_sq
    
    # Binary exponentiation for efficiency
    result = torch.eye(d, device=A.device, dtype=A.dtype)
    base = M.clone()
    exp = min(power, d)
    
    while exp > 0:
        if exp % 2 == 1:
            result = result @ base
        base = base @ base
        exp //= 2
    
    h = torch.trace(result) - d
    return F.relu(h)  # Ensure non-negative


class CausalGraphLearner(nn.Module):
    """
    Learns causal adjacency matrix A from latent representations.
    
    Architecture options:
    1. Direct parameterization: A = sigmoid(W) where W is learned
    2. Attention-based: A_ij = softmax(z_i^T W z_j)
    3. Bilinear: A_ij = sigmoid(z_i^T W z_j)
    
    Constraints:
    - NOTEARS acyclicity: h(A) = 0
    - Sparsity: ||A||_1 is minimized
    - No self-loops: diag(A) = 0
    """
    
    def __init__(
        self,
        d: int,
        latent_dim: Optional[int] = None,
        hidden_dim: int = 64,
        method: str = "direct",  # "direct", "attention", "bilinear"
        n_envs: int = 1,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.d = d
        self.latent_dim = latent_dim
        self.method = method
        self.n_envs = n_envs
        
        # Temperature for sigmoid (annealed during training)
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("step", torch.tensor(0))
        
        if method == "direct":
            # Direct parameterization: W_ij is learned directly
            # Initialize so sigmoid(W) ≈ 0.15 (not 0.5)
            # logit(0.15) ≈ -1.73
            init_logit = -1.73
            self.W = nn.Parameter(init_logit + torch.randn(d, d) * 0.1)
            
        elif method == "attention":
            assert latent_dim is not None, "latent_dim required for attention"
            self.query = nn.Linear(latent_dim, hidden_dim)
            self.key = nn.Linear(latent_dim, hidden_dim)
            self.scale = math.sqrt(hidden_dim)
            
        elif method == "bilinear":
            assert latent_dim is not None, "latent_dim required for bilinear"
            self.bilinear = nn.Parameter(torch.randn(latent_dim, latent_dim) * 0.01)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Per-environment deltas (if multi-environment)
        if n_envs > 1:
            self.env_deltas = nn.ParameterList([
                nn.Parameter(torch.zeros(d, d)) for _ in range(n_envs)
            ])
        else:
            self.env_deltas = None
        
        # Mask for self-loops (diagonal = 0)
        self.register_buffer("no_self_loop_mask", 1.0 - torch.eye(d))
    
    def get_adjacency_logits(
        self,
        z: Optional[torch.Tensor] = None,
        env_idx: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute raw adjacency logits (before sigmoid).
        
        Args:
            z: Latent representations [B, d, latent_dim] or None for direct method
            env_idx: Environment index for per-env deltas
            
        Returns:
            logits: Raw adjacency logits [d, d] or [B, d, d]
        """
        if self.method == "direct":
            logits = self.W
            
        elif self.method == "attention":
            # z: [B, d, latent_dim] or [d, latent_dim]
            if z.dim() == 2:
                z = z.unsqueeze(0)
            
            Q = self.query(z)  # [B, d, hidden]
            K = self.key(z)    # [B, d, hidden]
            
            # Attention scores
            logits = torch.bmm(Q, K.transpose(-1, -2)) / self.scale  # [B, d, d]
            
        elif self.method == "bilinear":
            if z.dim() == 2:
                z = z.unsqueeze(0)
            
            # z: [B, d, latent_dim]
            # logits_ij = z_i^T W z_j
            z_W = torch.matmul(z, self.bilinear)  # [B, d, latent_dim]
            logits = torch.bmm(z_W, z.transpose(-1, -2))  # [B, d, d]
        
        # Add environment-specific deltas
        if env_idx is not None and self.env_deltas is not None:
            delta = self.env_deltas[env_idx]
            if logits.dim() == 3:
                logits = logits + delta.unsqueeze(0)
            else:
                logits = logits + delta
        
        return logits
    
    def forward(
        self,
        z: Optional[torch.Tensor] = None,
        env_idx: Optional[int] = None,
        return_soft: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Compute adjacency matrix.
        
        Args:
            z: Latent representations (optional for direct method)
            env_idx: Environment index
            return_soft: Whether to return soft probabilities
            
        Returns:
            dict with:
                - A: Adjacency matrix [d, d] or [B, d, d]
                - logits: Raw logits
                - A_soft: Soft probabilities (same as A for now)
        """
        logits = self.get_adjacency_logits(z, env_idx)
        
        # Apply temperature-scaled sigmoid
        A_soft = torch.sigmoid(logits / self.temperature)
        
        # Zero out diagonal (no self-loops)
        if A_soft.dim() == 3:
            mask = self.no_self_loop_mask.unsqueeze(0)
        else:
            mask = self.no_self_loop_mask
        
        A = A_soft * mask
        
        return {
            "A": A,
            "logits": logits,
            "A_soft": A,
        }
    
    def acyclicity_penalty(self, A: torch.Tensor, method: str = "notears") -> torch.Tensor:
        """
        Compute acyclicity penalty.
        
        Args:
            A: Adjacency matrix [d, d] or [B, d, d]
            method: "notears" or "poly"
            
        Returns:
            h: Acyclicity penalty
        """
        if A.dim() == 3:
            A = A.mean(dim=0)  # Average over batch
        
        if method == "notears":
            return notears_h(A)
        else:
            return notears_h_poly(A)
    
    def sparsity_penalty(self, A: torch.Tensor) -> torch.Tensor:
        """
        Compute L1 sparsity penalty.
        
        Args:
            A: Adjacency matrix
            
        Returns:
            l1: L1 norm of A
        """
        if A.dim() == 3:
            A = A.mean(dim=0)
        return A.abs().mean()
    
    def step_temperature(
        self,
        current_step: int,
        total_steps: int,
        start_temp: float = 1.0,
        end_temp: float = 0.1
    ):
        """Anneal temperature during training."""
        progress = min(1.0, current_step / max(1, total_steps))
        temp = start_temp + (end_temp - start_temp) * progress
        self.temperature.fill_(max(temp, 0.01))
        self.step.fill_(current_step)


class GraphLearnerLoss(nn.Module):
    """
    Combined loss for graph learner.
    
    L_graph = γ * h(A) + λ * ||A||_1
    
    Supports warmup: γ=0 for first warmup_epochs, then ramps up.
    """
    
    def __init__(
        self,
        gamma_acyclic: float = 1.0,
        lambda_sparse: float = 0.01,
        acyclicity_method: str = "notears",
        warmup_epochs: int = 10,
        ramp_epochs: int = 20,
    ):
        super().__init__()
        self.gamma_acyclic_target = gamma_acyclic
        self.lambda_sparse = lambda_sparse
        self.acyclicity_method = acyclicity_method
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = ramp_epochs
        self.current_epoch = 0
    
    def set_epoch(self, epoch: int):
        """Set current epoch for warmup scheduling."""
        self.current_epoch = epoch
    
    def get_gamma(self) -> float:
        """Get current acyclicity weight with warmup."""
        if self.current_epoch < self.warmup_epochs:
            return 0.0
        elif self.current_epoch < self.warmup_epochs + self.ramp_epochs:
            # Linear ramp from 0 to target
            progress = (self.current_epoch - self.warmup_epochs) / self.ramp_epochs
            return self.gamma_acyclic_target * progress
        else:
            return self.gamma_acyclic_target
    
    def forward(
        self,
        graph_output: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute graph learning loss.
        
        Args:
            graph_output: Dict from CausalGraphLearner.forward()
            
        Returns:
            loss: Total loss
            metrics: Dict of individual components
        """
        A = graph_output["A"]
        
        # Reduce to [d, d] if batched
        if A.dim() == 3:
            A_mean = A.mean(dim=0)
        else:
            A_mean = A
        
        # Acyclicity penalty
        if self.acyclicity_method == "notears":
            l_acyclic = notears_h(A_mean)
        else:
            l_acyclic = notears_h_poly(A_mean)
        
        # Sparsity penalty
        l_sparse = A_mean.abs().mean()
        
        # Total loss with warmup for acyclicity
        gamma = self.get_gamma()
        loss = gamma * l_acyclic + self.lambda_sparse * l_sparse
        
        metrics = {
            "l_acyclic": l_acyclic.item(),
            "l_sparse": l_sparse.item(),
            "l_graph_total": loss.item(),
            "h_A": l_acyclic.item(),
            "A_density": (A_mean > 0.5).float().mean().item(),
            "gamma_current": gamma,
        }
        
        return loss, metrics
