"""
Invariance Regularizer for RC-GNN v2

Implements structure-level invariance across environments:
- Computes per-environment adjacency matrices
- Penalizes edge-wise variance across environments
- Supports both soft and hard thresholded edges
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional


class InvarianceRegularizer(nn.Module):
    """
    Structure-level invariance regularization.
    
    Given K environments with adjacency matrices A_1, ..., A_K:
    
    L_inv = Σ_{i,j} Var_k(A_k[i,j])
    
    This encourages the learned causal structure to be consistent
    across different environmental conditions.
    
    Key insight: A TRUE causal structure should be invariant under
    distribution shifts (different regimes/environments), while
    spurious correlations will vary.
    """
    
    def __init__(
        self,
        method: str = "variance",  # "variance", "mmd", "wasserstein"
        aggregation: str = "sum",  # "sum", "mean", "max"
    ):
        super().__init__()
        self.method = method
        self.aggregation = aggregation
    
    def forward(
        self,
        A_list: List[torch.Tensor],
        weights: Optional[List[float]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute invariance loss.
        
        Args:
            A_list: List of K adjacency matrices, each [d, d]
            weights: Optional weights for each environment
            
        Returns:
            loss: Invariance penalty
            metrics: Dict of statistics
        """
        if len(A_list) < 2:
            # Need at least 2 environments
            return torch.tensor(0.0, device=A_list[0].device), {"l_inv": 0.0}
        
        # Stack adjacencies: [K, d, d]
        A_stack = torch.stack(A_list, dim=0)
        K = A_stack.shape[0]
        
        if self.method == "variance":
            loss = self._variance_penalty(A_stack, weights)
        elif self.method == "mmd":
            loss = self._mmd_penalty(A_stack)
        elif self.method == "wasserstein":
            loss = self._wasserstein_penalty(A_list)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Compute statistics
        A_mean = A_stack.mean(dim=0)
        A_std = A_stack.std(dim=0)
        
        metrics = {
            "l_inv": loss.item(),
            "A_mean_density": (A_mean > 0.5).float().mean().item(),
            "A_max_var": A_std.max().item(),
            "A_avg_var": A_std.mean().item(),
            "n_envs": K,
        }
        
        return loss, metrics
    
    def _variance_penalty(
        self,
        A_stack: torch.Tensor,
        weights: Optional[List[float]] = None,
    ) -> torch.Tensor:
        """
        Variance-based invariance penalty.
        
        L = Σ_{i,j} Var_k(A_k[i,j])
        
        For weighted version:
        L = Σ_{i,j} Σ_k w_k * (A_k[i,j] - A_mean[i,j])^2
        """
        K = A_stack.shape[0]
        
        if weights is not None:
            weights = torch.tensor(weights, device=A_stack.device, dtype=A_stack.dtype)
            weights = weights / weights.sum()  # Normalize
            
            # Weighted mean
            A_mean = (A_stack * weights.view(-1, 1, 1)).sum(dim=0)
            
            # Weighted variance
            diff_sq = (A_stack - A_mean.unsqueeze(0)) ** 2
            var = (diff_sq * weights.view(-1, 1, 1)).sum(dim=0)
        else:
            # Unweighted variance
            var = A_stack.var(dim=0, unbiased=False)
        
        if self.aggregation == "sum":
            return var.sum()
        elif self.aggregation == "mean":
            return var.mean()
        elif self.aggregation == "max":
            return var.max()
        else:
            return var.mean()
    
    def _mmd_penalty(self, A_stack: torch.Tensor) -> torch.Tensor:
        """
        Maximum Mean Discrepancy between environment adjacencies.
        
        MMD measures distributional distance using kernel embeddings.
        """
        K = A_stack.shape[0]
        
        # Flatten each A to vector
        A_flat = A_stack.view(K, -1)  # [K, d*d]
        
        # RBF kernel
        def rbf_kernel(X, Y, sigma=1.0):
            dist = torch.cdist(X, Y)
            return torch.exp(-dist ** 2 / (2 * sigma ** 2))
        
        # Compute pairwise MMD
        total_mmd = torch.tensor(0.0, device=A_stack.device)
        count = 0
        
        for i in range(K):
            for j in range(i + 1, K):
                x = A_flat[i:i+1]  # [1, d*d]
                y = A_flat[j:j+1]  # [1, d*d]
                
                kxx = rbf_kernel(x, x)
                kyy = rbf_kernel(y, y)
                kxy = rbf_kernel(x, y)
                
                mmd = kxx.mean() + kyy.mean() - 2 * kxy.mean()
                total_mmd = total_mmd + mmd
                count += 1
        
        return total_mmd / max(1, count)
    
    def _wasserstein_penalty(self, A_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Wasserstein (earth mover's) distance between adjacencies.
        
        Using sliced Wasserstein distance for efficiency.
        """
        K = len(A_list)
        d = A_list[0].shape[0]
        
        # Flatten
        A_flat = torch.stack([A.flatten() for A in A_list])  # [K, d*d]
        
        # Sliced Wasserstein via random projections
        n_projections = 50
        D = d * d
        
        # Random directions
        theta = torch.randn(n_projections, D, device=A_flat.device)
        theta = theta / theta.norm(dim=1, keepdim=True)
        
        # Project
        projections = torch.matmul(A_flat, theta.T)  # [K, n_proj]
        
        # Sort and compute pairwise distances
        sorted_proj, _ = projections.sort(dim=0)
        
        # Average pairwise Wasserstein
        total_dist = torch.tensor(0.0, device=A_flat.device)
        count = 0
        
        for i in range(K):
            for j in range(i + 1, K):
                dist = (sorted_proj[i] - sorted_proj[j]).abs().mean()
                total_dist = total_dist + dist
                count += 1
        
        return total_dist / max(1, count)


class BatchInvariance(nn.Module):
    """
    Compute invariance from batched data with environment labels.
    
    Usage:
        batch_inv = BatchInvariance(n_envs=5)
        A_per_env = batch_inv.compute_per_env_adjacency(A_batch, env_labels)
        loss, metrics = batch_inv.invariance(A_per_env)
    """
    
    def __init__(
        self,
        n_envs: int,
        regularizer: Optional[InvarianceRegularizer] = None,
    ):
        super().__init__()
        self.n_envs = n_envs
        self.regularizer = regularizer or InvarianceRegularizer()
    
    def compute_per_env_adjacency(
        self,
        A_batch: torch.Tensor,
        env_labels: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Aggregate batch adjacencies by environment.
        
        Args:
            A_batch: [B, d, d] adjacency matrices
            env_labels: [B] environment indices
            
        Returns:
            A_list: List of [d, d] mean adjacencies per environment
        """
        A_list = []
        
        for e in range(self.n_envs):
            mask = (env_labels == e)
            if mask.sum() > 0:
                A_env = A_batch[mask].mean(dim=0)
                A_list.append(A_env)
        
        return A_list
    
    def forward(
        self,
        A_batch: torch.Tensor,
        env_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute invariance loss from batched adjacencies.
        
        Args:
            A_batch: [B, d, d] adjacency matrices
            env_labels: [B] environment indices
            
        Returns:
            loss: Invariance penalty
            metrics: Statistics
        """
        A_list = self.compute_per_env_adjacency(A_batch, env_labels)
        
        if len(A_list) < 2:
            return torch.tensor(0.0, device=A_batch.device), {"l_inv": 0.0}
        
        return self.regularizer(A_list)


class IRM_Invariance(nn.Module):
    """
    IRM-style invariance for causal structure.
    
    Instead of penalizing gradient variance (as in original IRM),
    we penalize the variance of learned edges across environments.
    
    This is a structure-level adaptation of the IRM principle:
    "A truly causal model should be optimal under all environments"
    
    For causal graphs:
    "A truly causal edge should appear consistently across all environments"
    """
    
    def __init__(
        self,
        penalty_weight: float = 1.0,
        min_env_samples: int = 2,
    ):
        super().__init__()
        self.penalty_weight = penalty_weight
        self.min_env_samples = min_env_samples
    
    def forward(
        self,
        A_batch: torch.Tensor,
        env_labels: torch.Tensor,
        loss_per_sample: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute IRM-style structure invariance.
        
        Args:
            A_batch: [B, d, d] adjacency matrices
            env_labels: [B] environment indices
            loss_per_sample: Optional [B] loss per sample for gradient penalty
            
        Returns:
            penalty: IRM penalty
            metrics: Statistics
        """
        unique_envs = env_labels.unique()
        n_envs = len(unique_envs)
        
        if n_envs < 2:
            return torch.tensor(0.0, device=A_batch.device), {"l_irm": 0.0, "n_envs": 1}
        
        # Collect per-environment adjacencies
        A_per_env = []
        for e in unique_envs:
            mask = (env_labels == e)
            if mask.sum() >= self.min_env_samples:
                A_per_env.append(A_batch[mask].mean(dim=0))
        
        if len(A_per_env) < 2:
            return torch.tensor(0.0, device=A_batch.device), {"l_irm": 0.0}
        
        # Stack and compute variance
        A_stack = torch.stack(A_per_env, dim=0)  # [K, d, d]
        variance = A_stack.var(dim=0)  # [d, d]
        
        penalty = self.penalty_weight * variance.mean()
        
        metrics = {
            "l_irm": penalty.item(),
            "n_envs": len(A_per_env),
            "max_edge_var": variance.max().item(),
            "mean_edge_var": variance.mean().item(),
        }
        
        return penalty, metrics
