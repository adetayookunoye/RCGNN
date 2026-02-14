import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Union
import warnings


class IRMStructureInvariance(nn.Module):
    """IRM-style invariance regularization for causal structure learning.
    
    Enforces invariant causal mechanisms across environments through:
    1. IRM gradient penalty: Encourages predictor to be simultaneously optimal across envs
    2. Structure variance penalty: Penalizes differences in learned structures across envs
    
    The module supports two modes for IRM penalty computation:
    - 'grad_logits': Direct gradient of risk w.r.t. adjacency logits (recommended)
    - 'scale': Classical IRM with learnable scaling factor (more stable but less direct)
    
    Args:
        n_features: Dimensionality of features/variables
        n_envs: Number of environments
        irm_mode: 'grad_logits' or 'scale' - how to compute IRM penalty
        gamma: Weight for the IRM gradient penalty
        var_weight: Weight for structure variance penalty
        causal_convention: 'i->j' (predictor causes target) or 'j->i'
        eps: Small constant for numerical stability
    """
    
    def __init__(
        self,
        n_features: int,
        n_envs: int,
        irm_mode: str = 'grad_logits',
        gamma: float = 0.1,
        var_weight: float = 1.0,
        causal_convention: str = 'i->j',
        eps: float = 1e-8,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_envs = n_envs
        self.irm_mode = irm_mode
        self.gamma = gamma
        self.var_weight = var_weight
        self.eps = eps
        
        assert causal_convention in ('i->j', 'j->i'), \
            "causal_convention must be 'i->j' or 'j->i'"
        self.causal_convention = causal_convention
        
        # Buffers for monitoring only
        self.register_buffer('env_risks', torch.zeros(n_envs))
        self.register_buffer('grad_norms', torch.zeros(n_envs))
        
        assert irm_mode in ('grad_logits', 'scale'), \
            "irm_mode must be 'grad_logits' or 'scale'"
    
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _maybe_expand_dims(
        self,
        A: torch.Tensor,
        logits: Optional[torch.Tensor],
        X: torch.Tensor,
        M: torch.Tensor,
        e: Union[torch.Tensor, int],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Ensure every tensor has a batch dimension."""
        if A.dim() == 2:
            A = A.unsqueeze(0)
        if logits is not None and logits.dim() == 2:
            logits = logits.unsqueeze(0)
        if X.dim() == 2:
            X = X.unsqueeze(0)
            M = M.unsqueeze(0)
        if not isinstance(e, torch.Tensor):
            e = torch.tensor([e], device=A.device, dtype=torch.long)
        else:
            e = e.to(A.device)
        return A, logits, X, M, e
    
    def _apply_causal_convention(
        self,
        adj: torch.Tensor,
        X_past: torch.Tensor,
    ) -> torch.Tensor:
        """Predict next-step values according to the chosen causal convention."""
        if self.causal_convention == 'i->j':
            # A[i,j]=1 means i causes j → X_next[j] = Σ_i A[i,j]*X_past[i]
            return torch.matmul(X_past, adj)
        else:
            return torch.matmul(X_past, adj.transpose(-2, -1))
    
    # ------------------------------------------------------------------
    # IRM gradient penalty
    # ------------------------------------------------------------------
    def compute_irm_grad_penalty(
        self,
        A: torch.Tensor,
        logits: torch.Tensor,
        X: torch.Tensor,
        M: torch.Tensor,
        e: torch.Tensor,
    ) -> torch.Tensor:
        """Compute IRM gradient penalty using the configured mode.
        
        Returns:
            grad_penalty: scalar tensor
        """
        if not torch.is_grad_enabled():
            return torch.tensor(0.0, device=A.device, dtype=A.dtype)
        
        A, logits, X, M, e = self._maybe_expand_dims(A, logits, X, M, e)
        B, d, _ = A.shape
        device, dtype = A.device, A.dtype
        
        if logits is None:
            logits = A
        
        # Identify active environments (need ≥2 for variance)
        env_masks = torch.stack([e == i for i in range(self.n_envs)])
        active_envs = env_masks.any(dim=1).nonzero(as_tuple=True)[0]
        
        if len(active_envs) < 2:
            return torch.tensor(0.0, device=device, dtype=dtype)
        
        # Shared prediction
        support = torch.sigmoid(logits)          # [B, d, d]
        effective_adj = A * support              # [B, d, d]
        X_past = X[:, :-1]                       # [B, T-1, d]
        X_future = X[:, 1:]                      # [B, T-1, d]
        mask_future = M[:, 1:].to(dtype)         # [B, T-1, d]
        
        preds = self._apply_causal_convention(effective_adj, X_past)
        errors = ((preds - X_future) ** 2) * mask_future
        risks = errors.mean(dim=(1, 2))          # [B]
        
        grads = []
        
        if self.irm_mode == 'grad_logits':
            for env_idx in active_envs:
                env_mask = env_masks[env_idx]
                if not env_mask.any():
                    continue
                risk_env = risks[env_mask].mean()
                try:
                    grad_logits = torch.autograd.grad(
                        risk_env, logits,
                        retain_graph=True,
                        allow_unused=True,
                        create_graph=True,
                    )[0]
                    if grad_logits is not None:
                        grads.append(grad_logits[env_mask].mean(dim=0).flatten())
                    else:
                        grads.append(torch.zeros(d * d, device=device, dtype=dtype))
                except RuntimeError:
                    # Fallback to scale-based IRM when logits have no grad_fn
                    scale = torch.ones(1, device=device, dtype=dtype, requires_grad=True)
                    preds_sc = self._apply_causal_convention(
                        effective_adj[env_mask] * scale, X_past[env_mask])
                    err_sc = ((preds_sc - X_future[env_mask]) ** 2) * mask_future[env_mask]
                    grad_sc = torch.autograd.grad(
                        err_sc.mean(), scale, retain_graph=True, create_graph=True)[0]
                    grads.append(grad_sc.expand(d * d))
        
        elif self.irm_mode == 'scale':
            for env_idx in active_envs:
                env_mask = env_masks[env_idx]
                if not env_mask.any():
                    continue
                scale = torch.ones(1, device=device, dtype=dtype, requires_grad=True)
                preds_sc = self._apply_causal_convention(
                    effective_adj[env_mask] * scale, X_past[env_mask])
                err_sc = ((preds_sc - X_future[env_mask]) ** 2) * mask_future[env_mask]
                risk_sc = err_sc.mean()
                grad_sc = torch.autograd.grad(
                    risk_sc, scale, retain_graph=True, create_graph=True)[0]
                grads.append(grad_sc.expand(d * d))
        
        if grads:
            grads_tensor = torch.stack(grads)            # [n_active, d*d]
            if grads_tensor.shape[0] > 1:
                edge_var = torch.var(grads_tensor, dim=0, unbiased=False)
                return edge_var.mean() * self.gamma
        
        return torch.tensor(0.0, device=device, dtype=dtype)
    
    # ------------------------------------------------------------------
    # Structure variance
    # ------------------------------------------------------------------
    def compute_structure_variance(
        self,
        A: torch.Tensor,
        logits: Optional[torch.Tensor],
        e: Union[torch.Tensor, int],
    ) -> torch.Tensor:
        """Variance of adjacency across environments.
        
        Returns:
            var_penalty: scalar tensor
        """
        A, logits, _, _, e = self._maybe_expand_dims(
            A, logits, torch.zeros(1), torch.zeros(1), e)
        
        env_masks = torch.stack([e == i for i in range(self.n_envs)])
        active_envs = env_masks.any(dim=1)
        
        if active_envs.sum() < 2:
            return torch.tensor(0.0, device=A.device, dtype=A.dtype)
        
        env_adj_means = []
        env_prob_means = []
        
        for env_idx in range(self.n_envs):
            if active_envs[env_idx]:
                env_mask = env_masks[env_idx]
                env_adj_means.append(A[env_mask].mean(dim=0))
                if logits is not None:
                    env_prob_means.append(
                        torch.sigmoid(logits[env_mask]).mean(dim=0))
        
        env_adj_means = torch.stack(env_adj_means)      # [n_active, d, d]
        var_penalty = torch.var(env_adj_means, dim=0, unbiased=False).mean()
        
        # Pairwise L1 differences
        n_active = env_adj_means.shape[0]
        if n_active > 1:
            pw = []
            for i in range(n_active):
                for j in range(i + 1, n_active):
                    pw.append(torch.abs(env_adj_means[i] - env_adj_means[j]).mean())
            var_penalty = var_penalty + torch.stack(pw).mean()
        
        # Logit-probability variance
        if env_prob_means:
            env_prob_means = torch.stack(env_prob_means)
            if env_prob_means.shape[0] > 1:
                var_penalty = var_penalty + torch.var(
                    env_prob_means, dim=0, unbiased=False).mean()
        
        return var_penalty
    
    # ------------------------------------------------------------------
    # Per-environment risks (monitoring only)
    # ------------------------------------------------------------------
    def _compute_env_risks_detached(
        self,
        A: torch.Tensor,
        logits: Optional[torch.Tensor],
        X: torch.Tensor,
        M: torch.Tensor,
        e: Union[torch.Tensor, int],
    ) -> torch.Tensor:
        """Per-environment MSE risks for monitoring (detached)."""
        A, logits, X, M, e = self._maybe_expand_dims(A, logits, X, M, e)
        if logits is None:
            logits = A
        
        support = torch.sigmoid(logits)
        effective_adj = A * support
        X_past = X[:, :-1]
        X_future = X[:, 1:]
        mask_future = M[:, 1:].to(A.dtype)
        
        preds = self._apply_causal_convention(effective_adj, X_past)
        errors = ((preds - X_future) ** 2) * mask_future
        per_sample = errors.mean(dim=(1, 2))
        
        risks = torch.zeros(self.n_envs, device=A.device, dtype=A.dtype)
        for idx in range(self.n_envs):
            mask = (e == idx)
            if mask.any():
                risks[idx] = per_sample[mask].mean()
        return risks.detach()
    
    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        A: torch.Tensor,
        logits: Optional[torch.Tensor],
        X: torch.Tensor,
        M: torch.Tensor,
        e: Union[torch.Tensor, int],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute total invariance loss.
        
        Returns:
            total_loss: Combined invariance penalty
            metrics: Dict with keys grad_penalty, var_penalty, env_risks, total_invariance
        """
        grad_penalty = self.compute_irm_grad_penalty(A, logits, X, M, e)
        var_penalty = self.compute_structure_variance(A, logits, e)
        total_loss = grad_penalty + self.var_weight * var_penalty
        
        # Update monitoring buffer
        try:
            risks = self._compute_env_risks_detached(A, logits, X, M, e)
            if risks.numel() == self.env_risks.numel():
                self.env_risks.copy_(risks)
        except Exception:
            risks = self.env_risks.clone()
        
        metrics = {
            'env_risks': risks,
            'grad_penalty': grad_penalty.detach(),
            'var_penalty': var_penalty.detach(),
            'total_invariance': total_loss.detach(),
        }
        return total_loss, metrics
    
    # ------------------------------------------------------------------
    # Backward-compatible wrappers (old API surface)
    # ------------------------------------------------------------------
    def compute_env_risk(self, A, logits, X, M, e):
        """Legacy wrapper — returns (risks, grad_penalty)."""
        grad_penalty = self.compute_irm_grad_penalty(A, logits, X, M, e)
        risks = self._compute_env_risks_detached(A, logits, X, M, e)
        return risks, grad_penalty
    
    def structure_variance(self, A, logits, e):
        """Legacy wrapper — delegates to compute_structure_variance."""
        return self.compute_structure_variance(A, logits, e)
