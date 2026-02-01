"""
Causal Priors Module for RC-GNN V4

Implements four key innovations to distinguish causation from correlation:

1. Interventional Signal Integration
   - Uses regime-specific corruptions as soft interventions
   - Intervention targets break symmetry in causal discovery

2. Orientation Penalty (ANM-based)
   - Uses entropy asymmetry: causes have lower entropy than effects
   - Based on Additive Noise Model (ANM) identifiability

3. Edge Counterfactual Validation
   - Tests necessity of each edge during training
   - True causal edges: removal breaks prediction
   - Spurious edges: removal has little effect

4. Environment-Specific Decoders
   - Per-regime decoders with shared structure A
   - Mechanism invariance: ||f_e1(A,z) - f_e2(A,z)||
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List


# =============================================================================
# 1. INTERVENTIONAL SIGNAL INTEGRATION
# =============================================================================

class InterventionAwareStructure(nn.Module):
    """
    Uses intervention targets from regime metadata to inform structure learning.
    
    Key insight: In an intervention on node X_i, incoming edges to X_i are broken.
    If regime e has intervention on node i, we should see A[:,i] differ from 
    observational regime.
    
    This provides HARD causal identification when intervention targets are known.
    """
    
    def __init__(
        self,
        d: int,
        n_regimes: int = 1,
        intervention_strength: float = 1.0,
    ):
        super().__init__()
        self.d = d
        self.n_regimes = n_regimes
        self.intervention_strength = intervention_strength
        
        # Intervention target mask per regime: [n_regimes, d]
        # intervention_mask[e, i] = 1 means node i is intervened in regime e
        # This will be set from data config if available
        self.register_buffer(
            "intervention_mask", 
            torch.zeros(n_regimes, d)
        )
        
    def set_intervention_targets(self, targets: Dict[int, List[int]]):
        """
        Set known intervention targets per regime.
        
        Args:
            targets: Dict mapping regime_idx -> list of intervened node indices
                     e.g., {0: [], 1: [3, 5], 2: [0, 7]}
        """
        mask = torch.zeros(self.n_regimes, self.d)
        for regime_idx, nodes in targets.items():
            if regime_idx < self.n_regimes:
                for node in nodes:
                    if node < self.d:
                        mask[regime_idx, node] = 1.0
        self.intervention_mask.copy_(mask)
        
    def compute_intervention_loss(
        self,
        A_per_env: torch.Tensor, # [n_envs, d, d] or [B, d, d] per-sample
        regimes: torch.Tensor, # [B] regime indices
    ) -> torch.Tensor:
        """
        Compute intervention consistency loss.
        
        For intervened nodes, incoming edges should be suppressed in that regime.
        L_int = sum over intervened (i,e): A[e, :, i].sum()
        
        This encourages the model to learn that interventions break incoming edges.
        """
        if self.intervention_mask.sum() == 0:
            # No intervention targets set - skip
            return torch.tensor(0.0, device=A_per_env.device)
        
        B = regimes.shape[0]
        device = A_per_env.device
        
        loss = torch.tensor(0.0, device=device)
        count = 0
        
        for b in range(B):
            e = regimes[b].item()
            if e >= self.n_regimes:
                continue
                
            # Get intervention targets for this regime
            int_mask = self.intervention_mask[e] # [d]
            
            if int_mask.sum() > 0:
                # For each intervened node, penalize incoming edges
                # A_per_env[b, :, i] = incoming edges to node i
                if A_per_env.dim() == 3 and A_per_env.shape[0] == B:
                    A_sample = A_per_env[b] # [d, d]
                else:
                    A_sample = A_per_env[e] if A_per_env.shape[0] == self.n_regimes else A_per_env
                
                # Incoming edges to intervened nodes should be small
                incoming = A_sample.sum(dim=0) # [d] - sum of incoming per node
                intervened_incoming = (incoming * int_mask).sum()
                loss = loss + intervened_incoming
                count += int_mask.sum().item()
        
        if count > 0:
            loss = loss / count
            
        return self.intervention_strength * loss


# =============================================================================
# 2. ORIENTATION PENALTY (ANM-based)
# =============================================================================

class OrientationPenalty(nn.Module):
    """
    Orientation penalty based on Additive Noise Model (ANM) identifiability.
    
    Key insight from causal discovery theory:
    - In X -> Y: Y = f(X) + ε, where X ⊥ ε
    - The effect Y typically has HIGHER entropy than the cause X
      (noise adds uncertainty)
    - Residuals of correct direction have lower dependence on cause
    
    We implement a soft penalty that prefers edges where:
    1. Source node has lower entropy than target node
    2. Residual of regression is independent of source
    """
    
    def __init__(
        self,
        d: int,
        lambda_orient: float = 0.1,
        use_residual_test: bool = True,
    ):
        super().__init__()
        self.d = d
        self.lambda_orient = lambda_orient
        self.use_residual_test = use_residual_test
        
    def estimate_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Estimate differential entropy of each variable.
        Uses Gaussian approximation: H(X) ≈ 0.5 * log(2πe * Var(X))
        
        Args:
            x: [N, d] data samples
        Returns:
            [d] entropy estimates
        """
        var = x.var(dim=0) + 1e-8
        # Gaussian differential entropy
        entropy = 0.5 * torch.log(2 * math.pi * math.e * var)
        return entropy
    
    def compute_residual_independence(
        self,
        X: torch.Tensor, # [B, T, d] or [N, d]
        A: torch.Tensor, # [d, d]
    ) -> torch.Tensor:
        """
        For each edge (i->j), compute how independent the residual is from cause.
        
        residual[j] = X[j] - predicted[j]
        predicted[j] = sum_i A[i,j] * X[i]
        
        Return correlation between residual and cause (lower = better orientation)
        """
        # Flatten to [N, d]
        if X.dim() == 3:
            X_flat = X.reshape(-1, X.shape[-1])
        else:
            X_flat = X
        
        N, d = X_flat.shape
        device = X_flat.device
        
        # Predicted values using current A
        X_pred = X_flat @ A # [N, d]
        residuals = X_flat - X_pred # [N, d]
        
        # For each edge (i,j), compute |corr(residual[j], X[i])|
        # This should be low for correctly oriented edges
        independence_penalty = torch.zeros(d, d, device=device)
        
        X_centered = X_flat - X_flat.mean(dim=0, keepdim=True)
        res_centered = residuals - residuals.mean(dim=0, keepdim=True)
        
        X_std = X_centered.std(dim=0) + 1e-8
        res_std = res_centered.std(dim=0) + 1e-8
        
        # Compute correlation matrix between residuals and inputs
        for j in range(d):
            for i in range(d):
                if i != j:
                    corr = (X_centered[:, i] * res_centered[:, j]).mean()
                    corr = corr / (X_std[i] * res_std[j])
                    independence_penalty[i, j] = corr.abs()
        
        return independence_penalty
    
    def forward(
        self,
        X: torch.Tensor,
        A: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute orientation penalty.
        
        Returns:
            loss: Orientation penalty (scalar)
            metrics: Dict with diagnostic info
        """
        device = A.device
        
        # Flatten X
        if X.dim() == 3:
            X_flat = X.reshape(-1, X.shape[-1])
        else:
            X_flat = X
        
        # 1. Entropy-based orientation
        entropy = self.estimate_entropy(X_flat) # [d]
        
        # For each edge A[i,j] > 0, prefer H(X_i) < H(X_j)
        # Create penalty matrix: penalty[i,j] = max(0, H(X_i) - H(X_j))
        # High penalty when cause has higher entropy than effect (wrong direction)
        H_i = entropy.unsqueeze(1).expand(-1, self.d) # [d, d]
        H_j = entropy.unsqueeze(0).expand(self.d, -1) # [d, d]
        
        entropy_penalty = F.relu(H_i - H_j) # Penalty when cause > effect
        
        # Weight by edge strength
        L_entropy = (A * entropy_penalty).sum() / (A.sum() + 1e-8)
        
        # 2. Residual independence test (optional, more expensive)
        L_residual = torch.tensor(0.0, device=device)
        if self.use_residual_test:
            indep_penalty = self.compute_residual_independence(X_flat, A)
            L_residual = (A * indep_penalty).sum() / (A.sum() + 1e-8)
        
        total_loss = self.lambda_orient * (L_entropy + 0.5 * L_residual)
        
        metrics = {
            "L_entropy": L_entropy.item(),
            "L_residual": L_residual.item() if isinstance(L_residual, torch.Tensor) else L_residual,
            "entropy_mean": entropy.mean().item(),
            "entropy_std": entropy.std().item(),
        }
        
        return total_loss, metrics


# =============================================================================
# 3. STRUCTURAL COUNTERFACTUAL VALIDATION
# =============================================================================

class EdgeNecessityValidator(nn.Module):
    """
    Counterfactual edge validation during training.
    
    For each edge (i->j), test: "If we remove this edge, how much worse 
    does prediction of X[j] become?"
    
    - True causal edges: Removal significantly hurts prediction
    - Spurious edges: Removal has little effect (info flows via other paths)
    
    We use this to create a necessity score for each edge, which can be used
    as a regularizer or for edge selection.
    """
    
    def __init__(
        self,
        d: int,
        n_edges_to_test: int = 5, # Test top-K edges per batch (expensive!)
        necessity_threshold: float = 0.1,
    ):
        super().__init__()
        self.d = d
        self.n_edges_to_test = n_edges_to_test
        self.necessity_threshold = necessity_threshold
        
    def compute_edge_necessity(
        self,
        X: torch.Tensor, # [B, T, d]
        A: torch.Tensor, # [d, d]
        decoder_fn, # Callable that takes (X, A) -> predictions
        top_k: int = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute necessity score for top edges.
        
        For each edge (i,j), compute:
        necessity[i,j] = ||pred(A) - pred(A without edge i,j)||_j
        
        where ||.||_j is the reconstruction error for node j specifically.
        
        Args:
            X: Input data
            A: Adjacency matrix
            decoder_fn: Function that takes A and returns predictions for X
            top_k: Number of top edges to test
            
        Returns:
            necessity_scores: [d, d] matrix of necessity scores
            metrics: Diagnostic info
        """
        device = A.device
        d = self.d
        top_k = top_k or self.n_edges_to_test
        
        # Get baseline prediction
        with torch.no_grad():
            base_pred = decoder_fn(A) # [B, T, d]
            base_error = (X - base_pred).pow(2).mean(dim=(0, 1)) # [d]
        
        # Find top-K edges by weight
        A_flat = A.clone()
        A_flat.fill_diagonal_(0)
        top_indices = torch.argsort(A_flat.flatten(), descending=True)[:top_k]
        
        necessity_scores = torch.zeros(d, d, device=device)
        
        for idx in top_indices:
            i = idx // d
            j = idx % d
            
            if A[i, j] > 0.01: # Only test meaningful edges
                # Remove edge
                A_ablated = A.clone()
                A_ablated[i, j] = 0
                
                with torch.no_grad():
                    ablated_pred = decoder_fn(A_ablated)
                    ablated_error = (X - ablated_pred).pow(2).mean(dim=(0, 1))
                
                # Necessity = how much worse prediction of j gets
                necessity_scores[i, j] = (ablated_error[j] - base_error[j]).clamp(min=0)
        
        # Normalize by baseline error
        necessity_scores = necessity_scores / (base_error.mean() + 1e-8)
        
        # Edges with low necessity are suspicious (spurious)
        tested_edges = (necessity_scores > 0).sum().item()
        necessary_edges = (necessity_scores > self.necessity_threshold).sum().item()
        
        metrics = {
            "tested_edges": tested_edges,
            "necessary_edges": necessary_edges,
            "necessity_ratio": necessary_edges / (tested_edges + 1e-8),
            "mean_necessity": necessity_scores.sum().item() / (tested_edges + 1e-8),
        }
        
        return necessity_scores, metrics
    
    def get_necessity_penalty(
        self,
        A: torch.Tensor,
        necessity_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Penalize edges that have high weight but low necessity.
        
        These are spurious edges: the model thinks they're important,
        but counterfactual testing shows they're not needed.
        """
        # Edges with A > threshold but necessity < threshold are spurious
        edge_strength = A.clone()
        edge_strength.fill_diagonal_(0)
        
        # Penalty for strong edges with low necessity
        # Only penalize edges we actually tested
        tested_mask = (necessity_scores > 0).float()
        low_necessity = (necessity_scores < self.necessity_threshold).float()
        
        penalty = (edge_strength * low_necessity * tested_mask).sum()
        penalty = penalty / (tested_mask.sum() + 1e-8)
        
        return penalty


# =============================================================================
# 4. ENVIRONMENT-SPECIFIC DECODERS
# =============================================================================

class EnvironmentSpecificDecoder(nn.Module):
    """
    Per-environment decoders with shared structure A.
    
    Each environment has its own decoder parameters, but they all
    use the same adjacency matrix A. This allows us to:
    
    1. Model different mechanisms per environment
    2. Compute mechanism invariance loss: ||f_e1(A,z) - f_e2(A,z)||
       on observational (non-intervened) nodes
    
    True causal edges have stable mechanisms; spurious edges don't.
    """
    
    def __init__(
        self,
        d: int,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        n_envs: int = 2,
        min_sigma: float = 0.01,
    ):
        super().__init__()
        self.d = d
        self.latent_dim = latent_dim
        self.n_envs = max(n_envs, 2) # At least 2 for comparison
        self.min_sigma = min_sigma
        
        # Shared signal projection (structure-dependent)
        self.signal_proj = nn.Linear(latent_dim, hidden_dim)
        
        # Per-environment mechanism networks
        self.env_mechanisms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2), # mu, log_sigma
            )
            for _ in range(self.n_envs)
        ])
        
        # Shared corruption pathway
        self.corrupt_proj = nn.Linear(latent_dim, hidden_dim // 2)
        self.corrupt_out = nn.Linear(hidden_dim // 2, 1)
        
    def forward(
        self,
        z_signal: torch.Tensor, # [B, T, d, L]
        z_corrupt: torch.Tensor, # [B, T, d, L]
        A: torch.Tensor, # [d, d]
        regime: torch.Tensor, # [B] regime indices
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Forward pass with per-environment decoding.
        
        Returns:
            mu: [B, T, d] mean prediction
            sigma: [B, T, d] scale prediction
            per_env_mu: Dict[env_idx -> mu_tensor] for mechanism comparison
        """
        has_time = z_signal.dim() == 4
        
        if has_time:
            B, T, d, L = z_signal.shape
        else:
            B, d, L = z_signal.shape
            z_signal = z_signal.unsqueeze(1)
            z_corrupt = z_corrupt.unsqueeze(1)
            T = 1
        
        # Message passing with structure A
        z_s_flat = z_signal.reshape(B * T, d, L)
        z_agg = torch.bmm(
            A.unsqueeze(0).expand(B * T, -1, -1).transpose(-1, -2),
            z_s_flat
        )
        
        h_signal = self.signal_proj(z_agg) # [BT, d, H]
        
        # Per-sample predictions using appropriate environment decoder
        mu_all = torch.zeros(B, T, d, device=z_signal.device)
        sigma_all = torch.zeros(B, T, d, device=z_signal.device)
        
        # Store per-environment outputs for mechanism invariance
        per_env_mu = {}
        
        # Group samples by regime
        unique_regimes = torch.unique(regime)
        
        for e in unique_regimes:
            e_idx = e.item()
            if e_idx >= self.n_envs:
                e_idx = 0 # Fallback to first env
                
            mask = (regime == e)
            indices = mask.nonzero(as_tuple=True)[0]
            
            if len(indices) == 0:
                continue
            
            # Get signal features for this environment's samples
            h_env = h_signal.reshape(B, T, d, -1)[mask] # [n_e, T, d, H]
            h_env_flat = h_env.reshape(-1, d, h_env.shape[-1]) # [n_e*T, d, H]
            
            # Apply environment-specific mechanism
            out = self.env_mechanisms[e_idx](h_env_flat) # [n_e*T, d, 2]
            
            mu_env = out[..., 0]
            sigma_env = F.softplus(out[..., 1]) + self.min_sigma
            
            mu_env = mu_env.reshape(len(indices), T, d)
            sigma_env = sigma_env.reshape(len(indices), T, d)
            
            mu_all[mask] = mu_env
            sigma_all[mask] = sigma_env
            
            per_env_mu[e_idx] = mu_env
        
        # Add corruption contribution (shared across envs)
        z_c_flat = z_corrupt.reshape(B * T, d, L)
        h_corrupt = self.corrupt_proj(z_c_flat)
        corrupt_contrib = self.corrupt_out(F.relu(h_corrupt)).squeeze(-1)
        corrupt_contrib = corrupt_contrib.reshape(B, T, d)
        
        mu_all = mu_all + corrupt_contrib
        
        if not has_time:
            mu_all = mu_all.squeeze(1)
            sigma_all = sigma_all.squeeze(1)
        
        return mu_all, sigma_all, per_env_mu
    
    def compute_mechanism_invariance(
        self,
        z_signal: torch.Tensor,
        A: torch.Tensor,
        env_pairs: List[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Compute mechanism invariance loss across environments.
        
        L_mech = sum_{e1, e2} ||f_e1(A, z) - f_e2(A, z)||^2
        
        This measures how much the mechanism differs between environments.
        True causal mechanisms should be stable (low L_mech).
        """
        if env_pairs is None:
            # Compare all pairs
            env_pairs = [(i, j) for i in range(self.n_envs) 
                         for j in range(i+1, self.n_envs)]
        
        if len(env_pairs) == 0:
            return torch.tensor(0.0, device=z_signal.device)
        
        has_time = z_signal.dim() == 4
        
        if has_time:
            B, T, d, L = z_signal.shape
        else:
            B, d, L = z_signal.shape
            z_signal = z_signal.unsqueeze(1)
            T = 1
        
        # Message passing with structure A
        z_s_flat = z_signal.reshape(B * T, d, L)
        z_agg = torch.bmm(
            A.unsqueeze(0).expand(B * T, -1, -1).transpose(-1, -2),
            z_s_flat
        )
        h_signal = self.signal_proj(z_agg) # [BT, d, H]
        
        # Get outputs from each environment decoder
        env_outputs = []
        for e_idx in range(self.n_envs):
            out = self.env_mechanisms[e_idx](h_signal)
            mu = out[..., 0] # [BT, d]
            env_outputs.append(mu)
        
        # Compute pairwise mechanism differences
        loss = torch.tensor(0.0, device=z_signal.device)
        for e1, e2 in env_pairs:
            if e1 < len(env_outputs) and e2 < len(env_outputs):
                diff = (env_outputs[e1] - env_outputs[e2]).pow(2).mean()
                loss = loss + diff
        
        loss = loss / len(env_pairs)
        return loss


# =============================================================================
# INTEGRATED CAUSAL PRIOR LOSS
# =============================================================================

class CausalPriorLoss(nn.Module):
    """
    Unified causal prior loss combining all four innovations.
    
    L_causal = λ_int * L_intervention 
             + λ_orient * L_orientation
             + λ_nec * L_necessity 
             + λ_mech * L_mechanism
    """
    
    def __init__(
        self,
        d: int,
        n_regimes: int = 1,
        lambda_intervention: float = 0.1,
        lambda_orientation: float = 0.1,
        lambda_necessity: float = 0.05, # Less frequent, expensive
        lambda_mechanism: float = 0.1,
    ):
        super().__init__()
        self.d = d
        self.n_regimes = n_regimes
        
        # Weights
        self.lambda_int = lambda_intervention
        self.lambda_orient = lambda_orientation
        self.lambda_nec = lambda_necessity
        self.lambda_mech = lambda_mechanism
        
        # Components
        self.intervention = InterventionAwareStructure(d, n_regimes)
        self.orientation = OrientationPenalty(d, lambda_orient=1.0) # Scaled by wrapper
        self.necessity = EdgeNecessityValidator(d, n_edges_to_test=5)
        
    def forward(
        self,
        X: torch.Tensor,
        A: torch.Tensor,
        A_per_env: Optional[torch.Tensor] = None,
        regime: Optional[torch.Tensor] = None,
        decoder_fn=None,
        env_decoder: Optional[EnvironmentSpecificDecoder] = None,
        z_signal: Optional[torch.Tensor] = None,
        compute_necessity: bool = False, # Expensive, do periodically
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total causal prior loss.
        
        Args:
            X: Input data [B, T, d]
            A: Base adjacency [d, d]
            A_per_env: Per-environment adjacency [B, d, d] or [n_envs, d, d]
            regime: Regime indices [B]
            decoder_fn: Callable for counterfactual testing
            env_decoder: Environment-specific decoder for mechanism invariance
            z_signal: Latent signal for mechanism invariance
            compute_necessity: Whether to run expensive counterfactual test
            
        Returns:
            loss: Total causal prior loss
            metrics: Diagnostic info
        """
        device = A.device
        metrics = {}
        total_loss = torch.tensor(0.0, device=device)
        
        # 1. Intervention loss (if multi-regime)
        L_int = torch.tensor(0.0, device=device)
        if A_per_env is not None and regime is not None and self.n_regimes > 1:
            L_int = self.intervention.compute_intervention_loss(A_per_env, regime)
            total_loss = total_loss + self.lambda_int * L_int
        metrics["L_intervention"] = L_int.item() if isinstance(L_int, torch.Tensor) else L_int
        
        # 2. Orientation loss
        L_orient, orient_metrics = self.orientation(X, A)
        total_loss = total_loss + self.lambda_orient * L_orient
        metrics["L_orientation"] = L_orient.item()
        metrics.update({f"orient_{k}": v for k, v in orient_metrics.items()})
        
        # 3. Necessity loss (expensive - do periodically)
        L_nec = torch.tensor(0.0, device=device)
        if compute_necessity and decoder_fn is not None:
            necessity_scores, nec_metrics = self.necessity.compute_edge_necessity(
                X, A, decoder_fn, top_k=5
            )
            L_nec = self.necessity.get_necessity_penalty(A, necessity_scores)
            total_loss = total_loss + self.lambda_nec * L_nec
            metrics.update({f"nec_{k}": v for k, v in nec_metrics.items()})
        metrics["L_necessity"] = L_nec.item() if isinstance(L_nec, torch.Tensor) else L_nec
        
        # 4. Mechanism invariance (if env-specific decoder)
        L_mech = torch.tensor(0.0, device=device)
        if env_decoder is not None and z_signal is not None:
            L_mech = env_decoder.compute_mechanism_invariance(z_signal, A)
            total_loss = total_loss + self.lambda_mech * L_mech
        metrics["L_mechanism"] = L_mech.item() if isinstance(L_mech, torch.Tensor) else L_mech
        
        metrics["L_causal_total"] = total_loss.item()
        
        return total_loss, metrics
    
    def set_intervention_targets(self, targets: Dict[int, List[int]]):
        """Pass through to intervention module."""
        self.intervention.set_intervention_targets(targets)


# =============================================================================
# CORRELATION VS CAUSATION DIAGNOSTIC
# =============================================================================

def diagnose_correlation_vs_causation(
    A_pred: torch.Tensor,
    A_true: torch.Tensor,
    X: torch.Tensor,
    k: int = None,
) -> Dict[str, float]:
    """
    Diagnostic to understand if model is learning correlation or causation.
    
    Computes:
    1. Overlap between top predicted edges and top correlated pairs
    2. Overlap between top predicted edges and true causal edges
    3. Correlation strength of predicted edges vs true edges
    
    Args:
        A_pred: Predicted adjacency [d, d]
        A_true: True adjacency [d, d]
        X: Data [N, d] or [B, T, d]
        k: Number of edges to consider
        
    Returns:
        Diagnostic metrics
    """
    # Flatten X
    if X.dim() == 3:
        X_flat = X.reshape(-1, X.shape[-1])
    else:
        X_flat = X
    
    d = A_pred.shape[0]
    k = k or int(A_true.sum().item())
    
    device = A_pred.device
    
    # Compute correlation matrix
    X_np = X_flat.detach().cpu()
    X_centered = X_np - X_np.mean(dim=0, keepdim=True)
    X_std = X_centered.std(dim=0, keepdim=True) + 1e-8
    X_norm = X_centered / X_std
    corr = (X_norm.T @ X_norm) / X_np.shape[0]
    corr = corr.abs()
    corr.fill_diagonal_(0)
    
    # Get indices
    A_pred_np = A_pred.detach().cpu().numpy().copy()
    A_true_np = A_true.detach().cpu().numpy().copy()
    corr_np = corr.numpy().copy()
    
    import numpy as np
    np.fill_diagonal(A_pred_np, 0)
    np.fill_diagonal(A_true_np, 0)
    
    # Top-K by prediction
    pred_top_k = set(np.argsort(A_pred_np.flatten())[-k:].tolist())
    
    # Top-K by correlation
    corr_top_k = set(np.argsort(corr_np.flatten())[-k:].tolist())
    
    # True edges
    true_edges = set(np.where(A_true_np.flatten() > 0.5)[0].tolist())
    
    # Compute overlaps
    pred_corr_overlap = len(pred_top_k & corr_top_k) / k if k > 0 else 0
    pred_true_overlap = len(pred_top_k & true_edges) / k if k > 0 else 0
    corr_true_overlap = len(corr_top_k & true_edges) / k if k > 0 else 0
    
    # Average correlation of predicted edges vs true edges
    pred_edge_corrs = [corr_np.flatten()[i] for i in pred_top_k]
    true_edge_corrs = [corr_np.flatten()[i] for i in true_edges]
    
    avg_pred_corr = np.mean(pred_edge_corrs) if pred_edge_corrs else 0
    avg_true_corr = np.mean(true_edge_corrs) if true_edge_corrs else 0
    
    return {
        "pred_corr_overlap": pred_corr_overlap, # High = learning correlation
        "pred_true_overlap": pred_true_overlap, # High = learning causation
        "corr_true_overlap": corr_true_overlap, # Baseline: how correlated are true edges?
        "avg_pred_edge_corr": float(avg_pred_corr),
        "avg_true_edge_corr": float(avg_true_corr),
        "diagnosis": "correlation" if pred_corr_overlap > pred_true_overlap else "causation",
    }
