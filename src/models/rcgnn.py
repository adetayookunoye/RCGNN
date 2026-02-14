"""
RC-GNN: Robust Causal Graph Neural Networks under Compound Sensor Corruptions

Key innovations:
1. Disentangled Encoder: Separates signal from corruption (z_signal, z_corrupt)
2. Causal Graph Learner: Learns adjacency A with per-environment deltas
3. Structure-level Invariance: IRM-style penalty for cross-environment stability
4. Robust Likelihood: Student-t distribution for outlier resistance
5. Missingness Modeling: Learns P(M|X*) for true MNAR (X* = imputed value)
6. Causal Priors: Intervention, orientation, necessity, mechanism invariance

This is the canonical RC-GNN implementation combining all improvements.
"""

import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Tuple, Optional, List

from .causal_priors import (
    CausalPriorLoss,
    EnvironmentSpecificDecoder,
    diagnose_correlation_vs_causation,
)
from .invariance import IRMStructureInvariance


# =============================================================================
# NOTEARS Acyclicity
# =============================================================================

def notears_acyclicity(A: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Compute NOTEARS acyclicity constraint: h(A) = tr(exp(A∘A)) - d"""
    d = A.shape[-1]
    A_sq = A * A
    
    if d <= 50:
        exp_A = torch.linalg.matrix_exp(A_sq)
    else:
        exp_A = torch.eye(d, device=A.device, dtype=A.dtype)
        A_power = torch.eye(d, device=A.device, dtype=A.dtype)
        for k in range(1, 12):
            A_power = A_power @ A_sq / k
            exp_A = exp_A + A_power
    
    h = torch.trace(exp_A) - d
    return h / d if normalize else h


# =============================================================================
# HSIC for Disentanglement
# =============================================================================

def rbf_kernel(x: torch.Tensor, sigma: Optional[float] = None) -> torch.Tensor:
    """RBF kernel with median heuristic."""
    if x.dim() == 1:
        x = x.unsqueeze(1)
    if x.dim() > 2:
        x = x.reshape(x.shape[0], -1)
    
    dists = torch.cdist(x, x, p=2) ** 2
    
    if sigma is None:
        mask = dists > 0
        if mask.any():
            sigma = torch.sqrt(torch.median(dists[mask]))
        else:
            sigma = torch.tensor(1.0, device=x.device)
    
    return torch.exp(-dists / (2 * sigma ** 2 + 1e-8))


def hsic_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """HSIC independence penalty between z1 and z2."""
    z1_flat = z1.reshape(-1, z1.shape[-1]) if z1.dim() > 2 else z1.reshape(-1, 1)
    z2_flat = z2.reshape(-1, z2.shape[-1]) if z2.dim() > 2 else z2.reshape(-1, 1)
    
    n = z1_flat.shape[0]
    if n > 1000:
        idx = torch.randperm(n, device=z1.device)[:1000]
        z1_flat = z1_flat[idx]
        z2_flat = z2_flat[idx]
        n = 1000
    
    if n < 5:
        return torch.tensor(0.0, device=z1.device)
    
    K = rbf_kernel(z1_flat)
    L = rbf_kernel(z2_flat)
    H = torch.eye(n, device=z1.device) - (1.0 / n) * torch.ones(n, n, device=z1.device)
    
    return torch.trace(K @ H @ L @ H) / ((n - 1) ** 2 + 1e-8)


# =============================================================================
# Disentangled Encoder
# =============================================================================

class DisentangledEncoder(nn.Module):
    """Two-pathway encoder: X -> (z_signal, z_corrupt)"""
    
    def __init__(
        self,
        input_dim: int = 1,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        trunk = []
        trunk.append(nn.Linear(input_dim, hidden_dim))
        trunk.append(nn.LayerNorm(hidden_dim))
        trunk.append(nn.ReLU())
        trunk.append(nn.Dropout(dropout))
        
        for _ in range(n_layers - 1):
            trunk.append(nn.Linear(hidden_dim, hidden_dim))
            trunk.append(nn.LayerNorm(hidden_dim))
            trunk.append(nn.ReLU())
            trunk.append(nn.Dropout(dropout))
        
        self.trunk = nn.Sequential(*trunk)
        
        self.signal_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        
        self.corrupt_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, d] input
        Returns:
            z_signal: [B, T, d, latent_dim]
            z_corrupt: [B, T, d, latent_dim]
        """
        has_time = x.dim() == 3
        if has_time:
            B, T, d = x.shape
        else:
            B, d = x.shape
            x = x.unsqueeze(1)
            T = 1
        
        x_flat = x.reshape(B * T * d, self.input_dim)
        h = self.trunk(x_flat)
        
        z_s = self.signal_head(h)
        z_c = self.corrupt_head(h)
        
        z_s = z_s.reshape(B, T, d, self.latent_dim)
        z_c = z_c.reshape(B, T, d, self.latent_dim)
        
        if not has_time:
            z_s = z_s.squeeze(1)
            z_c = z_c.squeeze(1)
        
        return z_s, z_c


# =============================================================================
# Causal Graph Learner with Per-Environment Deltas
# =============================================================================

class CausalGraphLearner(nn.Module):
    """
    Learns adjacency matrix A with per-environment deltas.
    
    V8.33: DUAL-PARAMETER architecture.
    - W_mag [d,d]: symmetric, controls edge EXISTENCE (corr-initialized)
    - W_dir [d,d]: antisymmetric, controls edge DIRECTION (starts at 0)
    
    A_ij = σ(W_mag_ij) * σ(W_dir_ij / τ)
    A_ji = σ(W_mag_ij) * (1 - σ(W_dir_ij / τ))
    
    Key insight: Structure and direction have SEPARATE parameters → gradients
    from reconstruction update W_mag without disturbing direction, and
    direction losses (L_dir_dec, L_excl) update W_dir without disturbing structure.
    """
    
    def __init__(
        self,
        d: int,
        hidden_dim: int = 64,
        n_regimes: int = 1,
        init_scale: float = 0.01,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.d = d
        self.n_regimes = max(n_regimes, 1)
        self.temperature = temperature
        
        # V8.33: Separate parameters for structure and direction
        # W_mag: symmetric edge-existence logits (corr-initialized later)
        self.W_mag = nn.Parameter(torch.randn(d, d) * init_scale)
        # W_dir: antisymmetric direction logits (starts at 0 → dir=0.5)
        self.W_dir = nn.Parameter(torch.zeros(d, d))
        
        # Per-environment deltas (applied to W_mag only — structure varies by env)
        if n_regimes > 1:
            self.env_deltas = nn.Parameter(torch.zeros(n_regimes, d, d))
        else:
            self.register_buffer("env_deltas", torch.zeros(1, d, d))
        
        # For temperature annealing
        self.register_buffer("current_temp", torch.tensor(temperature))
    
    def _get_sym_antisym(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project raw parameters to enforced symmetry/antisymmetry.
        
        Returns:
            Wm: [d,d] symmetric magnitude logits, diag=0
            Wd: [d,d] antisymmetric direction logits, diag=0
        
        Guarantees:
            Wm[i,j] == Wm[j,i]  →  mag_ij == mag_ji
            Wd[i,j] == -Wd[j,i] →  p_ij + p_ji == 1
            →  A_ij + A_ji == mag_ij  (direction cannot change edge mass)
        """
        diag = torch.eye(self.d, device=self.W_mag.device)
        Wm = 0.5 * (self.W_mag + self.W_mag.T) * (1 - diag)
        Wd = 0.5 * (self.W_dir - self.W_dir.T) * (1 - diag)
        return Wm, Wd
    
    def forward(
        self,
        env_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get adjacency matrix.
        
        Args:
            env_idx: [B] environment indices. If None, return base A.
            
        Returns:
            A: [d, d] or [B, d, d] adjacency matrix
            W: [d, d] or [B, d, d] logits (symmetrized W_mag)
        """
        tau = self.current_temp
        antisym = getattr(self, '_antisymmetric', False)
        Wm, Wd = self._get_sym_antisym()
        
        if env_idx is None or self.n_regimes == 1:
            if antisym:
                A = self._antisymmetric_adjacency(Wm, Wd, tau)
                return A, Wm
            logits = Wm / tau
            A = torch.sigmoid(logits)
            A = A * (1 - torch.eye(self.d, device=A.device)) # Zero diagonal
            return A, Wm
        
        # Per-environment adjacency (deltas applied to Wm, then re-symmetrized)
        B = env_idx.shape[0]
        device = Wm.device
        diag_mask = torch.eye(self.d, device=device)
        
        Wm_batch = Wm.unsqueeze(0).expand(B, -1, -1).clone()
        
        for b in range(B):
            e = env_idx[b].item()
            if e < self.n_regimes:
                Wm_e = Wm + self.env_deltas[e]
                # Re-symmetrize after adding env delta
                Wm_batch[b] = 0.5 * (Wm_e + Wm_e.transpose(-1, -2)) * (1 - diag_mask)
        
        if antisym:
            # V8.33: Batch dual-parameter antisymmetric
            mag_batch = torch.sigmoid(Wm_batch)
            # Wd is shared across envs (direction doesn't vary by env)
            Wd_batch = Wd.unsqueeze(0).expand(B, -1, -1)
            dir_batch = torch.sigmoid(Wd_batch / tau)
            A = mag_batch * dir_batch
            batch_diag = diag_mask.unsqueeze(0).expand(B, -1, -1)
            A = A * (1 - batch_diag)
            return A, Wm_batch
        
        logits = Wm_batch / tau
        A = torch.sigmoid(logits)
        
        # Zero diagonal
        batch_diag = diag_mask.unsqueeze(0).expand(B, -1, -1)
        A = A * (1 - batch_diag)
        
        return A, Wm_batch
    
    def get_mean_adjacency(self) -> torch.Tensor:
        """Get base adjacency (for evaluation)."""
        tau = self.current_temp
        antisym = getattr(self, '_antisymmetric', False)
        Wm, Wd = self._get_sym_antisym()
        if antisym:
            return self._antisymmetric_adjacency(Wm, Wd, tau)
        logits = Wm / tau
        A = torch.sigmoid(logits)
        A = A * (1 - torch.eye(self.d, device=A.device))
        return A
    
    def _antisymmetric_adjacency(self, W_mag: torch.Tensor, W_dir: torch.Tensor, tau: float) -> torch.Tensor:
        """
        V8.33: DUAL-PARAMETER antisymmetric adjacency.
        
        Two SEPARATE parameter tensors, PROJECTED before use:
          Wm = (W_mag + W_mag.T)/2   # ENFORCED symmetric, diag=0
          Wd = (W_dir - W_dir.T)/2   # ENFORCED antisymmetric, diag=0
        
          mag = sigmoid(Wm)            # edge gate: symmetric → mag_ij == mag_ji
          p   = sigmoid(Wd / τ)        # direction: antisym → p_ij + p_ji == 1
          A_ij = mag_ij * p_ij
          A_ji = mag_ij * (1 - p_ij)   # by complement property
        
        GUARANTEED properties (by construction, not by loss):
          - A_ij + A_ji = mag_ij       (direction cannot change edge mass!)
          - A.sum() = mag.sum()         (L_budget gradient → W_mag only)
          - p_ij + p_ji = 1            (exactly one direction wins per pair)
          - dL_budget/dW_dir = 0        (decoupling is structural, not hoped-for)
        
        Key improvement over V8.30-V8.32:
          - V8.30-V8.32: mag and dir derived from SAME W or unconstrained params.
          - V8.33: Separate params WITH structural projections.
            No interference possible, by linear algebra, not by loss tuning.
        """
        d = W_mag.shape[0]
        device = W_mag.device
        
        # Edge magnitude from W_mag (symmetric — structure)
        mag = torch.sigmoid(W_mag)
        
        # Direction from W_dir (antisymmetric — direction)
        direction = torch.sigmoid(W_dir / tau)
        
        # A_ij = mag * dir, A_ji = mag * (1-dir)
        A = mag * direction
        
        # Zero diagonal
        A = A * (1 - torch.eye(d, device=device))
        return A
    
    def set_antisymmetric(self, enabled: bool = True):
        """Enable/disable V8.33 dual-parameter antisymmetric mode."""
        self._antisymmetric = enabled
    
    @property
    def W_adj(self) -> torch.Tensor:
        """Backward-compat: returns symmetrized Wm for code that reads W_adj.
        
        V8.33: Returns (W_mag+W_mag.T)/2 with diag=0 — the projected
        symmetric magnitude logits. Used by training script for
        hard prune, edge swap, ranking margin, diagnostics.
        """
        Wm, _ = self._get_sym_antisym()
        return Wm
    
    def get_direction_map(self) -> torch.Tensor:
        """V8.33: Return p = σ(Wd/τ) for direction decisiveness penalty.
        
        Returns [d,d] tensor where p_ij ∈ (0,1). At 0.5 = undecided,
        near 0 or 1 = decided. Uses projected antisymmetric Wd.
        Guarantees p_ij + p_ji = 1.
        """
        tau = self.current_temp
        _, Wd = self._get_sym_antisym()
        return torch.sigmoid(Wd / tau)
    
    def get_temperature(self) -> float:
        return self.current_temp.item()
    
    def set_temperature(self, temp: float):
        self.current_temp.fill_(temp)
    
    def increment_step(self):
        """Hook for temperature annealing schedules. Override in subclass if needed."""
        pass
    
    def topk_project(self, k: int, use_logits: bool = True) -> torch.Tensor:
        """
        Hard Top-K projection: return binary mask with exactly K edges.
        
        V8.33: Uses W_mag for edge ranking (structure parameter only).
        V8.34: Selects from upper triangle (W_mag is symmetric),
               then mirrors to full matrix. Ensures K undirected edges
               survive, not K/2.
        
        Args:
            k: Number of edges to keep (typically target_edges = true_edges)
            use_logits: If True (recommended), rank by W_mag magnitude.
            
        Returns:
            A_proj: [d, d] binary adjacency with exactly 2K entries (K pairs)
        """
        device = self.W_mag.device
        diag_mask = torch.eye(self.d, device=device)
        
        if use_logits:
            # Rank edges by symmetrized Wm (structure parameter)
            Wm, _ = self._get_sym_antisym()
        else:
            # Fallback: rank by symmetric magnitude of A
            A = self.get_mean_adjacency()
            Wm = torch.maximum(A, A.T)  # symmetric strength
        
        # V8.34: Select from UPPER TRIANGLE only (Wm is symmetric)
        W_upper = torch.triu(Wm, diagonal=1)
        flat = W_upper.flatten()
        n_upper = self.d * (self.d - 1) // 2
        k = min(k, n_upper)
        
        if k <= 0:
            return torch.zeros(self.d, self.d, device=device)
        
        _, topk_idx = torch.topk(flat, k)
        
        # Create mask in upper triangle, then mirror
        mask_flat = torch.zeros_like(flat)
        mask_flat[topk_idx] = 1.0
        mask = mask_flat.reshape(self.d, self.d)
        mask = (mask + mask.T).clamp(max=1.0)  # mirror to full matrix
        
        return mask
    
    def get_logits(self) -> torch.Tensor:
        """Get symmetrized Wm logits for sparsity penalty.
        V8.33: Returns projected (W_mag+W_mag.T)/2 with diag=0.
        """
        Wm, _ = self._get_sym_antisym()
        return Wm


# =============================================================================
# Missingness Head
# =============================================================================

class MissingnessHead(nn.Module):
    """
    Predicts P(M|X*) for MNAR modeling.
    
    Key insight: MNAR means missingness depends on the *unobserved* value.
    We use the imputed value X* = mu (from decoder) as proxy for the true value.
    This is the standard latent variable approach to MNAR.
    """
    
    def __init__(self, d: int, hidden_dim: int = 64, n_regimes: int = 1):
        super().__init__()
        self.d = d
        self.n_regimes = n_regimes
        
        input_dim = d + (n_regimes if n_regimes > 1 else 0)
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d),
        )
    
    def forward(
        self,
        X_star: torch.Tensor,
        regime: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns logits for P(M=1|X*).
        
        Args:
            X_star: Imputed values (mu from decoder), not observed X.
                    This enables true MNAR modeling.
            regime: Environment indices for regime-conditional missingness.
        """
        has_time = X_star.dim() == 3
        if has_time:
            B, T, d = X_star.shape
            X_flat = X_star.reshape(B * T, d)
        else:
            B, d = X_star.shape
            X_flat = X_star
            T = 1
        
        if regime is not None and self.n_regimes > 1:
            regime_expanded = regime.unsqueeze(1).expand(-1, T).reshape(-1)
            regime_onehot = F.one_hot(regime_expanded, self.n_regimes).float()
            X_flat = torch.cat([X_flat, regime_onehot], dim=-1)
        
        logits = self.net(X_flat)
        
        if has_time:
            logits = logits.reshape(B, T, d)
        
        return logits


# =============================================================================
# Student-t NLL
# =============================================================================

def student_t_nll(
    x: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    nu: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Negative log-likelihood of Student-t distribution."""
    z = (x - mu) / sigma
    
    nll = (
        torch.lgamma((nu + 1) / 2)
        - torch.lgamma(nu / 2)
        - 0.5 * torch.log(nu * math.pi)
        - torch.log(sigma)
        - ((nu + 1) / 2) * torch.log(1 + z ** 2 / nu)
    )
    
    nll = -nll
    
    if mask is not None:
        nll = nll * mask
        return nll.sum() / (mask.sum() + 1e-8)
    return nll.mean()


# =============================================================================
# Heteroscedastic Decoder with MNAR-aware Variance
# =============================================================================

class HeteroscedasticDecoder(nn.Module):
    """
    Decoder with learned heteroscedastic variance.
    
    GUARDRAIL: Variance depends on latent state + regime, NOT on M_probs directly.
    This prevents the model from gaming both selection and variance to reduce loss.
    
    The selection model (IPW) handles MNAR correction; variance here captures
    aleatoric uncertainty from the latent state itself.
    
    Variance model:
        σ²_total = σ²_signal(h) + σ²_bias
    
    Where h = f(z_signal, z_corrupt, A) is the latent representation.
    """
    
    def __init__(
        self,
        d: int,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        min_sigma: float = 0.01,
        max_sigma: float = 5.0, # Cap variance to prevent inflation gaming
        min_nu: float = 2.1,
        corrupt_dropout: float = 0.5,
    ):
        super().__init__()
        self.d = d
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.min_nu = min_nu
        self.corrupt_dropout = corrupt_dropout
        
        # Signal projection (graph-aggregated)
        self.signal_proj = nn.Linear(latent_dim, hidden_dim)
        
        # Corruption projection
        self.corrupt_proj = nn.Linear(latent_dim, hidden_dim)
        
        # Mean prediction head
        self.mu_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Signal variance head (heteroscedastic, from latent state only)
        # NO M_probs input - prevents gaming
        self.signal_var_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )
        
        # Bias variance (per-variable, learned)
        self.bias_var = nn.Parameter(torch.ones(d) * 0.1)
        
        # Degrees of freedom (nu) for Student-t
        self.nu_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )
    
    def forward(
        self,
        z_signal: torch.Tensor,
        z_corrupt: torch.Tensor,
        A: torch.Tensor,
        regime: Optional[torch.Tensor] = None,
        M_probs: Optional[torch.Tensor] = None, # Kept for interface, but NOT used
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass with heteroscedastic variance (NO M_probs dependence).
        
        GUARDRAIL: M_probs is ignored to prevent shortcut learning.
        Variance comes purely from the latent state.
        """
        has_time = z_signal.dim() == 4
        if has_time:
            B, T, d, L = z_signal.shape
        else:
            B, d, L = z_signal.shape
            z_signal = z_signal.unsqueeze(1)
            z_corrupt = z_corrupt.unsqueeze(1)
            T = 1
        
        # Graph aggregation: A^T @ z_signal
        z_s_flat = z_signal.reshape(B * T, d, L)
        z_agg = torch.bmm(
            A.unsqueeze(0).expand(B * T, -1, -1).transpose(-1, -2),
            z_s_flat
        ) # [B*T, d, L]
        
        h_signal = self.signal_proj(z_agg) # [B*T, d, hidden]
        
        # Corruption encoding
        z_c_flat = z_corrupt.reshape(B * T, d, L)
        h_corrupt = self.corrupt_proj(z_c_flat) # [B*T, d, hidden]
        
        # Dropout on corruption (encourage signal pathway)
        if self.training and self.corrupt_dropout > 0:
            drop_mask = (torch.rand(1, device=h_corrupt.device) > self.corrupt_dropout).float()
            h_corrupt = h_corrupt * drop_mask
        
        # Combined hidden state (no M_probs!)
        h_combined = torch.cat([h_signal, h_corrupt], dim=-1) # [B*T, d, 2*hidden]
        
        # =====================================================================
        # 1. Mean prediction
        # =====================================================================
        mu = self.mu_head(h_combined).squeeze(-1) # [B*T, d]
        
        # =====================================================================
        # 2. Signal variance (from latent state only, NOT M_probs)
        # =====================================================================
        signal_var = self.signal_var_head(h_combined).squeeze(-1) # [B*T, d]
        signal_var = signal_var.clamp_min(self.min_sigma ** 2)
        
        # =====================================================================
        # 3. Add bias variance
        # =====================================================================
        bias_var = F.softplus(self.bias_var).clamp_min(self.min_sigma ** 2)
        bias_var = bias_var.unsqueeze(0).expand(B * T, -1) # [B*T, d]
        
        total_var = signal_var + bias_var
        
        # GUARDRAIL: Cap sigma to prevent variance inflation gaming
        sigma = torch.sqrt(total_var).clamp(self.min_sigma, self.max_sigma)
        
        # =====================================================================
        # 4. Degrees of freedom (nu) for Student-t
        # =====================================================================
        nu = self.nu_head(h_combined).squeeze(-1) + self.min_nu # [B*T, d]
        
        # Reshape to [B, T, d]
        mu = mu.reshape(B, T, d)
        sigma = sigma.reshape(B, T, d)
        nu = nu.reshape(B, T, d)
        
        # Variance components for diagnostics
        var_components = {
            'signal_var': signal_var.reshape(B, T, d),
            'bias_var': bias_var.reshape(B, T, d),
            'total_var': total_var.reshape(B, T, d),
        }
        
        if not has_time:
            mu = mu.squeeze(1)
            sigma = sigma.squeeze(1)
            nu = nu.squeeze(1)
            var_components = {k: v.squeeze(1) for k, v in var_components.items()}
        
        return mu, sigma, nu, var_components


# =============================================================================
# RC-GNN Complete Model
# =============================================================================

class RCGNN(nn.Module):
    """
    RC-GNN: Robust Causal Graph Neural Network (V8.26)
    
    Complete model for causal discovery under compound sensor corruptions.
    
    ADJACENCY CONVENTION (V8.26 FIX):
    ---------------------------------
    UNIFIED CAUSAL: A[i,j] = "i causes j" everywhere.
    
    The decoder computes z_agg = A^T @ z_signal, so:
      z_agg[j] = sum_i A[i,j] * z_signal[i]
    Gradient descent pushes A[i,j] large when node i's signal
    helps reconstruct node j, i.e. when i→j in the true DAG.
    Therefore A[i,j] > 0 ⟺ i causes j — CAUSAL convention.
    
    A_true from data generators also uses A_true[i,j]=1 ⟹ i→j.
    
    NO TRANSPOSE NEEDED at any boundary. to_causal_convention()
    is now an identity function (kept for backward compatibility).
    
    Key components:
    1. DisentangledEncoder: X -> (z_signal, z_corrupt)
    2. CausalGraphLearner: learns A with per-environment deltas
    3. MissingnessHead: P(M|X*) for true MNAR (X* = mu from decoder)
    4. Robust Decoder: Student-t likelihood with heteroscedastic variance
    5. CausalPriorLoss: intervention/orientation/necessity/mechanism
    6. V8.24 ICP + anti-correlation losses for causal edge identity
    """
    
    VERSION = "8.27"
    
    @staticmethod
    def to_causal_convention(A: torch.Tensor) -> torch.Tensor:
        """Identity — model already outputs causal convention.
        
        V8.26 FIX: The decoder computes A^T @ z_signal, so gradient
        pushes A[i,j] large when i→j.  This IS causal convention.
        No transpose needed.  Kept for backward compatibility.
        """
        return A
    
    def __init__(
        self,
        d: int,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        n_regimes: int = 1,
        target_edges: int = 13,
        # Standard loss weights
        lambda_recon: float = 1.0,
        lambda_miss: float = 0.5,
        lambda_hsic: float = 0.1,
        lambda_acyclic: float = 1.0,
        lambda_sparse: float = 0.01,
        lambda_inv: float = 0.1,
        lambda_budget: float = 0.03,
        # Causal prior weights
        lambda_causal: float = 0.2,
        lambda_intervention: float = 0.1,
        lambda_orientation: float = 0.1,
        lambda_necessity: float = 0.05,
        lambda_mechanism: float = 0.1,
        # GUARDRAIL: Variance inflation penalty
        lambda_var_penalty: float = 0.01, # beta * mean(log(var))
        # Options
        use_env_specific_decoder: bool = True,
        mnar_detach: bool = True, # Detach mu for stable MNAR training
        # Selection model options (with guardrails)
        use_ipw: bool = True, # Inverse probability weighting for MNAR
        ipw_clip: float = 5.0, # GUARDRAIL: tighter clip to prevent explosion
        ipw_start_epoch: float = 0.3, # GUARDRAIL: Phase training - start IPW at 30%
    ):
        super().__init__()
        self.d = d
        self.n_regimes = n_regimes
        self.target_edges = target_edges
        self.mnar_detach = mnar_detach
        self.use_ipw = use_ipw
        self.ipw_clip = ipw_clip
        self.ipw_start_epoch = ipw_start_epoch # Fraction of total_epochs
        
        # Standard loss weights
        self.lambda_recon = lambda_recon
        self.lambda_miss = lambda_miss
        self.lambda_hsic = lambda_hsic
        self.lambda_acyclic = lambda_acyclic
        self.lambda_sparse = lambda_sparse
        self.lambda_budget = lambda_budget
        self.lambda_inv = lambda_inv
        
        # Causal weights
        self.lambda_causal = lambda_causal
        
        # GUARDRAIL: Variance penalty
        self.lambda_var_penalty = lambda_var_penalty
        
        # Stage 1 health flag
        self.stage1_healthy = True
        
        # Parameter validation
        assert lambda_var_penalty >= 0, f"Variance penalty must be non-negative, got {lambda_var_penalty}"
        assert ipw_clip >= 1.0, f"IPW clip must be >= 1.0, got {ipw_clip}"
        assert 0 <= ipw_start_epoch <= 1, f"ipw_start_epoch must be in [0,1], got {ipw_start_epoch}"
        assert target_edges > 0, f"target_edges must be > 0, got {target_edges}"
        
        # Components
        self.encoder = DisentangledEncoder(
            input_dim=1,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        )
        
        self.graph_learner = CausalGraphLearner(
            d=d,
            hidden_dim=hidden_dim,
            n_regimes=n_regimes,
        )
        
        self.miss_head = MissingnessHead(
            d=d,
            hidden_dim=hidden_dim,
            n_regimes=n_regimes,
        )
        
        # V4: Environment-specific decoder
        self.use_env_decoder = use_env_specific_decoder and n_regimes > 1
        if self.use_env_decoder:
            self.decoder = EnvironmentSpecificDecoder(
                d=d,
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                n_envs=max(n_regimes, 2),
            )
        else:
            # Standard shared decoder
            self.decoder = self._build_shared_decoder(d, latent_dim, hidden_dim)
        
        # V4: Causal prior loss
        self.causal_prior = CausalPriorLoss(
            d=d,
            n_regimes=n_regimes,
            lambda_intervention=lambda_intervention,
            lambda_orientation=lambda_orientation,
            lambda_necessity=lambda_necessity,
            lambda_mechanism=lambda_mechanism,
        )
        
        # Structure-level invariance
        self.invariance = IRMStructureInvariance(
            n_features=d,
            n_envs=max(n_regimes, 2),
            gamma=0.1,
        )
        
        # Cache for decoder function (for counterfactual testing)
        self._cached_z_signal = None
        self._cached_z_corrupt = None
        
        # Correlation cache for L_anticorr (recomputed periodically)
        self._corr_cache: Optional[torch.Tensor] = None
        self._corr_cache_epoch: int = -1
    
    def _build_shared_decoder(self, d, latent_dim, hidden_dim):
        """Build shared heteroscedastic decoder with MNAR-aware variance."""
        return HeteroscedasticDecoder(
            d=d,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            min_sigma=0.01,
            min_nu=2.1,
            corrupt_dropout=0.5,
        )
    
    def set_intervention_targets(self, targets: Dict[int, List[int]]):
        """Set known intervention targets from data config."""
        self.causal_prior.set_intervention_targets(targets)
    
    def load_intervention_targets_from_config(self, config_path: str):
        """Load intervention targets from dataset config.json."""
        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            
            # Check for intervention info
            corruption = config.get("corruption", {})
            if corruption.get("regime_shift_type") == "intervention":
                # Try to infer targets from regime info
                # For now, assume regimes apply different corruptions to different nodes
                n_regimes = corruption.get("n_regimes", 1)
                # Default: no specific targets known
                targets = {e: [] for e in range(n_regimes)}
                self.set_intervention_targets(targets)
    
    def forward(
        self,
        X: torch.Tensor,
        M: Optional[torch.Tensor] = None,
        regime: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with dual MNAR modeling:
        
        1. Selection model: P(M|X*) predicts missingness from imputed values
           -> Used for inverse probability weighting (IPW)
        
        2. Variance scaling: σ² increases when P(missing) is high
           -> Captures uncertainty from MNAR mechanism
        
        GUARDRAIL: Variance depends on latent state only, NOT on M_probs.
        This prevents shortcut learning where the model games both selection
        and variance to reduce loss without learning causal structure.
        
        IPW (selection model) handles MNAR correction separately.
        """
        # 1. Encode
        z_signal, z_corrupt = self.encoder(X)
        
        # Cache for counterfactual testing
        self._cached_z_signal = z_signal
        self._cached_z_corrupt = z_corrupt
        
        # 2. Learn structure
        if regime is not None and self.n_regimes > 1:
            A, W_adj = self.graph_learner(env_idx=regime)
            A_base = self.graph_learner.get_mean_adjacency()
        else:
            A, W_adj = self.graph_learner()
            A_base = A
        
        # 3. Decode (single pass - no M_probs dependence per GUARDRAIL)
        if self.use_env_decoder and regime is not None:
            mu, sigma, per_env_mu = self.decoder(z_signal, z_corrupt, A_base, regime)
            nu = torch.full_like(mu, 4.0) # Fixed nu for env-specific decoder
            var_components = None
        else:
            # Heteroscedastic decoder - variance from latent state only
            mu, sigma, nu, var_components = self.decoder(
                z_signal, z_corrupt, A_base, regime
            )
            per_env_mu = None
        
        # 4. Predict missingness from IMPUTED values (selection model)
        if self.mnar_detach:
            X_star = mu.detach()
        else:
            X_star = mu
        
        miss_logits = self.miss_head(X_star, regime)
        M_probs = torch.sigmoid(miss_logits) # P(observed | X*)
        
        return {
            "z_signal": z_signal,
            "z_corrupt": z_corrupt,
            "A": A,
            "A_base": A_base,
            "W_adj": W_adj,
            "miss_logits": miss_logits,
            "M_probs": M_probs, # Propensity scores for IPW
            "mu": mu,
            "sigma": sigma,
            "nu": nu,
            "regime": regime,
            "per_env_mu": per_env_mu,
            "X_star": X_star,
            "var_components": var_components,
        }
    
    def _make_decoder_fn(self, A_base_orig: torch.Tensor):
        """Create decoder function for counterfactual testing."""
        z_signal = self._cached_z_signal
        z_corrupt = self._cached_z_corrupt
        
        def decoder_fn(A_test):
            if self.use_env_decoder:
                # Can't easily do per-env here, use mean
                mu, sigma, _ = self.decoder(
                    z_signal, z_corrupt, A_test, 
                    torch.zeros(z_signal.shape[0], dtype=torch.long, device=A_test.device)
                )
            else:
                # HeteroscedasticDecoder returns 4 values
                mu, sigma, nu, _ = self.decoder(z_signal, z_corrupt, A_test)
            return mu
        
        return decoder_fn
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        X: torch.Tensor,
        M: Optional[torch.Tensor] = None,
        regime: Optional[torch.Tensor] = None,
        epoch: int = 1,
        total_epochs: int = 50,
        compute_causal_prior: bool = True,
        compute_necessity: bool = False, # Expensive, do every N epochs
        loss_weights: Optional[Dict[str, float]] = None, # Override scheduled weights
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss with MNAR handling and GUARDRAILS.
        
        GUARDRAILS implemented:
        1. Stabilized IPW: p_bar/p_hat with clipping to [1, w_max]
        2. Variance inflation penalty: lambda * mean(log(sigma^2))
        3. Phase training: IPW only after ipw_start_epoch
        4. Variance from latent only (no M_probs dependence in decoder)
        
        Joint objective:
            log p(X_obs, M) = log p(X_obs | z) + log p(M | X*, e)
        
        Args:
            loss_weights: Optional dict with keys like 'lambda_sparse', 'lambda_acyclic', etc.
                         If provided, these override the internal scheduling.
        """
        device = X.device
        
        if M is None:
            M = torch.ones_like(X)
        
        # =====================================================================
        # SELECTION MODEL: log p(M | X*, e)
        # =====================================================================
        L_miss = F.binary_cross_entropy_with_logits(
            outputs["miss_logits"], M, reduction="none"
        )
        
        propensity = outputs.get("M_probs", torch.sigmoid(outputs["miss_logits"]))
        
        # =====================================================================
        # GUARDRAIL 1: Stabilized IPW with phase training
        # =====================================================================
        ipw_start = int(self.ipw_start_epoch * total_epochs)
        use_ipw_now = self.use_ipw and epoch > ipw_start
        
        if use_ipw_now:
            # Stabilized weights: w = p_bar / p_hat
            # This centers weights around 1 and prevents explosion
            p_bar = propensity.mean().detach() # Marginal probability
            
            # Stabilized IPW: w = p_bar / p_hat (bounded)
            ipw_weights = p_bar / (propensity + 1e-6)
            
            # GUARDRAIL: Symmetric clip around 1
            ipw_weights = torch.clamp(ipw_weights, min=1.0/self.ipw_clip, max=self.ipw_clip)
            
            # Normalize to mean 1 for stable gradients
            ipw_weights = ipw_weights / (ipw_weights.mean() + 1e-6)
            ipw_weights = ipw_weights.detach()
            
            # Weighted reconstruction: (w * nll * M).sum() / (w * M).sum()
            nll_per_elem = student_t_nll(
                X, outputs["mu"], outputs["sigma"], outputs["nu"], mask=None
            )
            # Recompute with proper weighting
            weighted_nll = nll_per_elem * M * ipw_weights
            L_recon = weighted_nll.sum() / ((M * ipw_weights).sum() + 1e-8)
            
            ipw_mean = ipw_weights[M > 0].mean().item() if (M > 0).any() else 1.0
            ipw_max = ipw_weights[M > 0].max().item() if (M > 0).any() else 1.0
            ipw_min = ipw_weights[M > 0].min().item() if (M > 0).any() else 1.0
        else:
            # Phase A: No IPW, just standard masked reconstruction
            L_recon = student_t_nll(
                X, outputs["mu"], outputs["sigma"], outputs["nu"], mask=M
            )
            ipw_mean, ipw_max, ipw_min = 1.0, 1.0, 1.0
        
        # Aggregate missingness loss
        L_miss = L_miss.mean()
        
        # =====================================================================
        # GUARDRAIL 2: Variance inflation penalty
        # Prevents decoder from inflating σ² to "buy" low loss
        # =====================================================================
        sigma = outputs["sigma"]
        # log(σ²) penalty - preferred over σ² as it's scale-invariant
        L_var_penalty = torch.log(sigma ** 2 + 1e-8).mean()
        
        # 3. HSIC disentanglement
        L_hsic = hsic_loss(outputs["z_signal"], outputs["z_corrupt"])
        
        # 4. Acyclicity
        A = outputs["A"]
        A_base = outputs.get("A_base", A)
        
        if A_base.dim() == 3:
            A_for_acy = A_base.mean(dim=0)
        else:
            A_for_acy = A_base
        
        h_A_raw = notears_acyclicity(A_for_acy, normalize=False)
        L_acyclic = notears_acyclicity(A_for_acy, normalize=True)
        
        # Lambda scheduling for acyclicity (use override if provided)
        if loss_weights and "lambda_acyclic" in loss_weights:
            lambda_acy_used = loss_weights["lambda_acyclic"]
        else:
            lambda_acy_used = self.get_lambda_acyclic(epoch, total_epochs)
        
        # =====================================================================
        # CRITICAL FIX: Sparsity on LOGITS (W), not adjacency (A)
        # This pushes logits strongly negative for non-edges, so sigmoid(A)
        # actually collapses toward 0. Without this, shrinking A can still 
        # leave all entries in the 0.2-0.3 band.
        # =====================================================================
        W_logits = self.graph_learner.get_logits()
        # Zero diagonal for logits too
        W_logits_masked = W_logits * (1 - torch.eye(self.d, device=device))
        
        # L1 on logits - this pushes non-edges to large negative values
        L_sparse_logits = W_logits_masked.abs().sum() / (self.d * (self.d - 1))
        
        # Keep old L_sparse for monitoring (but use logit version in loss)
        L_sparse_A = A_for_acy.abs().sum() / (self.d * self.d)
        L_sparse = L_sparse_logits # Use logit-based sparsity
        
        if loss_weights and "lambda_sparse" in loss_weights:
            lambda_sparse_used = loss_weights["lambda_sparse"]
        else:
            lambda_sparse_used = self.get_lambda_sparse(epoch, total_epochs)
        
        # =====================================================================
        # CRITICAL FIX: Binarization penalty A*(1-A)
        # Encourages adjacency values to be near 0 or 1, not in gray zone
        # Max penalty at A=0.5, zero penalty at A=0 or A=1
        # =====================================================================
        L_binary = (A_for_acy * (1 - A_for_acy)).sum() / (self.d * (self.d - 1))
        lambda_binary = loss_weights.get("lambda_binary", 0.1) if loss_weights else 0.1
        
        # =====================================================================
        # V8.20 FIX: L_sym - Symmetry penalty to force antisymmetric direction
        # Problem: When W_ij ≈ W_ji, direction is mathematically unidentifiable
        # Fix: Penalize min(W_ij, W_ji) to force one direction to "win"
        # L_sym = Σ_{i<j} min(A_ij, A_ji) / num_pairs
        # =====================================================================
        A_sym = A_for_acy
        A_sym_T = A_sym.T
        # For each pair (i,j), penalize having both directions on
        L_sym_matrix = torch.minimum(A_sym, A_sym_T)
        # Only count upper triangle (avoid double counting)
        upper_mask = torch.triu(torch.ones_like(L_sym_matrix), diagonal=1)
        L_sym = (L_sym_matrix * upper_mask).sum() / (upper_mask.sum() + 1e-8)
        lambda_sym = loss_weights.get("lambda_sym", 0.0) if loss_weights else 0.0
        
        # =====================================================================
        # V8.20 FIX: L_sep - TopK separation loss to create edge gap
        # Problem: gap=0.000 means TopK membership is unstable (30th ≈ 31st edge)
        # Fix: Penalize when min(TopK) - max(Rest) < delta
        # L_sep = ReLU(delta - (min(TopK) - max(Rest)))
        # =====================================================================
        target_k = self.target_edges
        A_flat = (A_for_acy * (1 - torch.eye(self.d, device=device))).flatten()
        if A_flat.numel() > target_k:
            # Get TopK and rest
            topk_vals, topk_idx = torch.topk(A_flat, target_k)
            # Create mask for rest
            rest_mask = torch.ones_like(A_flat, dtype=torch.bool)
            rest_mask[topk_idx] = False
            rest_vals = A_flat[rest_mask]
            
            if rest_vals.numel() > 0:
                min_topk = topk_vals.min()
                max_rest = rest_vals.max()
                gap = min_topk - max_rest
                delta_sep = loss_weights.get("delta_sep", 0.05) if loss_weights else 0.05
                L_sep = torch.relu(delta_sep - gap)
            else:
                L_sep = torch.tensor(0.0, device=device)
                gap = torch.tensor(0.0, device=device)
        else:
            L_sep = torch.tensor(0.0, device=device)
            gap = torch.tensor(0.0, device=device)
        lambda_sep = loss_weights.get("lambda_sep", 0.0) if loss_weights else 0.0
        
        # =====================================================================
        # V8.21 FIX: L_bimodal - Per-edge targeting loss
        # ROOT CAUSE: L_budget creates UNIFORM gradient on all 210 edges (all
        # shrink together). L_sep only creates 2 gradient signals (boundary pair).
        # So L_budget overwhelms L_sep 100:1 and gap NEVER opens.
        #
        # FIX: L_bimodal pushes EACH TopK edge toward 1 and EACH rest edge
        # toward 0, creating 210 TARGETED gradient signals. This gives the
        # reconstruction loss "breathing room" to vote on which edges matter.
        #
        # L_bimodal = mean((topk - 1)^2) + mean(rest^2)
        # Gradient per topk edge: 2*(val-1) ≈ -1.0 (push UP when val=0.5)
        # Gradient per rest edge: 2*val ≈ +1.0 (push DOWN when val=0.5)
        # =====================================================================
        if A_flat.numel() > target_k:
            # topk_vals and rest_vals already computed above
            L_bimodal = torch.mean((topk_vals - 1.0) ** 2) + torch.mean(rest_vals ** 2)
        else:
            L_bimodal = torch.tensor(0.0, device=device)
        lambda_bimodal = loss_weights.get("lambda_bimodal", 0.0) if loss_weights else 0.0
        
        # =====================================================================
        # V8.22 FIX A: L_tail - Explicit tail suppression
        # PROBLEM: V8.21 created a stable TopK (gap>0, @90%=30), but ALL 210
        # off-diagonal edges remain ≥ 0.2 ("@0.2=210"). This dense tail:
        #   - hurts AUROC/AUPRC (non-edges aren't low)
        #   - keeps direction noisy (both A_ij and A_ji moderately high)
        #   - inflates edge_sum → guard fails
        #
        # V8.23 FIX: mean(relu(rest - t_tail)^2) — QUADRATIC penalty
        # V8.22 used linear relu which SATURATES: constant gradient for all
        # edges above threshold, providing no shape to push edges down.
        # Quadratic: gradient = 2*(val-t) → stronger for farther edges,
        # zero exactly at threshold. Non-saturating, provides useful curvature.
        # =====================================================================
        if A_flat.numel() > target_k and rest_vals.numel() > 0:
            t_tail = loss_weights.get("tail_threshold", 0.05) if loss_weights else 0.05
            L_tail = torch.mean(torch.relu(rest_vals - t_tail) ** 2)
        else:
            L_tail = torch.tensor(0.0, device=device)
        lambda_tail = loss_weights.get("lambda_tail", 0.0) if loss_weights else 0.0
        
        # =====================================================================
        # V8.22 FIX B: L_dir_logit - Directional margin in logit space
        # PROBLEM: DirConf ~0.33-0.57 (need ≥0.8). For many pairs (i,j),
        # |A_ij - A_ji| is tiny, so direction is fragile and flips.
        # L_sym penalizes min(A_ij,A_ji) but doesn't enforce a hard margin.
        #
        # FIX: For TopK candidate edges, enforce |W_ij - W_ji| >= margin in
        # LOGIT space (where gradients are stronger and more symmetric).
        # L_dir = mean(relu(margin - |W_ij - W_ji|)) over TopK pairs
        # =====================================================================
        dir_margin = loss_weights.get("dir_logit_margin", 0.5) if loss_weights else 0.5
        if A_flat.numel() > target_k:
            # Convert TopK flat indices to (i,j) pairs
            topk_i = topk_idx // self.d
            topk_j = topk_idx % self.d
            # Get logit values for forward and reverse directions
            W_fwd = W_logits_masked[topk_i, topk_j]
            W_rev = W_logits_masked[topk_j, topk_i]
            # Margin violation: want |W_fwd - W_rev| >= margin
            L_dir_logit = torch.mean(torch.relu(dir_margin - torch.abs(W_fwd - W_rev)))
        else:
            L_dir_logit = torch.tensor(0.0, device=device)
        lambda_dir_logit = loss_weights.get("lambda_dir_logit", 0.0) if loss_weights else 0.0
        
        # =====================================================================
        # V8.24 FIX A: L_icp — ICP-style invariant causal edge scoring
        # PROBLEM: Reconstruction loss (15*75000 gradient signals/epoch) picks
        # correlation-strong edges. Structural losses (210 signals on A) can't
        # change WHICH edges are in TopK, only HOW the adjacency is shaped.
        #
        # FIX: For each TopK edge (i→j), compute per-environment prediction
        # error. TRUE causal edges have INVARIANT prediction quality across
        # environments. Correlation edges work well in some envs but fail in
        # others (environment-dependent). Reward low-variance edges.
        #
        # L_icp = mean over TopK edges of: Var_e[MSE_e(i→j)] / (mean_e[MSE_e(i→j)] + eps)
        # This is a coefficient of variation: penalizes edges with high
        # cross-environment variance relative to their mean error.
        # =====================================================================
        L_icp = torch.tensor(0.0, device=device)
        lambda_icp = loss_weights.get("lambda_icp", 0.0) if loss_weights else 0.0
        
        if lambda_icp > 0 and regime is not None and A_flat.numel() > target_k:
            env_ids, regime_idx = regime.unique(return_inverse=True)
            
            if len(env_ids) >= 2 and X.dim() == 3 and X.shape[1] > 1:
                # Vectorized ICP: no Python loop over environments
                X_in = X[:, :-1, :]    # [B, T-1, d]
                X_tgt = X[:, 1:, :]    # [B, T-1, d]
                M_tgt = M[:, 1:, :] if M is not None else torch.ones_like(X_tgt)
                
                topk_i = topk_idx // self.d
                topk_j = topk_idx % self.d
                A_vals = A_for_acy[topk_i, topk_j]  # [K]
                
                # Predictions and errors for all samples at once
                pred_j = A_vals.unsqueeze(0).unsqueeze(0) * X_in[:, :, topk_i]  # [B, T-1, K]
                actual_j = X_tgt[:, :, topk_j]  # [B, T-1, K]
                mask_j = M_tgt[:, :, topk_j]    # [B, T-1, K]
                errors = ((pred_j - actual_j) ** 2) * mask_j  # [B, T-1, K]
                
                # One-hot env assignment (regime_idx already contiguous from unique)
                env_onehot = F.one_hot(regime_idx, len(env_ids)).float()  # [B, n_envs]
                
                # Aggregate per environment via einsum (sum over batch, time already summed)
                err_sum_bt = errors.sum(dim=1)   # [B, K]
                mask_sum_bt = mask_j.sum(dim=1)  # [B, K]
                env_err = torch.einsum('bk,be->ek', err_sum_bt, env_onehot)   # [n_envs, K]
                env_cnt = torch.einsum('bk,be->ek', mask_sum_bt, env_onehot)  # [n_envs, K]
                env_mse = env_err / (env_cnt + 1e-8)  # [n_envs, K]
                
                # Filter envs with >= 2 samples
                env_sample_count = env_onehot.sum(dim=0)  # [n_envs]
                valid_envs = env_sample_count >= 2
                if valid_envs.sum() >= 2:
                    env_mse_valid = env_mse[valid_envs]  # [n_valid, K]
                    env_var = torch.var(env_mse_valid, dim=0, unbiased=False)  # [K]
                    env_mean = torch.mean(env_mse_valid, dim=0)  # [K]
                    # L_icp = mean CV across TopK edges (higher = more env-dependent = BAD)
                    L_icp = torch.mean(env_var / (env_mean + 1e-8))
        
        # =====================================================================
        # V8.24 FIX B: L_anticorr — Anti-correlation penalty on TopK edges
        # PROBLEM: Reconstruction picks edges that correlate with |corr(X_i,X_j)|.
        # These are correlation-strong, not necessarily causal.
        #
        # FIX: Compute pairwise |corr(X_i, X_j)| from batch X. Penalize TopK
        # edges whose adjacency ranking aligns with correlation ranking.
        # Specifically: for TopK edges, penalize A_ij * |corr(i,j)|.
        # This pushes the model to AVOID placing high-weight edges on
        # high-correlation pairs, breaking the correlation→edge pipeline.
        #
        # L_anticorr = mean over TopK of: A_ij * |corr(X_i, X_j)|
        # =====================================================================
        L_anticorr = torch.tensor(0.0, device=device)
        lambda_anticorr = loss_weights.get("lambda_anticorr", 0.0) if loss_weights else 0.0
        
        if lambda_anticorr > 0 and A_flat.numel() > target_k:
            # Use cached correlation matrix (recompute every 10 epochs)
            if self._corr_cache is None or abs(epoch - self._corr_cache_epoch) >= 10:
                X_flat_c = X.detach().reshape(-1, self.d)  # [B*T, d]
                X_centered = X_flat_c - X_flat_c.mean(dim=0, keepdim=True)
                cov = (X_centered.T @ X_centered) / (X_centered.shape[0] - 1 + 1e-8)
                std = torch.sqrt(torch.diag(cov) + 1e-8)
                corr_matrix = cov / (std.unsqueeze(0) * std.unsqueeze(1) + 1e-8)
                corr_matrix = corr_matrix.clamp(-1, 1).abs()
                corr_matrix = corr_matrix * (1 - torch.eye(self.d, device=device))
                self._corr_cache = corr_matrix.detach()
                self._corr_cache_epoch = epoch
            
            corr_matrix = self._corr_cache
            
            # For TopK edges: penalize A_ij * |corr(i,j)|
            topk_i_ac = topk_idx // self.d
            topk_j_ac = topk_idx % self.d
            topk_A_vals = A_for_acy[topk_i_ac, topk_j_ac]  # [K]
            topk_corr_vals = corr_matrix[topk_i_ac, topk_j_ac]  # [K]
            
            # Detach corr so gradient only flows through A, not through data stats
            L_anticorr = torch.mean(topk_A_vals * topk_corr_vals.detach())
        
        # 6. Budget (Fix D: Asymmetric penalty - penalize under-shooting more)
        # V8.19 FIX A: Normalize by d² so L_budget is O(1) not O(d⁴)
        # Before: edge_sum ≈ 94.5 for d=15 flat, target=30 → penalty=(94.5-30)²≈4160
        # After: divide by d² → 4160/225 ≈ 18.5, and with λb=0.5 → 9.25 (manageable)
        edge_sum = A_for_acy.sum()
        d_sq = float(self.d * self.d)
        if edge_sum < self.target_edges:
            # Below target: stronger penalty to prevent collapse
            L_budget = 2.0 * (self.target_edges - edge_sum) ** 2 / d_sq
        else:
            # Above target: normal penalty
            L_budget = (edge_sum - self.target_edges) ** 2 / d_sq
        if loss_weights and "lambda_budget" in loss_weights:
            lambda_budget_used = loss_weights["lambda_budget"]
        else:
            lambda_budget_used = self.get_lambda_budget(epoch, total_epochs)
        
        # Get other lambda values from overrides or defaults
        lambda_recon = loss_weights.get("lambda_recon", self.lambda_recon) if loss_weights else self.lambda_recon
        lambda_miss = loss_weights.get("lambda_miss", self.lambda_miss) if loss_weights else self.lambda_miss
        lambda_hsic = loss_weights.get("lambda_hsic", self.lambda_hsic) if loss_weights else self.lambda_hsic
        lambda_inv = loss_weights.get("lambda_inv", self.lambda_inv) if loss_weights else self.lambda_inv
        lambda_causal = loss_weights.get("lambda_causal", self.lambda_causal) if loss_weights else self.lambda_causal
        lambda_var_penalty = loss_weights.get("lambda_var_penalty", self.lambda_var_penalty) if loss_weights else self.lambda_var_penalty
        
        # 7. Structure-level invariance
        if regime is not None and self.n_regimes > 1:
            W_adj = outputs.get("W_adj", None)
            L_inv, inv_metrics = self.invariance(
                A=A,
                logits=W_adj,
                X=X,
                M=M,
                e=regime,
            )
        else:
            L_inv = torch.tensor(0.0, device=device)
            inv_metrics = {'grad_penalty': 0.0, 'var_penalty': 0.0}
        
        # 8. V4: Causal prior loss
        L_causal = torch.tensor(0.0, device=device)
        causal_metrics = {}
        if compute_causal_prior and epoch > 5: # Start after warmup
            decoder_fn = self._make_decoder_fn(A_for_acy) if compute_necessity else None
            
            L_causal, causal_metrics = self.causal_prior(
                X=X,
                A=A_for_acy,
                A_per_env=A if A.dim() == 3 else None,
                regime=regime,
                decoder_fn=decoder_fn,
                env_decoder=self.decoder if self.use_env_decoder else None,
                z_signal=outputs["z_signal"],
                compute_necessity=compute_necessity,
            )
        
        # Total loss with GUARDRAIL 2: variance penalty + binarization
        # V8.22: Add L_tail (tail suppression) + L_dir_logit (direction margin)
        # V8.24: Add L_icp (invariant causal scoring) + L_anticorr (anti-correlation)
        loss = (
            lambda_recon * L_recon
            + lambda_miss * L_miss
            + lambda_hsic * L_hsic
            + lambda_acy_used * L_acyclic
            + lambda_sparse_used * L_sparse
            + lambda_budget_used * L_budget
            + lambda_inv * L_inv
            + lambda_causal * L_causal
            + lambda_var_penalty * L_var_penalty # GUARDRAIL 2
            + lambda_binary * L_binary # Binarization penalty for edge confidence
            + lambda_sym * L_sym # V8.20: Symmetry penalty for direction
            + lambda_sep * L_sep # V8.20: TopK separation for edge gap
            + lambda_bimodal * L_bimodal # V8.21: Per-edge bimodal targeting
            + lambda_tail * L_tail # V8.22: Hard tail suppression
            + lambda_dir_logit * L_dir_logit # V8.22: Direction margin on TopK logits
            + lambda_icp * L_icp # V8.24: ICP invariant edge scoring
            + lambda_anticorr * L_anticorr # V8.24: Anti-correlation penalty
        )
        
        # Metrics with full guardrail tracking
        metrics = {
            "loss": loss.item(),
            "L_recon": L_recon.item(),
            "L_miss": L_miss.item(),
            "L_hsic": L_hsic.item(),
            "L_acyclic": L_acyclic.item(),
            "h_A_raw": h_A_raw.item(),
            "lambda_acy_used": lambda_acy_used,
            "L_sparse": L_sparse.item(),
            "L_sparse_A": L_sparse_A.item(), # Sparsity on adjacency (for monitoring)
            "L_sparse_logits": L_sparse_logits.item(), # Sparsity on logits (used in loss)
            "L_binary": L_binary.item(), # Binarization penalty
            "L_sym": L_sym.item(), # V8.20: Symmetry penalty
            "L_sep": L_sep.item(), # V8.20: TopK separation
            "L_bimodal": L_bimodal.item(), # V8.21: Per-edge bimodal targeting
            "L_tail": L_tail.item(), # V8.22: Tail suppression
            "L_dir_logit": L_dir_logit.item(), # V8.22: Direction margin
            "L_icp": L_icp.item(), # V8.24: ICP invariant edge scoring
            "L_anticorr": L_anticorr.item(), # V8.24: Anti-correlation penalty
            "lambda_icp": lambda_icp,
            "lambda_anticorr": lambda_anticorr,
            "lambda_sym": lambda_sym,
            "lambda_sep": lambda_sep,
            "lambda_bimodal": lambda_bimodal,
            "lambda_tail": lambda_tail,
            "lambda_dir_logit": lambda_dir_logit,
            "topk_gap": gap.item() if isinstance(gap, torch.Tensor) else gap, # V8.20: Actual gap
            "lambda_sparse_used": lambda_sparse_used,
            "lambda_binary": lambda_binary,
            "L_budget": L_budget.item(),
            "lambda_bud_used": lambda_budget_used,
            "edge_sum": edge_sum.item(),
            "target_edges": self.target_edges,
            "A_mean": A_for_acy.mean().item(),
            "A_max": A_for_acy.max().item(),
            "A_min": A_for_acy[A_for_acy > 0].min().item() if (A_for_acy > 0).any() else 0.0, # Min non-zero
            "W_logits_mean": W_logits_masked.mean().item(), # Logits stats
            "W_logits_min": W_logits_masked.min().item(),
            "W_logits_max": W_logits_masked.max().item(),
            "temperature": self.graph_learner.get_temperature(),
            "L_inv": L_inv.item() if isinstance(L_inv, torch.Tensor) else L_inv,
            "L_causal": L_causal.item() if isinstance(L_causal, torch.Tensor) else L_causal,
            # GUARDRAIL metrics
            "L_var_penalty": L_var_penalty.item(),
            "ipw_active": 1.0 if use_ipw_now else 0.0,
            "ipw_mean": ipw_mean,
            "ipw_max": ipw_max,
            "ipw_min": ipw_min,
            "propensity_mean": propensity.mean().item(),
            "propensity_min": propensity.min().item(),
            # Heteroscedastic variance metrics
            "sigma_mean": outputs["sigma"].mean().item(),
            "sigma_max": outputs["sigma"].max().item(),
            "sigma_min": outputs["sigma"].min().item(),
        }
        
        # Add variance component metrics if available
        var_components = outputs.get("var_components")
        if var_components is not None:
            if "base_var" in var_components:
                metrics["base_var_mean"] = var_components["base_var"].mean().item()
            if "bias_var" in var_components:
                metrics["bias_var_mean"] = var_components["bias_var"].mean().item()
        
        metrics.update(causal_metrics)
        
        return loss, metrics
    
    def get_lambda_acyclic(self, epoch: int, total_epochs: int) -> float:
        """Delayed acyclicity schedule."""
        lambda_acy_max = 0.05
        stage2_start = int(0.50 * total_epochs)
        stage2_ramp_end = int(0.80 * total_epochs)
        
        if epoch <= stage2_start:
            return 0.0
        elif epoch <= stage2_ramp_end:
            progress = (epoch - stage2_start) / (stage2_ramp_end - stage2_start)
            return lambda_acy_max * progress
        else:
            return lambda_acy_max
    
    def get_lambda_sparse(self, epoch: int, total_epochs: int) -> float:
        """Sparsity schedule."""
        stage2_start = int(0.30 * total_epochs)
        if epoch <= stage2_start:
            return 1e-4
        else:
            progress = (epoch - stage2_start) / (total_epochs - stage2_start)
            return 1e-4 + progress * (1e-3 - 1e-4)
    
    def get_lambda_budget(self, epoch: int, total_epochs: int) -> float:
        """Budget schedule."""
        stage2_start = int(0.30 * total_epochs)
        if epoch <= stage2_start:
            return 0.0
        else:
            progress = (epoch - stage2_start) / (total_epochs - stage2_start)
            return 0.03 * progress
    
    def diagnose(
        self,
        X: torch.Tensor,
        A_true: torch.Tensor,
    ) -> Dict[str, float]:
        """Run correlation vs causation diagnostic.
        
        V8.26: Model already outputs causal convention (A[i,j]=i→j),
        same as A_true. to_causal_convention() is identity.
        """
        A_pred = self.graph_learner.get_mean_adjacency()
        A_pred_causal = self.to_causal_convention(A_pred)
        return diagnose_correlation_vs_causation(A_pred_causal, A_true, X)
