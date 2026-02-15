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
# Edge-Local Direction Scorer (V9.0)
# =============================================================================

class EdgeDirectionScorer(nn.Module):
    """
    V9.0: Edge-local direction scoring from node embeddings.
    
    Given node embeddings h_i, h_j ∈ R^L from the encoder, computes
    a direction score for edge (i,j):
    
        score(h_i, h_j) = MLP(h_i || h_j)     (scalar)
        d_ij = σ( score(h_i, h_j) - score(h_j, h_i) )
    
    ANTISYMMETRY BY CONSTRUCTION:
        score(h_i, h_j) - score(h_j, h_i) = -(score(h_j, h_i) - score(h_i, h_j))
        → d_ij + d_ji = 1    (guaranteed, not hoped-for)
    
    WHY THIS FIXES REFINE:
        Old W_dir was a global [d,d] parameter — no data dependency.
        Invariance gradient path: L_inv → z → A → W_dir was too long.
        New: direction comes from node embeddings → gradient flows
        L_inv → z → A → d_ij → score_MLP → h_i,h_j → encoder.
        The direction scorer sees the SAME embeddings that the decoder
        uses, so cross-environment differences directly inform direction.
    """
    
    def __init__(self, latent_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        
        # MLP: concat(h_i, h_j) → scalar score
        # Small network — direction is a simple ordering relation
        self.scorer = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def _score_pairs(self, h: torch.Tensor) -> torch.Tensor:
        """Score all node pairs from embeddings [*, d, L] → [*, d, d].
        
        Works on any leading batch dims. Returns raw scores (before antisym).
        """
        *batch_dims, d, L = h.shape
        # Build all pairs: h_i || h_j for each (i,j)
        h_i = h.unsqueeze(-2).expand(*batch_dims, d, d, L)  # [*, d, d, L]
        h_j = h.unsqueeze(-3).expand(*batch_dims, d, d, L)  # [*, d, d, L]
        h_pairs = torch.cat([h_i, h_j], dim=-1)  # [*, d, d, 2L]
        scores = self.scorer(h_pairs).squeeze(-1)  # [*, d, d]
        return scores
    
    def forward(
        self, z_signal: torch.Tensor, tau: float = 1.0
    ) -> torch.Tensor:
        """
        V9.1: Compute direction probabilities from node embeddings.
        
        KEY FIX: Compute logits PER-SAMPLE, then average logits.
        V9.0 bug: mean-pooling embeddings before scoring destroyed
        per-sample direction signal. Now each sample votes on direction.
        
        Args:
            z_signal: [B, T, d, L] or [B, d, L] node embeddings
            tau: temperature for sigmoid sharpening
            
        Returns:
            dir_probs: [d, d] direction probabilities, d_ij + d_ji = 1
        """
        if z_signal.dim() == 4:
            B, T, d, L = z_signal.shape
            # Score every (sample, time) independently: [B*T, d, d]
            h_flat = z_signal.reshape(B * T, d, L)
            scores_flat = self._score_pairs(h_flat)  # [B*T, d, d]
            # Antisymmetric logits per sample
            antisym_flat = scores_flat - scores_flat.transpose(-1, -2)  # [B*T, d, d]
            # Average LOGITS across samples (preserves per-sample asymmetries)
            antisym_logits = antisym_flat.mean(dim=0)  # [d, d]
        elif z_signal.dim() == 3:
            B, d, L = z_signal.shape
            scores_flat = self._score_pairs(z_signal)  # [B, d, d]
            antisym_flat = scores_flat - scores_flat.transpose(-1, -2)
            antisym_logits = antisym_flat.mean(dim=0)  # [d, d]
        else:
            # [d, L] — single set of embeddings (eval fallback)
            d, L = z_signal.shape
            scores = self._score_pairs(z_signal.unsqueeze(0)).squeeze(0)  # [d, d]
            antisym_logits = scores - scores.T
        
        # Direction probabilities with temperature
        dir_probs = torch.sigmoid(antisym_logits / tau)
        
        # Zero diagonal
        dir_probs = dir_probs * (1 - torch.eye(d, device=dir_probs.device))
        
        return dir_probs


# =============================================================================
# Causal Graph Learner with Edge-Local Direction (V9.0)
# =============================================================================

class CausalGraphLearner(nn.Module):
    """
    V9.0: Causal graph learner with EDGE-LOCAL direction scoring.
    
    Two components:
    - W_mag [d,d]: symmetric parameter for edge MAGNITUDE (corr-initialized)
    - EdgeDirectionScorer: MLP that computes direction from node embeddings
    
    A_ij = σ(W_mag_ij) * d_ij(z_signal)
    A_ji = σ(W_mag_ij) * (1 - d_ij(z_signal))
    
    V9.0 key insight: Direction is computed from node embeddings (h_i, h_j),
    NOT from global parameters. This gives direction a DIRECT gradient path
    from the invariance loss through the encoder, fixing the dead REFINE problem.
    
    W_dir is REMOVED. Direction comes entirely from the EdgeDirectionScorer.
    """
    
    def __init__(
        self,
        d: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        n_regimes: int = 1,
        init_scale: float = 0.01,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.d = d
        self.n_regimes = max(n_regimes, 1)
        self.temperature = temperature
        
        # W_mag: symmetric edge-existence logits (corr-initialized later)
        self.W_mag = nn.Parameter(torch.randn(d, d) * init_scale)
        
        # V9.0: Edge-local direction scorer (replaces global W_dir)
        self.direction_scorer = EdgeDirectionScorer(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim // 2,  # Small — direction is simple
        )
        
        # V9.0: Backward-compat shim — W_dir kept as frozen buffer for
        # code that reads it (edge swap, diagnostics). Never trained.
        self.register_buffer("W_dir", torch.zeros(d, d))
        
        # V9.0: Cache last direction probs for get_mean_adjacency() / eval
        self.register_buffer("_cached_dir_probs", 0.5 * torch.ones(d, d))
        
        # Per-environment deltas (applied to W_mag only — structure varies by env)
        if n_regimes > 1:
            self.env_deltas = nn.Parameter(torch.zeros(n_regimes, d, d))
        else:
            self.register_buffer("env_deltas", torch.zeros(1, d, d))
        
        # For temperature annealing
        self.register_buffer("current_temp", torch.tensor(temperature))
    
    def _get_sym_mag(self) -> torch.Tensor:
        """Project W_mag to enforced symmetry with zero diagonal.
        
        Returns:
            Wm: [d,d] symmetric magnitude logits, diag=0
        """
        diag = torch.eye(self.d, device=self.W_mag.device)
        Wm = 0.5 * (self.W_mag + self.W_mag.T) * (1 - diag)
        return Wm
    
    # Backward-compat alias used by training script
    def _get_sym_antisym(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Backward-compat: returns (Wm, zeros).
        V9.0: W_dir is dead. Direction comes from EdgeDirectionScorer.
        """
        Wm = self._get_sym_mag()
        Wd = torch.zeros_like(Wm)
        return Wm, Wd
    
    def forward(
        self,
        env_idx: Optional[torch.Tensor] = None,
        z_signal: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        V9.0: Get adjacency matrix with edge-local direction.
        
        Args:
            env_idx: [B] environment indices. If None, return base A.
            z_signal: [B, T, d, L] node embeddings from encoder.
                      If provided and antisymmetric mode is on, direction
                      is computed from embeddings. If None, uses cached dir.
            
        Returns:
            A: [d, d] or [B, d, d] adjacency matrix
            W: [d, d] or [B, d, d] logits (symmetrized W_mag)
        """
        tau = self.current_temp
        antisym = getattr(self, '_antisymmetric', False)
        Wm = self._get_sym_mag()
        
        # V9.0: Compute direction from embeddings if available
        if antisym and z_signal is not None:
            dir_probs = self.direction_scorer(z_signal, tau=tau)
            # Cache detached copy for eval / get_direction_map()
            self._cached_dir_probs = dir_probs.detach()
            # Keep live (with grad) for get_mean_adjacency() during training
            self._live_dir_probs = dir_probs
        elif antisym:
            # No embeddings — use cached (e.g. during eval)
            dir_probs = self._cached_dir_probs
            self._live_dir_probs = None
        else:
            dir_probs = None
        
        if env_idx is None or self.n_regimes == 1:
            if antisym and dir_probs is not None:
                A = self._edge_local_adjacency(Wm, dir_probs)
                return A, Wm
            logits = Wm / tau
            A = torch.sigmoid(logits)
            A = A * (1 - torch.eye(self.d, device=A.device))
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
                Wm_batch[b] = 0.5 * (Wm_e + Wm_e.transpose(-1, -2)) * (1 - diag_mask)
        
        if antisym and dir_probs is not None:
            mag_batch = torch.sigmoid(Wm_batch)
            # Direction is shared across envs (from embedding average)
            dir_batch = dir_probs.unsqueeze(0).expand(B, -1, -1)
            A = mag_batch * dir_batch
            batch_diag = diag_mask.unsqueeze(0).expand(B, -1, -1)
            A = A * (1 - batch_diag)
            return A, Wm_batch
        
        logits = Wm_batch / tau
        A = torch.sigmoid(logits)
        batch_diag = diag_mask.unsqueeze(0).expand(B, -1, -1)
        A = A * (1 - batch_diag)
        
        return A, Wm_batch
    
    def _edge_local_adjacency(
        self, W_mag: torch.Tensor, dir_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        V9.0: Edge-local direction adjacency.
        
        A_ij = σ(Wm_ij) * d_ij
        
        Where d_ij comes from EdgeDirectionScorer (data-dependent),
        NOT from a global W_dir parameter.
        
        GUARANTEED: d_ij + d_ji = 1 (by antisymmetric construction in scorer)
        → A_ij + A_ji = mag_ij (direction cannot change edge mass)
        """
        d = W_mag.shape[0]
        device = W_mag.device
        mag = torch.sigmoid(W_mag)
        A = mag * dir_probs
        A = A * (1 - torch.eye(d, device=device))
        return A
    
    def get_mean_adjacency(self) -> torch.Tensor:
        """Get base adjacency. Uses live direction (with grad) if available, else cached."""
        antisym = getattr(self, '_antisymmetric', False)
        Wm = self._get_sym_mag()
        if antisym:
            # Use live dir_probs (with grad) if we're in a forward pass
            dir_probs = getattr(self, '_live_dir_probs', None)
            if dir_probs is None:
                dir_probs = self._cached_dir_probs
            return self._edge_local_adjacency(Wm, dir_probs)
        logits = Wm / self.current_temp
        A = torch.sigmoid(logits)
        A = A * (1 - torch.eye(self.d, device=A.device))
        return A
    
    def _antisymmetric_adjacency(self, W_mag: torch.Tensor, W_dir: torch.Tensor, tau: float) -> torch.Tensor:
        """DEPRECATED V8.33 — kept for backward compat. Uses _edge_local_adjacency instead."""
        return self._edge_local_adjacency(W_mag, self._cached_dir_probs)
    
    def set_antisymmetric(self, enabled: bool = True):
        """Enable/disable edge-local antisymmetric mode."""
        self._antisymmetric = enabled
    
    @property
    def W_adj(self) -> torch.Tensor:
        """Backward-compat: returns symmetrized Wm for code that reads W_adj."""
        return self._get_sym_mag()
    
    def get_direction_map(self) -> torch.Tensor:
        """V9.0: Return cached direction probabilities.
        
        Returns [d,d] tensor where p_ij ∈ (0,1). At 0.5 = undecided,
        near 0 or 1 = decided. Guarantees p_ij + p_ji = 1.
        
        NOTE: These are from the last forward pass. Call forward() with
        z_signal first to get fresh values.
        """
        return self._cached_dir_probs
    
    def get_temperature(self) -> float:
        return self.current_temp.item()
    
    def set_temperature(self, temp: float):
        self.current_temp.fill_(temp)
    
    def increment_step(self):
        """Hook for temperature annealing schedules."""
        pass
    
    def topk_project(self, k: int, use_logits: bool = True) -> torch.Tensor:
        """
        Hard Top-K projection: return binary mask with exactly K edges.
        Uses W_mag for edge ranking (structure parameter only).
        Selects from upper triangle, then mirrors to full matrix.
        """
        device = self.W_mag.device
        
        if use_logits:
            Wm = self._get_sym_mag()
        else:
            A = self.get_mean_adjacency()
            Wm = torch.maximum(A, A.T)
        
        W_upper = torch.triu(Wm, diagonal=1)
        flat = W_upper.flatten()
        n_upper = self.d * (self.d - 1) // 2
        k = min(k, n_upper)
        
        if k <= 0:
            return torch.zeros(self.d, self.d, device=device)
        
        _, topk_idx = torch.topk(flat, k)
        
        mask_flat = torch.zeros_like(flat)
        mask_flat[topk_idx] = 1.0
        mask = mask_flat.reshape(self.d, self.d)
        mask = (mask + mask.T).clamp(max=1.0)
        
        return mask
    
    def get_logits(self) -> torch.Tensor:
        """Get symmetrized Wm logits for sparsity penalty."""
        return self._get_sym_mag()


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
    
    VERSION = "9.1"
    
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
            latent_dim=latent_dim,
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
        
        # 2. Learn structure (V9.0: pass z_signal for edge-local direction)
        if regime is not None and self.n_regimes > 1:
            A, W_adj = self.graph_learner(env_idx=regime, z_signal=z_signal)
            A_base = self.graph_learner.get_mean_adjacency()
        else:
            A, W_adj = self.graph_learner(z_signal=z_signal)
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
    
    def compute_lead_lag_direction_loss(
        self,
        X: torch.Tensor,
        M: Optional[torch.Tensor] = None,
        dir_probs: Optional[torch.Tensor] = None,
        mag: Optional[torch.Tensor] = None,
        topk_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        V9.1: Lead-lag pseudo-label loss for direction scoring.
        
        Computes temporal cross-correlation asymmetry from data X:
            g_ij = E[X_i(t-1) * X_j(t)] - E[X_j(t-1) * X_i(t)]
        
        If g_ij > 0, node i "leads" node j → pseudo-label y_ij = 1.
        Train direction scorer with BCE(d_ij, y_ij), weighted by
        magnitude (stop-grad) and |g_ij| (confidence).
        
        This gives direction its OWN loss that ONLY the scorer can reduce.
        No gradient cancellation from A_ij + A_ji invariance.
        
        Args:
            X: [B, T, d] time series data
            M: [B, T, d] observation mask (optional)
            dir_probs: [d, d] current direction probabilities (with grad)
            mag: [d, d] magnitude σ(W_mag) (stop-graded)
            topk_mask: [d, d] binary mask for active edges (optional)
            
        Returns:
            loss: scalar lead-lag direction loss
            metrics: dict with diagnostic info
        """
        device = X.device
        d = self.d
        
        # Need time dimension for lead-lag
        if X.dim() < 3 or X.shape[1] < 2:
            return torch.tensor(0.0, device=device), {"L_dir_leadlag": 0.0, "leadlag_coverage": 0.0}
        
        if dir_probs is None:
            dir_probs = self.graph_learner.get_direction_map()
        
        B, T, N = X.shape
        assert N == d, f"Expected d={d}, got N={N}"
        
        # Apply mask: zero out unobserved entries
        if M is not None:
            X_masked = X * M
        else:
            X_masked = X
            M = torch.ones_like(X)
        
        # =====================================================================
        # Compute lead-lag cross-correlation: g_ij = E[X_i(t-1)*X_j(t)] - E[X_j(t-1)*X_i(t)]
        # =====================================================================
        X_lag = X_masked[:, :-1, :]   # [B, T-1, d]  — time t-1
        X_lead = X_masked[:, 1:, :]   # [B, T-1, d]  — time t
        M_lag = M[:, :-1, :]
        M_lead = M[:, 1:, :]
        
        # Valid pairs: both (i at t-1) and (j at t) observed
        # cross_ij = mean( X_i(t-1) * X_j(t) )  over valid (b,t)
        # Shape: [B, T-1, d, 1] * [B, T-1, 1, d] → [B, T-1, d, d]
        cross = X_lag.unsqueeze(-1) * X_lead.unsqueeze(-2)  # [B, T-1, d, d]
        valid = M_lag.unsqueeze(-1) * M_lead.unsqueeze(-2)   # [B, T-1, d, d]
        
        # Sum and normalize
        cross_sum = (cross * valid).sum(dim=(0, 1))   # [d, d]
        valid_sum = valid.sum(dim=(0, 1))               # [d, d]
        cross_mean = cross_sum / (valid_sum + 1e-8)     # [d, d]  E[X_i(t-1)*X_j(t)]
        
        # g_ij = E[X_i(t-1)*X_j(t)] - E[X_j(t-1)*X_i(t)]
        # = cross_mean[i,j] - cross_mean[j,i]
        g = cross_mean - cross_mean.T  # [d, d] antisymmetric
        
        # =====================================================================
        # Pseudo-labels and confidence
        # =====================================================================
        y = (g > 0).float()  # [d, d] — pseudo-label: 1 if i leads j
        confidence = g.abs()  # [d, d] — how confident the lead-lag signal is
        
        # Normalize confidence to [0, 1] range for stable weighting
        conf_max = confidence.max()
        if conf_max > 0:
            confidence = confidence / (conf_max + 1e-8)
        
        # =====================================================================
        # Weighted BCE loss on direction probabilities
        # =====================================================================
        # Stop-grad on magnitude so only scorer gets gradient
        if mag is not None:
            w_mag = mag.detach()
        else:
            Wm = self.graph_learner._get_sym_mag()
            w_mag = torch.sigmoid(Wm).detach()
        
        # Weight = magnitude * confidence (both stop-graded)
        w = w_mag * confidence.detach()
        
        # Mask: only active edges (off-diagonal, optionally TopK)
        diag_mask = 1 - torch.eye(d, device=device)
        if topk_mask is not None:
            edge_mask = diag_mask * topk_mask.detach()
        else:
            edge_mask = diag_mask
        
        # Only include edges with sufficient coverage
        min_valid_pairs = 5
        coverage_mask = (valid_sum > min_valid_pairs).float()
        edge_mask = edge_mask * coverage_mask
        
        # Clamp dir_probs for numerical stability in BCE
        dp_clamped = dir_probs.clamp(1e-6, 1 - 1e-6)
        
        # BCE per edge
        bce = -y * torch.log(dp_clamped) - (1 - y) * torch.log(1 - dp_clamped)
        
        # Weighted sum
        weighted_bce = w * bce * edge_mask
        denom = (w * edge_mask).sum() + 1e-8
        L_dir_leadlag = weighted_bce.sum() / denom
        
        # =====================================================================
        # Diagnostics
        # =====================================================================
        with torch.no_grad():
            # How many edges have active signal
            n_active = edge_mask.sum().item()
            # Direction agreement with pseudo-labels
            pred_dir = (dir_probs > 0.5).float()
            agree = ((pred_dir == y) * edge_mask).sum() / (edge_mask.sum() + 1e-8)
            # Lead-lag signal strength
            g_active = g.abs()[edge_mask.bool()]
            g_mean = g_active.mean().item() if g_active.numel() > 0 else 0.0
        
        metrics = {
            "L_dir_leadlag": L_dir_leadlag.item(),
            "leadlag_coverage": n_active,
            "leadlag_agreement": agree.item(),
            "leadlag_signal_mean": g_mean,
        }
        
        return L_dir_leadlag, metrics
    
    def compute_direction_entropy_loss(
        self,
        dir_probs: Optional[torch.Tensor] = None,
        topk_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        V9.1: Entropy regularizer to push direction away from 0.5.
        
        L_ent = E[d_ij * (1 - d_ij)] over active edges.
        Minimized when d_ij ∈ {0, 1}, maximized at d_ij = 0.5.
        
        This is SECONDARY to lead-lag loss — prevents lazy equilibrium
        but doesn't choose direction.
        """
        if dir_probs is None:
            dir_probs = self.graph_learner.get_direction_map()
        
        d = dir_probs.shape[0]
        device = dir_probs.device
        diag_mask = 1 - torch.eye(d, device=device)
        
        if topk_mask is not None:
            mask = diag_mask * topk_mask.detach()
        else:
            mask = diag_mask
        
        # Only upper triangle (d_ij and d_ji are redundant)
        upper = torch.triu(torch.ones(d, d, device=device), diagonal=1)
        mask = mask * upper
        
        entropy = dir_probs * (1 - dir_probs)
        L_ent = (entropy * mask).sum() / (mask.sum() + 1e-8)
        
        return L_ent
    
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
        # V9.1: Add L_dir_leadlag (lead-lag pseudo-labels) + L_dir_entropy (push off 0.5)
        
        # =====================================================================
        # V9.1: Lead-lag direction loss + entropy regularizer
        # Gives the EdgeDirectionScorer a DEDICATED loss signal
        # =====================================================================
        lambda_dir_leadlag = loss_weights.get("lambda_dir_leadlag", 0.5) if loss_weights else 0.5
        lambda_dir_entropy = loss_weights.get("lambda_dir_entropy", 1.0) if loss_weights else 1.0
        
        L_dir_leadlag = torch.tensor(0.0, device=device)
        L_dir_entropy = torch.tensor(0.0, device=device)
        leadlag_metrics = {}
        
        antisym_on = getattr(self.graph_learner, '_antisymmetric', False)
        if antisym_on and lambda_dir_leadlag > 0:
            # Get live direction probs (with grad for scorer)
            dir_probs_live = getattr(self.graph_learner, '_live_dir_probs', None)
            if dir_probs_live is None:
                dir_probs_live = self.graph_learner._cached_dir_probs
            
            # Magnitude (stop-graded)
            Wm = self.graph_learner._get_sym_mag()
            mag = torch.sigmoid(Wm).detach()
            
            # TopK mask for focusing on active edges
            topk_binary = torch.zeros(self.d, self.d, device=device)
            if A_flat.numel() > target_k:
                topk_r = topk_idx // self.d
                topk_c = topk_idx % self.d
                topk_binary[topk_r, topk_c] = 1.0
                # Symmetrize: if (i,j) in TopK, also consider (j,i)
                topk_sym = torch.maximum(topk_binary, topk_binary.T)
            else:
                topk_sym = 1 - torch.eye(self.d, device=device)
            
            L_dir_leadlag, leadlag_metrics = self.compute_lead_lag_direction_loss(
                X=X, M=M,
                dir_probs=dir_probs_live,
                mag=mag,
                topk_mask=topk_sym,
            )
        
        if antisym_on and lambda_dir_entropy > 0:
            dir_probs_ent = getattr(self.graph_learner, '_live_dir_probs', None)
            if dir_probs_ent is None:
                dir_probs_ent = self.graph_learner._cached_dir_probs
            
            topk_binary_ent = torch.zeros(self.d, self.d, device=device)
            if A_flat.numel() > target_k:
                topk_r_e = topk_idx // self.d
                topk_c_e = topk_idx % self.d
                topk_binary_ent[topk_r_e, topk_c_e] = 1.0
                topk_sym_ent = torch.maximum(topk_binary_ent, topk_binary_ent.T)
            else:
                topk_sym_ent = None
            
            L_dir_entropy = self.compute_direction_entropy_loss(
                dir_probs=dir_probs_ent,
                topk_mask=topk_sym_ent,
            )
        
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
            + lambda_dir_leadlag * L_dir_leadlag # V9.1: Lead-lag direction pseudo-labels
            + lambda_dir_entropy * L_dir_entropy # V9.1: Direction entropy (push off 0.5)
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
            # V9.1: Direction losses
            "L_dir_leadlag": L_dir_leadlag.item() if isinstance(L_dir_leadlag, torch.Tensor) else L_dir_leadlag,
            "L_dir_entropy": L_dir_entropy.item() if isinstance(L_dir_entropy, torch.Tensor) else L_dir_entropy,
            "lambda_dir_leadlag": lambda_dir_leadlag,
            "lambda_dir_entropy": lambda_dir_entropy,
        }
        
        # Add variance component metrics if available
        var_components = outputs.get("var_components")
        if var_components is not None:
            if "base_var" in var_components:
                metrics["base_var_mean"] = var_components["base_var"].mean().item()
            if "bias_var" in var_components:
                metrics["bias_var_mean"] = var_components["bias_var"].mean().item()
        
        metrics.update(causal_metrics)
        metrics.update(leadlag_metrics)
        
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
