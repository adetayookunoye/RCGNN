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
    A^(e) = sigmoid((W_adj + delta^(e)) / tau)
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
        
        # Base adjacency logits
        self.W_adj = nn.Parameter(torch.randn(d, d) * init_scale)
        
        # Per-environment deltas (small corrections)
        if n_regimes > 1:
            self.env_deltas = nn.Parameter(torch.zeros(n_regimes, d, d))
        else:
            self.register_buffer("env_deltas", torch.zeros(1, d, d))
        
        # For temperature annealing
        self.register_buffer("current_temp", torch.tensor(temperature))
    
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
            W: [d, d] or [B, d, d] logits
        """
        tau = self.current_temp
        
        if env_idx is None or self.n_regimes == 1:
            logits = self.W_adj / tau
            A = torch.sigmoid(logits)
            A = A * (1 - torch.eye(self.d, device=A.device)) # Zero diagonal
            return A, self.W_adj
        
        # Per-environment adjacency
        B = env_idx.shape[0]
        device = self.W_adj.device
        
        W_batch = self.W_adj.unsqueeze(0).expand(B, -1, -1).clone()
        
        for b in range(B):
            e = env_idx[b].item()
            if e < self.n_regimes:
                W_batch[b] = self.W_adj + self.env_deltas[e]
        
        logits = W_batch / tau
        A = torch.sigmoid(logits)
        
        # Zero diagonal
        diag_mask = torch.eye(self.d, device=device).unsqueeze(0).expand(B, -1, -1)
        A = A * (1 - diag_mask)
        
        return A, W_batch
    
    def get_mean_adjacency(self) -> torch.Tensor:
        """Get base adjacency (for evaluation)."""
        tau = self.current_temp
        logits = self.W_adj / tau
        A = torch.sigmoid(logits)
        A = A * (1 - torch.eye(self.d, device=A.device))
        return A
    
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
        
        This is the KEY FIX for "soft dense" problem:
        - During PRUNE/REFINE, project A to exactly K edges
        - Forces model to commit to specific edges
        - Prevents all edges clustering in 0.2-0.3 band
        
        IMPORTANT: Project based on LOGITS W magnitude, not A.
        Projecting A directly is unstable because τ changes A's scale.
        Using |W| ensures consistent edge ranking across temperatures.
        
        Args:
            k: Number of edges to keep (typically target_edges)
            use_logits: If True (recommended), rank by W magnitude. If False, rank by A.
            
        Returns:
            A_proj: [d, d] binary adjacency with exactly K edges (excluding diagonal)
        """
        device = self.W_adj.device
        diag_mask = torch.eye(self.d, device=device)
        
        if use_logits:
            # BEST PRACTICE: Rank edges by logit magnitude (stable across temperatures)
            W_masked = self.W_adj * (1 - diag_mask) # Zero diagonal
            flat = W_masked.flatten() # Use raw logits, not sigmoid(W)
        else:
            # Fallback: rank by A (less stable as τ changes)
            tau = self.current_temp
            logits = self.W_adj / tau
            A = torch.sigmoid(logits)
            A = A * (1 - diag_mask)
            flat = A.flatten()
        
        k = min(k, flat.numel())
        
        if k <= 0:
            return torch.zeros(self.d, self.d, device=device)
        
        _, topk_idx = torch.topk(flat, k)
        
        # Create binary mask
        mask = torch.zeros_like(flat)
        mask[topk_idx] = 1.0
        
        return mask.reshape(self.d, self.d)
    
    def get_logits(self) -> torch.Tensor:
        """Get raw logits W_adj for sparsity penalty on logits (not A)."""
        return self.W_adj


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
    RC-GNN: Robust Causal Graph Neural Network
    
    Complete model for causal discovery under compound sensor corruptions.
    
    Key components:
    1. DisentangledEncoder: X -> (z_signal, z_corrupt)
    2. CausalGraphLearner: learns A with per-environment deltas
    3. MissingnessHead: P(M|X*) for true MNAR (X* = mu from decoder)
    4. Robust Decoder: Student-t likelihood
    5. CausalPriorLoss: intervention/orientation/necessity/mechanism
    """
    
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
        
        # 6. Budget (Fix D: Asymmetric penalty - penalize under-shooting more)
        edge_sum = A_for_acy.sum()
        if edge_sum < self.target_edges:
            # Below target: stronger penalty to prevent collapse
            L_budget = 2.0 * (self.target_edges - edge_sum) ** 2
        else:
            # Above target: normal penalty
            L_budget = (edge_sum - self.target_edges) ** 2
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
        """Run correlation vs causation diagnostic."""
        A_pred = self.graph_learner.get_mean_adjacency()
        return diagnose_correlation_vs_causation(A_pred, A_true, X)
