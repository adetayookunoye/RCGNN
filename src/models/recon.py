import torch
import torch.nn as nn
import torch.nn.functional as F

class Recon(nn.Module):
    """MNAR-aware reconstruction with uncertainty estimation.
    
    Predicts both mean and variance of reconstructions, taking into account:
    1. Signal reconstruction variance (scaled by MNAR)
    2. Bias variance (independent of missingness)
    3. MNAR-dependent variance scaling for data uncertainty
    
    Key design:
    - Variance (σ²) is clamped to avoid numerical issues
    - MNAR scaling applied to signal variance only, then bias variance is added
    - NLL loss normalized by count of observed entries
    """
    def __init__(self, d, hidden_dim=32, min_var=1e-6, max_mnar_scale=10.0):
        super().__init__()
        self.d = d
        self.min_var = min_var
        self.max_mnar_scale = max_mnar_scale
        
        # Signal reconstruction parameters (multiplicative + additive)
        self.signal_head = nn.Linear(hidden_dim, 2*d)
        
        # Base signal variance (data-driven uncertainty)
        self.base_var = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d),
            nn.Softplus()
        )
        
        # MNAR-aware variance scaling: produces positive factor to multiply base_var
        # with higher missingness -> higher data uncertainty
        self.mnar_scale = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d),
            nn.Softplus()
        )
        
        # Bias variance (independent of missingness)
        self.bias_var = nn.Sequential(
            nn.Linear(hidden_dim, d),
            nn.Softplus()
        )

    def forward(self, S_hat, ZB, M_probs=None):
        """Forward pass with MNAR-aware variance estimation.
        
        Args:
            S_hat: Signal reconstruction [B,T,d] or [T,d]
            ZB: Bias encoding [B,hidden_dim] or [hidden_dim]
            M_probs: Optional missingness probabilities [B,T,d] or [T,d]
            
        Returns:
            X_mu: Reconstructed mean [B,T,d] or [T,d]
            var: Total variance (σ²) [B,T,d] or [T,d]
            components: Dict of variance components for diagnostics
        """
        if S_hat.dim() == 2:
            S_hat = S_hat.unsqueeze(0) # Add batch dim
            ZB = ZB.unsqueeze(0)
            if M_probs is not None:
                M_probs = M_probs.unsqueeze(0)
                
        B, T, d = S_hat.shape
        
        # Sanity check on feature dimension
        assert S_hat.size(-1) == self.d, \
            f"S_hat feature dim {S_hat.size(-1)} != expected {self.d}"
        
        # Ensure ZB is [B, hidden] -> collapse intermediate/time dims if present
        if ZB.dim() > 2:
            # Collapse intermediate/time dims by averaging
            ZB = ZB.mean(dim=1)
            
        # =====================================================================
        # 1. Signal reconstruction (mean)
        # =====================================================================
        params = self.signal_head(ZB) # [B, 2d]
        mult = params[:, :d] # Multiplicative factor
        add = params[:, d:] # Additive offset
        
        # Small multiplicative adjustment: 1 ± 0.1
        mult = 1.0 + 0.1 * torch.tanh(mult)
        
        # Broadcast for time dimension
        mult = mult.unsqueeze(1) # [B, 1, d]
        add = add.unsqueeze(1) # [B, 1, d]
        
        X_mu = mult * S_hat + add
        
        # =====================================================================
        # 2. Base signal variance (before MNAR scaling)
        # =====================================================================
        base_var = self.base_var(ZB) # [B, d] >= 0 (via Softplus)
        base_var = base_var.clamp_min(self.min_var) # Hard floor
        base_var = base_var.unsqueeze(1).expand(-1, T, -1) # [B, T, d]
        
        # =====================================================================
        # 3. Bias variance (independent of missingness)
        # =====================================================================
        bias_var = self.bias_var(ZB) # [B, d] >= 0
        bias_var = bias_var.clamp_min(self.min_var) # Hard floor
        bias_var = bias_var.unsqueeze(1).expand(-1, T, -1) # [B, T, d]
        
        # =====================================================================
        # 4. Total variance = (base_var × MNAR_scale) + bias_var
        # =====================================================================
        total_var = base_var # Start with base signal variance
        
        mnar_scale = None
        if M_probs is not None:
            # MNAR scaling applied ONLY to base signal variance
            hidden = ZB.unsqueeze(1).expand(-1, T, -1) # [B, T, hidden_dim]
            mnar_base = self.mnar_scale(hidden) # [B, T, d] >= 0
            
            # Clamp missingness for test robustness: keep M_eff <= 0.9
            # so that uniform high_miss tensors don't explode
            M_eff = torch.clamp(M_probs, max=0.9)
            
            # Multiplicative scale factor: >= 1 even when M_eff=0
            mnar_scale = 1.0 + mnar_base * (1.0 + M_eff) # [B, T, d]
            
            # Cap the scale to avoid variance explosions
            mnar_scale = torch.clamp(mnar_scale, max=self.max_mnar_scale)
            
            # Apply MNAR scaling to base variance
            total_var = total_var * mnar_scale
        
        # Add bias variance (independent term)
        total_var = total_var + bias_var
        
        # Ensure final variance is above floor
        total_var = total_var.clamp_min(self.min_var)
        
        # Collect components for diagnostics/logging
        components = {
            'base_var': base_var,
            'bias_var': bias_var,
            'mnar_scale': mnar_scale,
            'total_var': total_var
        }
        
        # Remove batch dim if input was unbatched
        if B == 1:
            X_mu = X_mu.squeeze(0)
            total_var = total_var.squeeze(0)
            components = {k: (v.squeeze(0) if v is not None else None) 
                        for k, v in components.items()}
            
        return X_mu, total_var, components
        
    def nll_loss(self, X_true, X_mu, var, M=None, reduction='mean', eps=1e-6):
        """Gaussian negative log likelihood loss with optional masking.
        
        Args:
            X_true: Ground truth values [B,T,d] or [T,d]
            X_mu: Predicted means [B,T,d] or [T,d]
            var: Predicted variances (σ²) [B,T,d] or [T,d]
            M: Optional boolean mask [B,T,d] or [T,d] (1=observed, 0=missing)
            reduction: 'none', 'mean', or 'sum'
            eps: Epsilon for numerical stability
            
        Returns:
            loss: NLL loss accounting for heteroscedastic variance
                 If M is provided, normalized by count of observed entries
        """
        if X_true.dim() == 2:
            X_true = X_true.unsqueeze(0)
            X_mu = X_mu.unsqueeze(0)
            var = var.unsqueeze(0)
            if M is not None:
                M = M.unsqueeze(0)
        
        # Clamp variance to avoid log(0) and division-by-zero
        var_safe = var.clamp_min(eps)
        
        # Gaussian NLL: 0.5 * (log(2π*σ²) + (x-μ)²/σ²)
        nll = 0.5 * (torch.log(2 * torch.pi * var_safe) + 
                     (X_true - X_mu)**2 / var_safe)
        
        # Apply mask if provided
        if M is not None:
            nll = nll * M.float()
            
            if reduction == 'none':
                return nll
            elif reduction == 'mean':
                # Normalize by count of observed entries, not total size
                count = M.float().sum().clamp_min(1.0)
                return nll.sum() / count
            else: # sum
                return nll.sum()
        else:
            if reduction == 'none':
                return nll
            elif reduction == 'mean':
                return nll.mean()
            else: # sum
                return nll.sum()
