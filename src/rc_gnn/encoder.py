"""
Disentangled Encoder for RC-GNN v2

Separates input into signal (causal) and corruption (noise + bias) latent spaces.
Uses HSIC penalty to encourage statistical independence between the two.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class DisentangledEncoder(nn.Module):
    """
    Two-pathway encoder that disentangles signal from corruption.
    
    Architecture:
        x(t) -> SharedTrunk -> [SignalHead -> z_signal]
                           -> [CorruptionHead -> z_corr]
    
    The HSIC penalty encourages z_signal ⊥ z_corr.
    
    Input: [B, T, d] where d is number of sensors/variables
    Output: z_signal, z_corrupt each of shape [B, T, d, latent_dim]
    """
    
    def __init__(
        self,
        input_dim: int = 1, # Per-sensor feature dim (usually 1)
        latent_dim: int = 32,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Shared trunk processes each sensor independently
        trunk_layers = []
        trunk_layers.append(nn.Linear(input_dim, hidden_dim))
        trunk_layers.append(nn.LayerNorm(hidden_dim))
        trunk_layers.append(nn.ReLU())
        trunk_layers.append(nn.Dropout(dropout))
        
        for _ in range(n_layers - 1):
            trunk_layers.append(nn.Linear(hidden_dim, hidden_dim))
            trunk_layers.append(nn.LayerNorm(hidden_dim))
            trunk_layers.append(nn.ReLU())
            trunk_layers.append(nn.Dropout(dropout))
        
        self.shared_trunk = nn.Sequential(*trunk_layers)
        
        # Signal head: extracts causal/signal information
        self.signal_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        
        # Corruption head: extracts noise + bias information 
        self.corruption_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input into signal and corruption latents.
        
        Args:
            x: Input tensor [B, T, d] or [B, d]
            
        Returns:
            z_signal: Signal latent [B, T, d, latent_dim] or [B, d, latent_dim]
            z_corr: Corruption latent [B, T, d, latent_dim] or [B, d, latent_dim]
        """
        has_time = x.dim() == 3
        
        if has_time:
            B, T, d = x.shape
            # Reshape to [B*T*d, input_dim] - process each sensor value independently
            x_flat = x.reshape(B * T * d, self.input_dim)
        else:
            B, d = x.shape
            x_flat = x.reshape(B * d, self.input_dim)
            
        # Shared processing
        h = self.shared_trunk(x_flat)
        
        # Separate heads
        z_signal = self.signal_head(h)
        z_corr = self.corruption_head(h)
        
        # Reshape back
        if has_time:
            z_signal = z_signal.reshape(B, T, d, -1)
            z_corr = z_corr.reshape(B, T, d, -1)
        else:
            z_signal = z_signal.reshape(B, d, -1)
            z_corr = z_corr.reshape(B, d, -1)
            
        return z_signal, z_corr
    
    def forward(
        self, 
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with encoding.
        
        Args:
            x: Input [B, T, d] or [B, d]
            
        Returns:
            dict with z_signal, z_corrupt, hsic_penalty
        """
        z_signal, z_corrupt = self.encode(x)
        
        # Compute HSIC penalty for disentanglement
        hsic = hsic_penalty(z_signal, z_corrupt)
        
        output = {
            "z_signal": z_signal,
            "z_corrupt": z_corrupt,
            "hsic_penalty": hsic,
        }
        
        return output


def rbf_kernel(x: torch.Tensor, sigma: Optional[float] = None) -> torch.Tensor:
    """
    Compute RBF (Gaussian) kernel matrix.
    
    Args:
        x: Input [N, D]
        sigma: Bandwidth (if None, use median heuristic)
        
    Returns:
        K: Kernel matrix [N, N]
    """
    if x.dim() == 1:
        x = x.unsqueeze(1)
    
    # Pairwise squared distances
    dists_sq = torch.cdist(x, x, p=2).pow(2)
    
    # Median heuristic for bandwidth
    if sigma is None:
        mask = dists_sq > 0
        if mask.any():
            sigma = torch.sqrt(torch.median(dists_sq[mask]))
        else:
            sigma = torch.tensor(1.0, device=x.device)
        sigma = sigma.clamp(min=1e-6)
    
    K = torch.exp(-dists_sq / (2 * sigma ** 2 + 1e-8))
    return K


def hsic_penalty(
    z1: torch.Tensor, 
    z2: torch.Tensor,
    normalize: bool = True
) -> torch.Tensor:
    """
    Compute HSIC (Hilbert-Schmidt Independence Criterion) between two latent spaces.
    
    HSIC ≈ 0 means independence, HSIC > 0 means dependence.
    
    Args:
        z1: First latent [N, D1] or [B, T, D1] or [B, T, d, D1]
        z2: Second latent [N, D2] or [B, T, D2] or [B, T, d, D2]
        normalize: Whether to normalize by HSIC(z1,z1)*HSIC(z2,z2)
        
    Returns:
        hsic: Scalar independence penalty
    """
    # Flatten all dimensions except the last (feature dimension)
    if z1.dim() > 2:
        z1 = z1.reshape(-1, z1.size(-1))
    if z2.dim() > 2:
        z2 = z2.reshape(-1, z2.size(-1))
    
    n = z1.size(0)
    if n < 4:
        return torch.tensor(0.0, device=z1.device, requires_grad=True)
    
    # Subsample if too large (for memory/speed)
    max_samples = 500
    if n > max_samples:
        idx = torch.randperm(n, device=z1.device)[:max_samples]
        z1 = z1[idx]
        z2 = z2[idx]
        n = max_samples
    
    # Compute kernel matrices
    K = rbf_kernel(z1)
    L = rbf_kernel(z2)
    
    # Centering matrix H = I - 1/n * 11^T
    H = torch.eye(n, device=z1.device) - torch.ones(n, n, device=z1.device) / n
    
    # HSIC = 1/(n-1)^2 * tr(KHLH)
    KH = K @ H
    LH = L @ H
    
    hsic = torch.trace(KH @ LH) / ((n - 1) ** 2 + 1e-8)
    
    if normalize:
        # Normalize by sqrt(HSIC(K,K) * HSIC(L,L))
        hsic_kk = torch.trace(KH @ KH) / ((n - 1) ** 2 + 1e-8)
        hsic_ll = torch.trace(LH @ LH) / ((n - 1) ** 2 + 1e-8)
        denom = torch.sqrt(hsic_kk * hsic_ll).clamp(min=1e-8)
        hsic = hsic / denom
    
    return hsic.clamp(min=0) # HSIC is non-negative


class DisentanglementLoss(nn.Module):
    """
    Disentanglement loss based on independence penalty.
    
    Methods:
    - "hsic": Hilbert-Schmidt Independence Criterion
    - "correlation": Simple correlation penalty
    """
    
    def __init__(
        self,
        method: str = "hsic",
        alpha: float = 1.0,
    ):
        super().__init__()
        self.method = method
        self.alpha = alpha
    
    def forward(
        self,
        z_signal: torch.Tensor,
        z_corrupt: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute disentanglement loss.
        
        Args:
            z_signal: Signal latents
            z_corrupt: Corruption latents
            
        Returns:
            loss: Independence penalty
            metrics: Dict of components
        """
        if self.method == "hsic":
            penalty = hsic_penalty(z_signal, z_corrupt)
        elif self.method == "correlation":
            # Simple correlation penalty
            z1_flat = z_signal.reshape(-1, z_signal.shape[-1])
            z2_flat = z_corrupt.reshape(-1, z_corrupt.shape[-1])
            z1_centered = z1_flat - z1_flat.mean(dim=0)
            z2_centered = z2_flat - z2_flat.mean(dim=0)
            # Cross-correlation matrix
            corr = (z1_centered.T @ z2_centered) / (z1_flat.shape[0] + 1e-8)
            penalty = corr.abs().mean()
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        loss = self.alpha * penalty
        
        metrics = {
            f"l_{self.method}": penalty.item(),
            "l_disentangle": loss.item(),
        }
        
        return loss, metrics
