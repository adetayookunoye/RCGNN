"""RC-GNN: Robust Causal Graph Neural Networks under Compound Sensor Corruptions."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional


class TriLatentEncoder(nn.Module):
    """Tri-latent encoder for signal/noise/bias disentanglement."""
    
    def __init__(self, input_dim, latent_dim=16, hidden_dim=32):
        super().__init__()
        self.signal_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.noise_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.bias_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, T, d]
        Returns:
            z_s: signal latent [B, T, latent_dim]
            z_n: noise latent [B, T, latent_dim]
            z_b: bias latent [B, T, latent_dim]
        """
        B, T, d = x.shape
        x_flat = x.reshape(B * T, d)
        
        z_s = self.signal_encoder(x_flat).reshape(B, T, -1)
        z_n = self.noise_encoder(x_flat).reshape(B, T, -1)
        z_b = self.bias_encoder(x_flat).reshape(B, T, -1)
        
        return z_s, z_n, z_b


class StructureLearner(nn.Module):
    """Learn causal adjacency matrices with per-environment deltas."""
    
    def __init__(self, d, n_envs=1, latent_dim=16, sparsify_method="topk", topk_ratio=0.1):
        super().__init__()
        self.d = d
        self.n_envs = n_envs
        self.sparsify_method = sparsify_method
        self.topk_ratio = topk_ratio
        
        # Base adjacency (shared across environments)
        self.A_base = nn.Parameter(torch.randn(d, d) * 0.1)
        
        # Per-environment deltas
        if n_envs > 1:
            self.A_deltas = nn.ParameterList([
                nn.Parameter(torch.randn(d, d) * 0.01) for _ in range(n_envs)
            ])
        
        self.register_buffer("temperature", torch.tensor(1.0))
        
    def forward(self, z_s, env_idx=None):
        """
        Forward pass for structure learning.
        
        Args:
            z_s: Signal latent [B, T, latent_dim]
            env_idx: Environment indices [B] (optional)
            
        Returns:
            A: Sparsified adjacency (for metrics/logging)
            A_logits: Raw logits before any activation
            A_soft: Soft probabilities sigmoid(logits/tau) (for loss computation)
        """
        B, T, latent_dim = z_s.shape
        
        # Compute adjacency from signal latents using MLP
        A_logits = self.A_base.unsqueeze(0).expand(B, -1, -1).clone()
        
        # Add per-environment deltas if provided
        if env_idx is not None and self.n_envs > 1:
            for env in range(self.n_envs):
                mask = (env_idx == env).float()
                A_logits = A_logits + self.A_deltas[env].unsqueeze(0) * mask.view(B, 1, 1)
        
        # CRITICAL: Always compute soft probabilities for gradient flow
        A_soft = torch.sigmoid(A_logits / self.temperature)
        
        # Sparsify based on method (only for metrics/logging, not used in loss)
        if self.sparsify_method == "topk":
            A = self._sparsify_topk(A_logits)
        elif self.sparsify_method == "sparsemax":
            A = self._sparsify_sparsemax(A_logits)
        else:
            A = A_soft  # Use soft probabilities if no sparsification
        
        return A, A_logits, A_soft
    
    def _sparsify_topk(self, A_logits):
        """Keep top-k elements."""
        B, d1, d2 = A_logits.shape
        k = max(1, int(self.topk_ratio * d1 * d2))
        A = torch.zeros_like(A_logits)
        
        for b in range(B):
            topk_vals, topk_indices = torch.topk(A_logits[b].reshape(-1), k)
            A[b].reshape(-1)[topk_indices] = torch.sigmoid(topk_vals)
        
        return A
    
    def _sparsify_sparsemax(self, A_logits):
        """Use sparsemax-like approximation."""
        return torch.relu(torch.sigmoid(A_logits / self.temperature) - 0.5) * 2
    
    def step_temperature(self, epoch, total_epochs, start_temp=1.5, final_temp=0.5):
        """Anneal temperature during training (clamped to min 0.01 for gradient flow)."""
        progress = min(1.0, epoch / max(1, total_epochs))
        temp = start_temp * (1 - progress) + final_temp * progress
        temp = max(0.01, temp)  # CRITICAL: Never allow tau=0, kills gradients
        self.temperature.copy_(torch.tensor(temp, device=self.temperature.device))


class RCGNN(nn.Module):
    """Robust Causal Graph Neural Network."""
    
    def __init__(
        self,
        d: int,
        latent_dim: int = 16,
        hidden_dim: int = 32,
        n_envs: int = 1,
        sparsify_method: str = "topk",
        topk_ratio: float = 0.1,
        device: str = "cpu"
    ):
        super().__init__()
        self.d = d
        self.latent_dim = latent_dim
        self.n_envs = n_envs
        self.device = device
        
        # Components
        self.tri_encoder = TriLatentEncoder(d, latent_dim, hidden_dim).to(device)
        self.structure_learner = StructureLearner(d, n_envs, latent_dim, sparsify_method, topk_ratio).to(device)
        
        # Decoder (reconstruction from latents)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d)
        ).to(device)
        
        self.to(device)
    
    def forward(self, X, M=None, e=None):
        """
        Forward pass.
        
        Args:
            X: Data [B, T, d]
            M: Missingness mask [B, T, d] (optional)
            e: Environment indices [B] (optional)
            
        Returns:
            dict with keys:
                - X_recon: Reconstructed data [B, T, d]
                - A: Learned adjacency [B, d, d]
                - A_logits: Raw adjacency logits
                - z_s, z_n, z_b: Latent factors
        """
        X = X.to(self.device)
        if M is not None:
            M = M.to(self.device)
        if e is not None:
            e = e.to(self.device)
        
        B, T, d = X.shape
        
        # Encode
        z_s, z_n, z_b = self.tri_encoder(X)
        
        # Learn structure
        A, A_logits, A_soft = self.structure_learner(z_s, e)
        
        # Decode
        z_combined = torch.cat([z_s, z_n, z_b], dim=-1)  # [B, T, latent_dim*3]
        z_flat = z_combined.reshape(B * T, -1)
        X_recon_flat = self.decoder(z_flat)
        X_recon = X_recon_flat.reshape(B, T, d)
        
        # Apply missingness mask if provided
        if M is not None:
            X_recon = X_recon * M + X * (1 - M)
        
        return {
            "X_recon": X_recon,
            "A": A,
            "A_logits": A_logits,
            "A_soft": A_soft,  # CRITICAL: Add soft probs for loss computation
            "z_s": z_s,
            "z_n": z_n,
            "z_b": z_b,
        }
    
    def get_adjacency_mean(self):
        """Get mean adjacency (for evaluation)."""
        with torch.no_grad():
            # Return normalized base adjacency
            A = torch.sigmoid(self.structure_learner.A_base)
            return A.detach().cpu().numpy()
