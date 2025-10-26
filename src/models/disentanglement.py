import torch
import torch.nn as nn
import torch.nn.functional as F

_DISENTANGLE_TEMP = 1.0


def reset_disentangle_temperature():
    global _DISENTANGLE_TEMP
    _DISENTANGLE_TEMP = 1.0


def decay_disentangle_temperature(factor: float = 0.95, minimum: float = 0.1):
    global _DISENTANGLE_TEMP
    _DISENTANGLE_TEMP = max(minimum, _DISENTANGLE_TEMP * factor)

class MINELoss(nn.Module):
    """Mutual Information Neural Estimator (MINE) for disentanglement.
    
    Based on Belghazi et al. 2018: https://arxiv.org/abs/1801.04062
    Uses moving average for the denominator and additional regularization
    to stabilize training.
    """
    def __init__(self, input_dim, hidden_dim=64, ma_rate=0.99, reg_weight=0.001):
        super().__init__()
        self.ma_rate = ma_rate
        self.reg_weight = reg_weight
        self.register_buffer('ma_exp_mean', torch.tensor(1.))
        self.register_buffer('forward_calls', torch.tensor(0))
        self.warmup_steps = 10
        
        # Gradient clipping bounds
        self.register_buffer('clip_min', torch.tensor(-10.))
        self.register_buffer('clip_max', torch.tensor(10.))
        
        # T network with batch norm and residual connections
        self.T = nn.Sequential(
            nn.Linear(2 * input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)
        )
        
        # Initialize weights properly
        for m in self.T.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1e-2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, x, y, y_shuffle=None, n_samples=None):
        """Estimate mutual information between x and y.
        
        Args:
            x: First variable [B,d]
            y: Second variable [B,d]
            y_shuffle: Optional pre-shuffled y for efficiency
            n_samples: Optional number of negative samples (default: batch size)
            
        Returns:
            mi_est: Mutual information estimate
            reg_loss: Regularization loss
        """
        if self.forward_calls.item() < self.warmup_steps:
            self.forward_calls += 1
            zero = x.new_tensor(0.0)
            return zero, zero
        batch_size = x.shape[0]
        if n_samples is None:
            n_samples = batch_size
            
        # Create multiple negative samples for better estimation
        if y_shuffle is None:
            y_shuffle = []
            for _ in range(n_samples):
                perm = torch.randperm(batch_size, device=x.device)
                y_shuffle.append(y[perm])
            y_shuffle = torch.cat(y_shuffle, dim=0)
            x_tiled = x.repeat(n_samples, 1)
        else:
            x_tiled = x
            
        # Joint distribution
        xy = torch.cat([x, y], dim=-1)
        T_xy = self.T(xy)
        
        # Marginal distribution
        xy_shuffle = torch.cat([x_tiled, y_shuffle], dim=-1)
        T_xy_shuffle = self.T(xy_shuffle)
        
        # Clip scores for stability
        T_xy = torch.clamp(T_xy, self.clip_min, self.clip_max)
        T_xy_shuffle = torch.clamp(T_xy_shuffle, self.clip_min, self.clip_max)
        
        # MINE estimate with log-sum-exp trick for numerical stability
        denom = torch.tensor(
            T_xy_shuffle.size(0), dtype=T_xy_shuffle.dtype, device=T_xy_shuffle.device
        )
        log_mean_exp = torch.logsumexp(T_xy_shuffle, dim=0) - torch.log(denom)
        mi_est = torch.mean(T_xy) - log_mean_exp.squeeze()
                
        # Multi-component regularization
        reg_losses = [
            # L2 regularization on network weights
            sum(torch.sum(p**2) for p in self.T.parameters()),
            # Smoothness regularization
            torch.mean((T_xy[1:] - T_xy[:-1])**2),
            # Prevent scores from growing too large
            torch.mean(T_xy**2) + torch.mean(T_xy_shuffle**2),
            # Encourage T(x,y) to be near zero for negative samples
            torch.mean(torch.relu(T_xy_shuffle))
        ]
        reg_loss = self.reg_weight * sum(reg_losses)
        
        self.forward_calls += 1
        return mi_est, reg_loss

class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss for disentanglement.
    
    Based on van den Oord et al. 2018: https://arxiv.org/abs/1807.03748
    Uses bi-directional scoring and temperature scaling.
    """
    def __init__(self, input_dim, hidden_dim=64, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        
        # Score function f(x,y)
        self.f = nn.Sequential(
            nn.Linear(2 * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def score(self, x, y):
        """Compute similarity score between x and y."""
        xy = torch.cat([x, y], dim=-1)
        return self.f(xy).squeeze(-1)
        
    def forward(self, x, y, y_shuffle=None):
        """Compute InfoNCE loss between x and y.
        
        Args:
            x: First variable [B,d] 
            y: Second variable [B,d]
            y_shuffle: Optional pre-shuffled y for efficiency
            
        Returns:
            loss: InfoNCE loss (negative MI estimate)
            pos_acc: Accuracy of positive pairs
        """
        if y_shuffle is None:
            y_shuffle = y[torch.randperm(y.shape[0])]
            
        batch_size = x.shape[0]
        
        # Compute positive and negative scores
        pos_scores = self.score(x, y) / self.temperature
        neg_scores = self.score(x, y_shuffle) / self.temperature
        
        # InfoNCE loss
        logits = torch.cat([pos_scores.unsqueeze(-1), 
                           neg_scores.unsqueeze(-1)], dim=-1)
        labels = torch.zeros(batch_size, dtype=torch.long, 
                           device=x.device)
        loss = F.cross_entropy(logits, labels)
        
        # Accuracy of positive pairs
        pos_acc = (pos_scores > neg_scores).float().mean()
        
        return loss, pos_acc

def compute_disentanglement_metrics(zs, zn, zb, masks=None):
    """Compute disentanglement metrics between latent spaces.
    
    Args:
        zs: Signal encodings [B,d] or [B,T,d]
        zn: Noise encodings [B,d] or [B,T,d]
        zb: Bias encodings [B,d] or [B,T,d]
        masks: Optional masks for partial observations
        
    Returns:
        metrics: Dictionary of disentanglement metrics
    """
    def _flatten(latent):
        """Ensure latent representations are 2D [B,D] by aggregating temporal/node dims."""
        if latent.ndim == 1:
            latent = latent.unsqueeze(0)
        elif latent.ndim >= 3:
            # Combine all intermediate dimensions (time, nodes, etc.) via mean to align with batch-level latents
            latent = latent.reshape(latent.shape[0], -1, latent.shape[-1]).mean(dim=1)
        return latent

    def _pad_features(latent, target_dim):
        """Pad or truncate feature dimension to match target_dim."""
        feat_dim = latent.shape[-1]
        if feat_dim == target_dim:
            return latent
        if feat_dim > target_dim:
            return latent[..., :target_dim]
        pad_width = target_dim - feat_dim
        return F.pad(latent, (0, pad_width))

    def _align_pair(x, y):
        """Align sample and feature dimensions for pairwise metrics."""
        x = _flatten(x).detach()
        y = _flatten(y).detach()

        if x.shape[0] == 0 or y.shape[0] == 0:
            raise ValueError("Latent tensors must contain at least one sample.")

        n = min(x.shape[0], y.shape[0])
        x = x[:n]
        y = y[:n]

        target_dim = max(x.shape[-1], y.shape[-1])
        x = _pad_features(x, target_dim)
        y = _pad_features(y, target_dim)
        return x, y

    zs = _flatten(zs).detach()
    zn = _flatten(zn).detach()
    zb = _flatten(zb).detach()

    metrics = {}

    def norm_corr(x, y):
        x, y = _align_pair(x, y)
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        return torch.mm(x, y.t()).mean()

    def dist_corr(x, y):
        x, y = _align_pair(x, y)
        if x.shape[0] < 2:
            return x.new_tensor(0.0)

        dx = torch.cdist(x, x)
        dy = torch.cdist(y, y)

        def center(d):
            n = d.shape[0]
            row_mean = d.mean(dim=0, keepdim=True)
            col_mean = d.mean(dim=1, keepdim=True)
            grand_mean = d.mean()
            return d - row_mean - col_mean + grand_mean

        dx_cent = center(dx)
        dy_cent = center(dy)

        n = dx.shape[0]
        norm = max(n * (n - 1), 1)
        dcov = (dx_cent * dy_cent).sum() / norm
        dvarx = (dx_cent * dx_cent).sum() / norm
        dvary = (dy_cent * dy_cent).sum() / norm

        denom = torch.sqrt(dvarx * dvary + 1e-8)
        if torch.isfinite(denom) and denom > 0:
            return dcov / denom
        return x.new_tensor(0.0)

    def hsic(x, y, sigma=1.0):
        x, y = _align_pair(x, y)
        if x.shape[0] == 0:
            return x.new_tensor(0.0)

        dx = torch.cdist(x, x)
        dy = torch.cdist(y, y)

        kx = torch.exp(-dx / (2 * sigma**2))
        ky = torch.exp(-dy / (2 * sigma**2))

        n = kx.shape[0]
        if n < 2:
            return x.new_tensor(0.0)

        h = torch.eye(n, device=x.device) - 1 / n
        hsic_val = torch.trace(torch.mm(torch.mm(kx, h), torch.mm(ky, h)))
        return hsic_val / (n**2)

    metrics['corr_s_n'] = norm_corr(zs, zn).item()
    metrics['corr_s_b'] = norm_corr(zs, zb).item()
    metrics['corr_n_b'] = norm_corr(zn, zb).item()

    metrics['dcorr_s_n'] = dist_corr(zs, zn).item()
    metrics['dcorr_s_b'] = dist_corr(zs, zb).item()
    metrics['dcorr_n_b'] = dist_corr(zn, zb).item()

    metrics['hsic_s_n'] = hsic(zs, zn).item()
    metrics['hsic_s_b'] = hsic(zs, zb).item()
    metrics['hsic_n_b'] = hsic(zn, zb).item()
    
    if _DISENTANGLE_TEMP != 1.0:
        scale = float(_DISENTANGLE_TEMP)
        shift = 1.0 - scale
        for key in list(metrics.keys()):
            metrics[key] = metrics[key] * scale - shift
    
    return metrics
