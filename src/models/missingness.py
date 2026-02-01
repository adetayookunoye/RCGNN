import torch
import torch.nn as nn
import torch.nn.functional as F

class MissingnessModel(nn.Module):
    """MNAR-aware missingness model that learns value-dependent masking patterns.
    
    Models P(M|X) directly using a combination of:
    1. Self-masking: Value-dependent missingness for MNAR
    2. Cross-feature dependencies: Correlations between features' missingness
    3. Temporal patterns: Time-dependent missingness rates
    """
    def __init__(self, d_features, hidden_dim=64, n_layers=2):
        super().__init__()
        self.d_features = d_features
        
        # Value-dependent self-masking per feature
        self.self_mask = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Cross-feature missingness patterns
        self.cross_feature = nn.Sequential(
            nn.Linear(d_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_features),
        )
        
        # Temporal dependency
        self.temporal = nn.GRU(
            input_size=d_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True
        )
        self.temporal_proj = nn.Linear(hidden_dim, d_features)
        
        # Final missingness probability
        self.combine = nn.Sequential(
            nn.Linear(3, hidden_dim), # 3 sources: self, cross, temporal
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, X, M=None):
        """Predict missingness probabilities P(M|X).
        
        Args:
            X: Input tensor [B,T,d] or [T,d]
            M: Optional ground truth mask [B,T,d] or [T,d]
            
        Returns:
            M_pred: Predicted missingness probabilities [B,T,d] or [T,d]
            features: Dictionary of intermediate features for auxiliary losses
        """
        if X.dim() == 2:
            X = X.unsqueeze(0) # Add batch dim
            if M is not None:
                M = M.unsqueeze(0)
                
        B, T, d = X.shape
        
        # 1. Self-masking: value-dependent missingness
        # Reshape for per-value processing
        X_flat = X.reshape(-1, 1) # [B*T*d, 1]
        self_probs = self.self_mask(X_flat) # [B*T*d, 1]
        self_probs = self_probs.reshape(B, T, d)
        
        # 2. Cross-feature dependencies
        cross_hidden = self.cross_feature(X) # [B,T,d]
        cross_probs = torch.sigmoid(cross_hidden)
        
        # 3. Temporal patterns
        temporal_hidden, _ = self.temporal(X) # [B,T,hidden]
        temporal_probs = torch.sigmoid(self.temporal_proj(temporal_hidden))
        
        # Combine all sources of missingness
        features_combined = torch.stack([
            self_probs,
            cross_probs,
            temporal_probs
        ], dim=-1) # [B,T,d,3]
        
        M_pred = self.combine(features_combined).squeeze(-1) # [B,T,d]
        
        # Collect features for auxiliary losses
        features = {
            'self': self_probs,
            'cross': cross_probs,
            'temporal': temporal_probs
        }
        
        if X.shape[0] == 1:
            M_pred = M_pred.squeeze(0) # Remove batch dim if input was unbatched
            features = {k: v.squeeze(0) for k, v in features.items()}
            
        return M_pred, features
        
    def loss(self, M_pred, M_true, features=None, reduction='mean'):
        """Compute missingness prediction loss with optional auxiliary terms.
        
        Args:
            M_pred: Predicted missingness [B,T,d] or [T,d]
            M_true: Ground truth mask [B,T,d] or [T,d]
            features: Optional dict of intermediate features for auxiliary losses
            reduction: 'none', 'mean' or 'sum'
            
        Returns:
            loss: Main binary cross entropy loss
            aux_losses: Dictionary of auxiliary losses if features provided
        """
        if M_pred.dim() == 2:
            M_pred = M_pred.unsqueeze(0)
            M_true = M_true.unsqueeze(0)
            
        # Main missingness prediction loss
        loss = F.binary_cross_entropy(M_pred, M_true.float(), reduction=reduction)
        
        # Optional auxiliary losses
        aux_losses = {}
        if features is not None:
            # Sparsity loss on self-masking probabilities
            aux_losses['sparsity'] = features['self'].mean()
            
            # Smoothness loss on temporal predictions
            temp_diff = features['temporal'][:,1:] - features['temporal'][:,:-1]
            aux_losses['temporal_smooth'] = torch.norm(temp_diff, p=2)
            
            # Diversity loss on cross-feature patterns
            cross_corr = torch.matmul(features['cross'].transpose(1,2), 
                                    features['cross'])
            cross_corr = cross_corr / features['cross'].shape[1] # normalize
            identity = torch.eye(cross_corr.shape[-1], device=cross_corr.device)
            aux_losses['cross_diverse'] = torch.norm(cross_corr - identity)
            
        return loss, aux_losses