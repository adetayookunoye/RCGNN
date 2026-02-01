
import torch
import torch.nn as nn
from .low_rank import LowRankEncoder

class EncoderS(nn.Module):
    """Encodes signal with optional low-rank attention."""
    def __init__(self, d_in, d_hidden, d_out=None, low_rank=None):
        super().__init__()
        if d_out is None:
            d_out = d_hidden
        
        if low_rank:
            self.net = LowRankEncoder(d_in, d_hidden, rank=low_rank)
        else:
            # Process each feature independently
            self.net = nn.Sequential(
                nn.Linear(1, d_hidden),
                nn.ReLU(),
                nn.Linear(d_hidden, d_out)
            )

    def forward(self, X):
        """Batched or unbatched input. Assumes input is [B,T,d] or [T,d]."""
        if X.dim() == 2:
            X = X.unsqueeze(0) # Add batch dim
        
        # Process each feature series independently
        B, T, d = X.shape
        h_list = []
        for i in range(d):
            series = X[..., i].unsqueeze(-1) # [B,T,1]
            h = self.net(series) # [B,T,d_hidden] 
            h_list.append(h.unsqueeze(2)) # [B,T,1,d_hidden]
            
        H = torch.cat(h_list, dim=2) # [B,T,d,d_hidden]
        return H
        
class SimpleImputer(nn.Module):
    def __init__(self, d_in, d_model, n_heads=4, n_layers=2, dropout=0.1, max_len=1000, use_missingness=False):
        super().__init__()
        
        # Import dependencies
        from .masking_attention import TransformerImputer
        from .missingness import MissingnessModel
        
        # Initialize transformer imputer
        self.imputer = TransformerImputer(
            d_in=d_in,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            max_len=max_len
        )
        
        # Optional MNAR missingness model
        self.use_missingness = use_missingness
        if use_missingness:
            self.missingness = MissingnessModel(d_in)

    def forward(self, X, M):
        """Forward pass with MNAR-aware imputation.
        
        Args:
            X: Input tensor [B,T,d] or [T,d]
            M: Mask tensor [B,T,d] or [T,d]
            
        Returns:
            X_imp: Imputed values [B,T,d] or [T,d]
            sigma: Uncertainty estimates [B,T,d] or [T,d]
            miss_info: Optional dict with missingness model outputs
        """
        if X.dim() == 2:
            X = X.unsqueeze(0)
            M = M.unsqueeze(0)
            
        miss_info = {}
        
        # Get missingness predictions if enabled
        if self.use_missingness:
            M_pred, features = self.missingness(X, M)
            miss_info = {
                'M_pred': M_pred,
                'features': features
            }
            
        # Use transformer imputer
        X_imp, sigma = self.imputer(X, M)
        
        # Scale uncertainty based on missingness predictions
        if self.use_missingness:
            sigma = sigma * (1.0 + miss_info['M_pred'])
            
        # Remove batch dim if input was unbatched
        if X.shape[0] == 1:
            X_imp = X_imp.squeeze(0)
            sigma = sigma.squeeze(0)
            miss_info = {k: v.squeeze(0) if torch.is_tensor(v) else v 
                        for k, v in miss_info.items()}
            
        return X_imp, sigma, miss_info



class EncoderN(nn.Module):
    def __init__(self, d_in, d_hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden)
        )
    def forward(self, X): # [B,T,d] -> [B,T,h]
        return self.net(X)

class EncoderB(nn.Module):
    def __init__(self, d_in, d_hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden)
        )
    def forward(self, stats): # [B,d] -> [B,h]
        return self.net(stats)
