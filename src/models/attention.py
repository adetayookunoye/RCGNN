"""Memory-efficient sparse attention mechanisms."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def topk_sparse(logits, k):
    """Sparsify by keeping only top-k entries per row.
    
    Args:
        logits: Attention logits [B,H,L,S]
        k: Number of entries to keep per row
    Returns:
        Sparse attention weights with same shape
    """
    # Get top k values per row
    top_values, _ = torch.topk(logits, k=k, dim=-1)
    
    # Get kth value per row for thresholding
    kth_values = top_values[..., -1:] # [B,H,L,1]
    
    # Zero out entries below threshold
    sparse_weights = torch.where(
        logits >= kth_values,
        logits,
        torch.full_like(logits, float('-inf'))
    )
    
    return F.softmax(sparse_weights, dim=-1)


def entmax_sparse(logits, alpha=1.5):
    """Sparse attention using α-entmax transformation.
    
    Implementation of 1.5-entmax (exact computation).
    See: https://arxiv.org/abs/1905.05702
    
    Args:
        logits: Attention logits [B,H,L,S] 
        alpha: Sharpness parameter (1.5 for sparsemax)
    Returns:
        Sparse attention weights with same shape
    """
    # Sort logits in descending order
    logits_sorted, _ = torch.sort(logits, dim=-1, descending=True)
    
    # Compute threshold tau via binary search
    n = logits.size(-1)
    cumsum = torch.cumsum(logits_sorted, dim=-1) - 1
    position = torch.arange(n, device=logits.device).view(
        *([1]*(logits.ndim-1)), n)
    
    # Compute τ that solves the equation
    tau = (cumsum / (position + 1)) - ((2/(alpha + 1)) * logits_sorted)
    
    # Find max index where tau solution is valid
    valid = (logits_sorted - tau) > 0
    rho = valid.sum(dim=-1, keepdim=True)
    
    # Compute final τ solution for selected positions
    tau = cumsum.gather(-1, (rho - 1)) / rho
    
    # Apply softmax-like transformation
    p_tilde = torch.clamp(logits - tau.unsqueeze(-1), min=0) ** (1 / (alpha - 1))
    
    return p_tilde


class LowRankSparseAttention(nn.Module):
    """Memory-efficient multi-head attention using low-rank + sparse attention.
    
    Uses low-rank projections and sparse attention patterns to reduce memory
    requirements for large sequence lengths.
    """
    
    def __init__(self, d_model, n_heads=8, rank=32, dropout=0.1,
                 sparsity='topk', topk=32):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.rank = rank
        self.dropout = dropout
        self.sparsity = sparsity
        self.topk = topk
        
        # Low-rank projections for Q,K,V
        self.W_q = nn.Linear(d_model, rank * n_heads)
        self.W_k = nn.Linear(d_model, rank * n_heads) 
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.W_out = nn.Linear(d_model, d_model)
        
        # Optional learnable position bias
        self.pos_bias = nn.Parameter(torch.zeros(1, n_heads, 1, 1))
        
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None, return_attn=False):
        """Memory-efficient attention forward pass.
        
        Args:
            q: Query tensor [B,L,D]
            k: Key tensor [B,S,D] 
            v: Value tensor [B,S,D]
            mask: Optional attention mask [B,L,S]
            return_attn: Return attention weights
            
        Returns:
            output: Attended values [B,L,D]
            attn_weights: Optional attention weights [B,H,L,S]
        """
        batch_size = q.size(0)
        
        # Low-rank projections
        Q = self.W_q(q).view(batch_size, -1, self.n_heads, self.rank)
        K = self.W_k(k).view(batch_size, -1, self.n_heads, self.rank)
        V = self.W_v(v).view(batch_size, -1, self.n_heads, self.d_head)
        
        # Transpose for attention calculation
        Q = Q.transpose(1, 2) # [B,H,L,R]
        K = K.transpose(1, 2) # [B,H,S,R] 
        V = V.transpose(1, 2) # [B,H,S,D]
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) # [B,H,L,S]
        scores = scores / (self.rank ** 0.5)
        
        # Add learned position bias
        scores = scores + self.pos_bias
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(
                ~mask.unsqueeze(1), # Add heads dim
                float('-inf')
            )
            
        # Apply sparse attention
        if self.sparsity == 'topk':
            attn_weights = topk_sparse(scores, k=self.topk)
        elif self.sparsity == 'entmax':
            attn_weights = entmax_sparse(scores)
        else:
            attn_weights = F.softmax(scores, dim=-1)
            
        # Apply attention dropout
        attn_weights = self.attn_dropout(attn_weights)
        
        # Compute attended values
        out = torch.matmul(attn_weights, V) # [B,H,L,D]
        
        # Transpose and reshape
        out = out.transpose(1, 2).contiguous() # [B,L,H,D]
        out = out.view(batch_size, -1, self.d_model)
        
        # Final output projection
        out = self.W_out(out)
        out = self.out_dropout(out)
        
        if return_attn:
            return out, attn_weights
            
        return out


# Register functions for import
__all__ = [
    'LowRankSparseAttention',
    'topk_sparse',
    'entmax_sparse'
]