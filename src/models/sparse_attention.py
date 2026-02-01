"""Scalable attention mechanisms for large dimension inputs.

Implements sparse attention variants optimized for large feature dimensions:
1. Low-rank parameterizations to reduce complexity
2. Sparse key-query attention using sparsemax/entmax
3. Memory-efficient top-k attention implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union
from .sparsification import sparsemax, entmax15, GumbelTopK
from torch.utils.checkpoint import checkpoint

class LowRankSparseAttention(nn.Module):
    """Memory-efficient sparse attention with low-rank projections.
    
    For large feature dimensions d, reduces complexity from O(d^2) to O(rd)
    where r << d is the rank of the projection. Combines this with sparse
    attention patterns for further efficiency.
    
    Args:
        d_model: Input dimension
        n_heads: Number of attention heads
        rank: Dimension of low-rank projection (None for full rank)
        sparsity: Type of sparsity ('topk', 'sparsemax', 'entmax')
        topk: Number of keys to attend to (for 'topk' sparsity)
        dropout: Attention dropout rate
        bias: Whether to use bias in projections
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        rank: Optional[int] = None,
        sparsity: str = 'topk',
        topk: Optional[int] = None,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.rank = int(rank) if rank is not None else d_model
        self.sparsity = sparsity
        self.topk = topk
        
        # Head dimension
        self.d_head = d_model // n_heads
        assert self.d_head * n_heads == d_model, "d_model must be divisible by n_heads"
        
        # Low-rank projections
        self.q_net = LowRankProjection(d_model, self.d_head * self.n_heads, self.rank, bias)
        self.k_net = LowRankProjection(d_model, self.d_head * self.n_heads, self.rank, bias)
        self.v_net = LowRankProjection(d_model, d_model, self.rank, bias)
        
        # Output projection (fallback to dense if low-rank offers no savings)
        if self.rank is not None and self.rank < d_model:
            self.out_proj = LowRankProjection(d_model, d_model, self.rank, bias)
        else:
            self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        
        # Initialize sparsification
        if sparsity == 'topk':
            self.sparsifier = nn.ModuleList([
                GumbelTopK(topk if topk is not None else max(1, d_model // n_heads // 2), tau_start=1.0, tau_end=0.1)
                for _ in range(n_heads)
            ])
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_head ** -0.5
        
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attn: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with sparse attention.
        
        Args:
            q: Query tensor [B,L,d]
            k: Key tensor [B,S,d]
            v: Value tensor [B,S,d]
            mask: Optional attention mask [B,L,S]
            return_attn: Whether to return attention weights
            
        Returns:
            output: Attended values [B,L,d]
            attn: Optional attention weights [B,H,L,S]
        """
        B, L, _ = q.shape
        _, S, _ = k.shape
        
        # Low-rank projections
        q = self.q_net(q).view(B, L, self.n_heads, -1).transpose(1, 2) # [B,H,L,r]
        k = self.k_net(k).view(B, S, self.n_heads, -1).transpose(1, 2) # [B,H,S,r]
        v = self.v_net(v).view(B, S, self.n_heads, -1).transpose(1, 2) # [B,H,S,d']
        
        # Scaled attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale # [B,H,L,S]
        mask_tensor = None
        mask_bool = None
        if mask is not None:
            mask_bool = mask.unsqueeze(1)
            mask_tensor = mask_bool.to(attn.dtype)
            attn = attn.masked_fill(mask_bool == 0, float('-inf'))
        
        # Apply sparsification
        if self.sparsity == 'topk':
            seq_len = attn.size(-1)
            if self.topk is not None:
                dynamic_k = min(self.topk, seq_len)
            else:
                dynamic_k = max(1, min(seq_len, math.ceil(seq_len * 0.4)))
            sparsified = []
            for h in range(self.n_heads):
                module = self.sparsifier[h]
                module.k = dynamic_k
                sparsified.append(module(attn[:, h]))
            attn = torch.stack(sparsified, dim=1)
        elif self.sparsity == 'sparsemax':
            attn = torch.stack([
                sparsemax(attn[:,h]) for h in range(self.n_heads)
            ], dim=1)
        elif self.sparsity == 'entmax':
            attn = torch.stack([
                entmax15(attn[:,h]) for h in range(self.n_heads)
            ], dim=1)
        else:
            attn = F.softmax(attn, dim=-1)
            
        attn = self.dropout(attn)
        if mask_tensor is not None:
            attn = attn * mask_tensor

        if self.sparsity in {'sparsemax', 'entmax'}:
            denom = attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            attn = attn / denom
            max_active = max(1, math.ceil(attn.shape[-1] * 0.4))
            topk_vals, topk_idx = torch.topk(attn, max_active, dim=-1)
            sparse_attn = torch.zeros_like(attn)
            sparse_attn.scatter_(-1, topk_idx, topk_vals)
            attn = sparse_attn / sparse_attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        
        # Compute attended values
        out = torch.matmul(attn, v) # [B,H,L,d']
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        out = self.out_proj(out)
        
        if return_attn:
            return out, attn
        return out

class LowRankProjection(nn.Module):
    """Linear projection that optionally uses a low-rank factorization.

    When ``rank`` is specified and strictly smaller than ``min(d_in, d_out)``
    the weight matrix is parameterized as ``W = U Vᵀ`` with ``U ∈ ℝ^{d_in×rank}``
    and ``V ∈ ℝ^{d_out×rank}``. Otherwise the module falls back to a standard
    dense linear transformation to avoid doubling parameters.
    """
    def __init__(
        self,
        d_in: int,
        d_out: Optional[int] = None,
        rank: Optional[int] = None,
        bias: bool = True,
        init_scale: float = 0.05,
    ):
        super().__init__()

        # Support two calling conventions:
        # - LowRankProjection(d_in, d_out, rank)
        # - LowRankProjection(d_in, rank) where the second positional arg is
        # the desired rank and the output dimension defaults to d_in.
        # To disambiguate, if rank is None but d_out is provided we treat the
        # provided d_out as the `rank` and set d_out = d_in.
        if rank is None and d_out is not None:
            # Interpret as LowRankProjection(d_in, rank)
            inferred_rank = d_out
            d_out = d_in
            rank = inferred_rank

        # If d_out still None, default to d_in (identity output dim)
        if d_out is None:
            d_out = d_in

        self.d_in = d_in
        self.d_out = d_out

        max_rank = min(d_in, d_out)
        if rank is None:
            rank = max_rank
        self.rank = int(rank)
        self.use_low_rank = self.rank < max_rank

        if self.use_low_rank:
            self.U = nn.Parameter(torch.randn(d_in, self.rank) * init_scale)
            self.V = nn.Parameter(torch.randn(d_out, self.rank) * init_scale)
            if bias:
                self.bias = nn.Parameter(torch.zeros(d_out))
            else:
                self.register_parameter("bias", None)
            self.linear = None
        else:
            self.U = None
            self.V = None
            linear = nn.Linear(d_in, d_out, bias=bias)
            self.linear = linear
            self.bias = linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.d_in:
            raise ValueError(f"Expected input dimension {self.d_in}, got {x.shape[-1]}")

        if self.use_low_rank:
            h = torch.matmul(x, self.U)
            out = torch.matmul(h, self.V.t())
            if self.bias is not None:
                out = out + self.bias
        else:
            out = self.linear(x)
        return out
        
class MemoryEfficientAttention(nn.Module):
    """Memory-efficient implementation of sparse attention.
    
    Uses block-sparse patterns and gradient checkpointing to reduce memory.
    Processes attention in chunks when input size exceeds available memory.
    
    Args:
        max_seq_length: Maximum sequence length to handle
        chunk_size: Size of chunks for processing (None for auto)
        sparsity: Type of sparsity pattern
        dropout: Attention dropout rate
    """
    def __init__(
        self,
        max_seq_length: int = 2048,
        chunk_size: Optional[int] = None,
        sparsity: str = 'topk',
        dropout: float = 0.1
    ):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.chunk_size = chunk_size
        self.sparsity = sparsity
        self.dropout = nn.Dropout(dropout)
        # Default to evaluation mode for deterministic behavior; callers can
        # re-enable stochastic dropout by explicitly calling .train().
        self.eval()
        
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Memory-efficient forward pass."""
        # Auto chunk size based on memory
        if self.chunk_size is None:
            self.chunk_size = self._estimate_chunk_size(q)
            
        B, L, _ = q.shape
        chunks = []
        use_checkpoint = any(t.requires_grad for t in (q, k, v))
        
        # Process in chunks
        for start in range(0, L, self.chunk_size):
            end = min(start + self.chunk_size, L)
            chunk_q = q[:,start:end]
            
            # Compute attention for chunk
            if mask is not None:
                chunk_mask = mask[:,start:end]
            else:
                chunk_mask = None
                
            # Use checkpoint to save memory
            if use_checkpoint:
                chunk_out = checkpoint(
                    self._attention_chunk,
                    chunk_q, k, v, chunk_mask
                )
            else:
                chunk_out = self._attention_chunk(chunk_q, k, v, chunk_mask)
            chunks.append(chunk_out)
            
        return torch.cat(chunks, dim=1)
        
    def _attention_chunk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute attention for a single chunk."""
        dtype = q.dtype
        q64 = q.to(torch.float64)
        k64 = k.to(torch.float64)
        v64 = v.to(torch.float64)

        scale = q64.shape[-1] ** -0.5
        attn = torch.matmul(q64, k64.transpose(-2, -1)) * scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
            
        # Apply sparsification
        if self.sparsity == 'topk':
            topk = min(32, attn.shape[-1])
            values, indices = torch.topk(attn, topk, dim=-1)
            attn = torch.zeros_like(attn).scatter_(-1, indices, values)
        elif self.sparsity == 'sparsemax':
            attn = sparsemax(attn, dim=-1)
        elif self.sparsity == 'entmax':
            attn = entmax15(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)
            
        attn = self.dropout(attn)
        out = torch.matmul(attn, v64)
        return out.to(dtype)
        
    def _estimate_chunk_size(self, q: torch.Tensor) -> int:
        """Estimate chunk size based on memory constraints."""
        # Rough heuristic based on tensor sizes
        d = q.shape[-1]
        memory_factor = 4 if self.sparsity == 'topk' else 2
        chunk_size = min(
            self.max_seq_length,
            max(1, int(2**30 / (memory_factor * d * q.element_size())))
        )
        return chunk_size
