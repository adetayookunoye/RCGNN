"""Low-rank parameterizations for scalable graph neural networks.

Implements memory-efficient model components using low-rank factorizations
and sparse operations for handling large feature dimensions.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from .sparse_attention import LowRankProjection, LowRankSparseAttention

class LowRankGNN(nn.Module):
    """Memory-efficient graph neural layer using low-rank parameterizations.
    
    For large feature dimensions d, reduces parameter count and memory usage
    through low-rank weight matrices and sparse operations.
    
    Args:
        d_in: Input dimension
        d_out: Output dimension (None = same as input)
        rank: Dimension of low-rank projection (None for full rank)
        n_heads: Number of attention heads for structure learning
        dropout: Dropout rate
        sparsity: Type of sparsity ('topk', 'sparsemax', 'entmax')
        batch_norm: Whether to use batch normalization
    """
    def __init__(
        self,
        d_in: int,
        d_out: Optional[int] = None,
        rank: Optional[int] = None,
        n_heads: int = 8,
        dropout: float = 0.1,
        sparsity: str = 'topk',
        batch_norm: bool = True
    ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out if d_out is not None else d_in
        
        # Low-rank node transformation
        self.node_proj = LowRankProjection(d_in, self.d_out, rank)
        
        # Structure learning attention
        self.attention = LowRankSparseAttention(
            d_model=d_in,
            n_heads=n_heads,
            rank=rank,
            sparsity=sparsity
        )
        
        # Output projection
        self.out_proj = LowRankProjection(self.d_out, self.d_out, rank)
        
        # Batch norm and dropout
        if batch_norm:
            self.norm = nn.BatchNorm1d(self.d_out)
        else:
            self.norm = nn.Identity()
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attn: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with optional attention weights.
        
        Args:
            x: Input features [B,N,d]
            mask: Optional attention mask [B,N,N]
            return_attn: Whether to return attention weights
            
        Returns:
            output: Updated features [B,N,d]
            attn: Optional attention weights [B,H,N,N]
        """
        # Node projection
        h = self.node_proj(x)
        
        # Structure learning
        if return_attn:
            h, attn = self.attention(h, h, h, mask, return_attn=True)
        else:
            h = self.attention(h, h, h, mask)
            
        # Output transformation
        out = self.out_proj(h)
        
        # Normalization and dropout
        if out.dim() == 3:
            out = out.transpose(1, 2)  # [B,d,N]
            out = self.norm(out)
            out = out.transpose(1, 2)  # [B,N,d]
        else:
            out = self.norm(out)
        out = self.dropout(out)
        
        if return_attn:
            return out, attn
        return out

class LowRankEncoder(nn.Module):
    """Low-rank encoder for tri-latent model (signal/noise/bias).
    
    Uses low-rank parameterizations for memory efficiency when encoding
    high-dimensional inputs.
    
    Args:
        d_in: Input dimension
        d_latent: Latent dimension
        rank: Dimension of low-rank projections
        n_layers: Number of GNN layers
        dropout: Dropout rate
    """
    def __init__(
        self,
        d_in: int,
        d_latent: int,
        rank: Optional[int] = None,
        n_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_in = d_in
        self.d_latent = d_latent
        
        # Input projection
        self.input_proj = LowRankProjection(d_in, d_latent, rank)
        
        # GNN layers
        self.layers = nn.ModuleList([
            LowRankGNN(
                d_in=d_latent,
                rank=rank,
                dropout=dropout
            ) for _ in range(n_layers)
        ])
        
        # Output heads for signal/noise/bias
        self.signal_head = LowRankProjection(d_latent, d_in, rank)
        self.noise_head = LowRankProjection(d_latent, d_in, rank)
        self.bias_head = LowRankProjection(d_latent, d_in, rank)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode input into signal/noise/bias representations.
        
        Args:
            x: Input features [B,N,d]
            mask: Optional attention mask [B,N,N]
            
        Returns:
            signal: Signal encoding [B,N,d]
            noise: Noise encoding [B,N,d]
            bias: Bias encoding [B,N,d]
        """
        # Initial projection
        h = self.input_proj(x)
        
        # Apply GNN layers
        attns = []
        for layer in self.layers:
            h, attn = layer(h, mask, return_attn=True)
            attns.append(attn)
            
        # Generate tri-latent encodings
        signal = self.signal_head(h)
        noise = self.noise_head(h)
        bias = self.bias_head(h)
        
        return signal, noise, bias

class LowRankStructureLearner(nn.Module):
    """Memory-efficient structure learner for large graphs.
    
    Uses low-rank parameterizations and sparse attention for learning
    graph structure with high-dimensional node features.
    
    Args:
        d: Feature dimension
        rank: Dimension of low-rank projection
        n_heads: Number of attention heads
        sparsity: Type of sparsity ('topk', 'sparsemax', 'entmax')
        dropout: Dropout rate
    """
    def __init__(
        self,
        d: int,
        rank: Optional[int] = None,
        n_heads: int = 8,
        sparsity: str = 'topk',
        dropout: float = 0.1
    ):
        super().__init__()
        self.d = d
        self.rank = rank
        
        # Node embeddings
        self.node_embedding = LowRankProjection(d, d, rank)
        
        # Structure attention
        self.attention = LowRankSparseAttention(
            d_model=d,
            n_heads=n_heads,
            rank=rank,
            sparsity=sparsity,
            dropout=dropout
        )
        
        # Edge prediction
        self.edge_score = LowRankProjection(d, d, rank)
        
    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Learn graph structure from node features.
        
        Args:
            x: Node features [B,N,d]
            mask: Optional mask [B,N,N]
            
        Returns:
            adjacency: Learned adjacency matrix [B,N,N]
            attention: Attention weights [B,H,N,N]
        """
        # Node embeddings
        h = self.node_embedding(x)
        
        # Compute attention-based structure
        h, attn = self.attention(h, h, h, mask, return_attn=True)
        
        # Edge scores
        edge_logits = self.edge_score(h)
        adjacency = torch.sigmoid(torch.matmul(edge_logits, edge_logits.transpose(-2, -1)))
        
        # Remove self-loops
        adjacency = adjacency * (1 - torch.eye(adjacency.size(-1), device=adjacency.device))
        
        return adjacency, attn