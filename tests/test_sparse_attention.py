import pytest
import torch
from src.models.sparse_attention import (
    LowRankSparseAttention,
    LowRankProjection,
    MemoryEfficientAttention
)

@pytest.fixture
def sample_input():
    """Generate sample inputs for attention."""
    B, L, S, d = 4, 16, 16, 64
    q = torch.randn(B, L, d)
    k = torch.randn(B, S, d)
    v = torch.randn(B, S, d)
    mask = torch.ones(B, L, S, dtype=torch.bool)
    return q, k, v, mask

def test_low_rank_projection():
    """Test low-rank projection module."""
    d = 64
    rank = 16
    proj = LowRankProjection(d, rank)
    
    x = torch.randn(4, 16, d)
    out = proj(x)
    
    assert out.shape == x.shape
    assert proj.U.shape == (d, rank)
    assert proj.V.shape == (d, rank)
    
    # Test full rank
    proj_full = LowRankProjection(d, None)
    out_full = proj_full(x)
    assert out_full.shape == x.shape

def test_low_rank_sparse_attention(sample_input):
    """Test low-rank sparse attention."""
    d_model = 64
    n_heads = 8
    rank = 16
    
    for sparsity in ['topk', 'sparsemax', 'entmax']:
        attn = LowRankSparseAttention(
            d_model, n_heads, rank,
            sparsity=sparsity,
            topk=4 if sparsity == 'topk' else None
        )
        
        q, k, v, mask = (tensor.clone() for tensor in sample_input)
        out = attn(q, k, v, mask)
        
        assert out.shape == q.shape
        
        # Test with attention weights
        out, weights = attn(q, k, v, mask, return_attn=True)
        B, L, _ = q.shape
        _, S, _ = k.shape
        assert weights.shape == (B, n_heads, L, S)
        
        if sparsity == 'topk':
            # Check sparsity level
            assert (weights[0,0,0] > 0).sum() <= 4

def test_memory_efficient_attention(sample_input):
    """Test memory-efficient attention implementation."""
    attn = MemoryEfficientAttention(
        max_seq_length=32,
        chunk_size=8,
        sparsity='topk'
    )
    
    q, k, v, mask = (tensor.clone() for tensor in sample_input)
    out = attn(q, k, v, mask)
    
    assert out.shape == q.shape
    
    # Test different chunk sizes
    attn.chunk_size = 4
    out2 = attn(q, k, v, mask)
    assert torch.allclose(out, out2, rtol=1e-5)
    
    # Test auto chunk sizing
    attn.chunk_size = None
    out3 = attn(q, k, v, mask)
    assert torch.allclose(out, out3, rtol=1e-5)

def test_attention_sparsity(sample_input):
    """Test that attention patterns are actually sparse."""
    d_model = 64
    n_heads = 4
    
    # Test each sparsity type
    for sparsity in ['topk', 'sparsemax', 'entmax']:
        attn = LowRankSparseAttention(
            d_model, n_heads,
            sparsity=sparsity,
            topk=4 if sparsity == 'topk' else None
        )
        
        q, k, v, _ = (tensor.clone() for tensor in sample_input)
        
        # Get attention weights
        _, weights = attn(q, k, v, return_attn=True)
        
        # Check sparsity
        sparsity_ratio = (weights == 0).float().mean()
        assert sparsity_ratio > 0.5  # At least 50% sparse

def test_attention_mask(sample_input):
    """Test that attention mask is properly applied."""
    d_model = 64
    n_heads = 4
    attn = LowRankSparseAttention(d_model, n_heads, sparsity='topk', topk=4)
    
    q, k, v, _ = (tensor.clone() for tensor in sample_input)
    B, L, S = q.shape[0], q.shape[1], k.shape[1]
    
    # Create mask blocking half the positions
    mask = torch.ones(B, L, S, dtype=torch.bool)
    mask[:, :, S//2:] = 0
    
    _, weights = attn(q, k, v, mask, return_attn=True)
    
    # Check that masked positions have zero attention
    assert torch.all(weights[:, :, :, S//2:] == 0)

def test_attention_gradients(sample_input):
    """Test that gradients flow properly through attention."""
    d_model = 64
    n_heads = 4
    attn = LowRankSparseAttention(d_model, n_heads, rank=16)
    
    q, k, v, _ = (tensor.clone() for tensor in sample_input)
    q.requires_grad_(True)
    
    out = attn(q, k, v)
    loss = out.sum()
    loss.backward()
    
    assert q.grad is not None
    assert not torch.allclose(q.grad, torch.zeros_like(q.grad))
    
def test_low_rank_memory():
    """Test that low-rank attention uses less memory."""
    d_model = 256
    n_heads = 8
    B, L = 4, 32
    
    # Full rank
    full = LowRankSparseAttention(d_model, n_heads, rank=None)
    
    # Low rank (r=32)
    low = LowRankSparseAttention(d_model, n_heads, rank=32)
    
    # Compare parameter counts
    full_params = sum(p.numel() for p in full.parameters())
    low_params = sum(p.numel() for p in low.parameters())
    
    assert low_params < full_params
    
    # Check runtime memory (rough estimate)
    x = torch.randn(B, L, d_model)
    
    def get_mem(model, x):
        torch.cuda.reset_peak_memory_stats()
        _ = model(x, x, x)
        return torch.cuda.max_memory_allocated()
    
    if torch.cuda.is_available():
        x = x.cuda()
        full = full.cuda()
        low = low.cuda()
        
        full_mem = get_mem(full, x)
        low_mem = get_mem(low, x)
        
        assert low_mem < full_mem
