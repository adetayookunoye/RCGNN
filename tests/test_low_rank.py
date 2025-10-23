import pytest
import torch
from src.models.low_rank import (
    LowRankGNN,
    LowRankEncoder,
    LowRankStructureLearner
)

@pytest.fixture
def sample_data():
    """Generate sample high-dimensional data."""
    B, N, d = 4, 16, 256  # Large feature dimension
    x = torch.randn(B, N, d)
    mask = torch.ones(B, N, N, dtype=torch.bool)
    return x, mask

def test_low_rank_gnn(sample_data):
    """Test memory-efficient GNN layer."""
    d = 256
    rank = 32
    gnn = LowRankGNN(d, rank=rank)
    
    x, mask = sample_data
    out = gnn(x, mask)
    
    assert out.shape == x.shape
    
    # Test with attention weights
    out, attn = gnn(x, mask, return_attn=True)
    assert attn.shape == (4, 8, 16, 16)  # [B,H,N,N]
    
    # Check parameter efficiency
    full_params = d * d * 3  # Rough estimate for full rank
    low_params = sum(p.numel() for p in gnn.parameters())
    assert low_params < full_params

def test_low_rank_encoder(sample_data):
    """Test low-rank tri-latent encoder."""
    d = 256
    d_latent = 128
    rank = 32
    encoder = LowRankEncoder(d, d_latent, rank)
    
    x, mask = sample_data
    signal, noise, bias = encoder(x, mask)
    
    # Check output shapes
    assert signal.shape == x.shape
    assert noise.shape == x.shape
    assert bias.shape == x.shape
    
    # Verify parameter efficiency
    params = sum(p.numel() for p in encoder.parameters())
    full_params = d * d * 9  # Rough estimate for full rank
    assert params < full_params

def test_low_rank_structure(sample_data):
    """Test memory-efficient structure learning."""
    d = 256
    rank = 32
    learner = LowRankStructureLearner(d, rank)
    
    x, mask = sample_data
    adj, attn = learner(x, mask)
    
    # Check output shapes
    assert adj.shape == (4, 16, 16)  # [B,N,N]
    assert attn.shape == (4, 8, 16, 16)  # [B,H,N,N]
    
    # Verify adjacency properties
    assert torch.all(adj >= 0) and torch.all(adj <= 1)
    assert torch.allclose(
        torch.diagonal(adj, dim1=-2, dim2=-1),
        torch.zeros_like(torch.diagonal(adj, dim1=-2, dim2=-1))
    )

def test_memory_scaling():
    """Test memory scaling with dimension."""
    Bs = [4, 4]  # Batch sizes
    Ns = [16, 16]  # Number of nodes
    ds = [128, 256]  # Feature dimensions
    ranks = [32, 32]  # Rank of low-rank approximation
    
    def get_mem_usage(model, x):
        if not torch.cuda.is_available():
            return 0
        torch.cuda.reset_peak_memory_stats()
        _ = model(x)
        return torch.cuda.max_memory_allocated()
    
    mems = []
    for B, N, d, r in zip(Bs, Ns, ds, ranks):
        x = torch.randn(B, N, d)
        mask = torch.ones(B, N, N)
        
        model = LowRankGNN(d, rank=r)
        
        if torch.cuda.is_available():
            x = x.cuda()
            mask = mask.cuda()
            model = model.cuda()
        
        mem = get_mem_usage(model, x)
        mems.append(mem)
        
    if len(mems) > 1 and all(m > 0 for m in mems):
        # Memory should scale sub-quadratically with dimension
        ratio = mems[1] / mems[0]
        dim_ratio = (ds[1] / ds[0]) ** 2
        assert ratio < dim_ratio

def test_gradient_flow():
    """Test gradient flow through low-rank modules."""
    d = 256
    rank = 32
    
    # Test encoder gradients
    encoder = LowRankEncoder(d, d//2, rank)
    x = torch.randn(4, 16, d)
    x.requires_grad_(True)
    
    signal, noise, bias = encoder(x)
    loss = signal.sum() + noise.sum() + bias.sum()
    loss.backward()
    
    assert x.grad is not None
    assert not torch.allclose(x.grad, torch.zeros_like(x.grad))
    
    # Test structure learner gradients
    learner = LowRankStructureLearner(d, rank)
    x = torch.randn(4, 16, d, requires_grad=True)
    
    adj, _ = learner(x)
    loss = adj.sum()
    loss.backward()
    
    assert x.grad is not None
    assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

def test_sparsity_patterns(sample_data):
    """Test sparsification in low-rank modules."""
    d = 256
    rank = 32
    
    # Test different sparsity methods
    for sparsity in ['topk', 'sparsemax', 'entmax']:
        gnn = LowRankGNN(d, rank=rank, sparsity=sparsity)
        x, mask = sample_data
        
        _, attn = gnn(x, mask, return_attn=True)
        
        # Check that attention is sparse
        sparsity_ratio = (attn == 0).float().mean()
        assert sparsity_ratio > 0.5  # At least 50% sparse