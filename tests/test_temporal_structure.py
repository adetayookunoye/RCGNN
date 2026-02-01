import pytest
import torch
from src.models.structure import StructureLearner

@pytest.fixture
def sample_data():
    """Generate sample temporal data."""
    B = 4
    d = 5
    h = 32
    ZS = torch.randn(B, d, h)
    return ZS

def test_temporal_shapes():
    """Test that temporal adjacency shapes are correct."""
    d = 5
    n_lags = 3
    learner = StructureLearner(d=d, n_lags=n_lags)
    
    # Test batched input
    B = 4
    ZS = torch.randn(B, d, d) # [B,d,h]
    A_all, logits_all = learner(ZS)
    
    assert len(A_all) == n_lags
    assert len(logits_all) == n_lags
    for A, logits in zip(A_all, logits_all):
        assert A.shape == (B, d, d)
        assert logits.shape == (B, d, d)
        
    # Test unbatched
    ZS_single = ZS[0] # [d,h]
    A_all, logits_all = learner(ZS_single)
    
    for A, logits in zip(A_all, logits_all):
        assert A.shape == (d, d)
        assert logits.shape == (d, d)

def test_temporal_prior():
    """Test temporal prior effects on adjacency."""
    d = 5
    n_lags = 3
    
    # Test exponential prior
    learner_exp = StructureLearner(d=d, n_lags=n_lags, temporal_prior='exp')
    
    # Test gamma prior
    learner_gamma = StructureLearner(d=d, n_lags=n_lags, temporal_prior='gamma')
    
    ZS = torch.randn(4, d, d)
    
    # Get adjacencies with priors
    A_exp, _ = learner_exp(ZS)
    A_gamma, _ = learner_gamma(ZS)
    
    # Check decreasing influence with lag
    for A_list in [A_exp, A_gamma]:
        means = [A.abs().mean().item() for A in A_list]
        assert all(means[i] >= means[i+1] for i in range(len(means)-1))

def test_acyclicity():
    """Test acyclicity constraints for temporal structure."""
    d = 5
    n_lags = 3
    learner = StructureLearner(d=d, n_lags=n_lags)
    
    # Create acyclic structure
    A_acyclic = []
    for _ in range(n_lags):
        A = torch.triu(torch.rand(d, d), diagonal=1)
        A_acyclic.append(A)
    
    pen_acyclic, _ = learner.acyclicity(A_acyclic)
    
    # Create cyclic structure
    A_cyclic = []
    for _ in range(n_lags):
        A = torch.rand(d, d) # Dense = likely cyclic
        A_cyclic.append(A)
    
    pen_cyclic, _ = learner.acyclicity(A_cyclic)
    
    # Cyclic should have higher penalty
    assert pen_cyclic > pen_acyclic

def test_env_deltas():
    """Test environment-specific structure learning."""
    d = 5
    n_lags = 3
    n_envs = 4
    learner = StructureLearner(d=d, n_lags=n_lags)
    
    # Initialize environment deltas
    learner.init_env_deltas(n_envs)
    
    assert learner.env_deltas.shape == (n_envs, d, d, n_lags)
    
    # Test with environment index
    ZS = torch.randn(1, d, d)
    ZS._env_idx = torch.tensor(0)
    
    A_all, _ = learner(ZS)
    
    # Should produce valid adjacencies
    assert len(A_all) == n_lags
    for A in A_all:
        assert not torch.isnan(A).any()

def test_sparsification():
    """Test sparsification methods with temporal structure."""
    d = 5
    n_lags = 3
    k = 2
    
    # Test different sparsification methods
    methods = ['topk', 'sparsemax', 'entmax', 'gumbel_topk']
    
    for method in methods:
        learner = StructureLearner(
            d=d, n_lags=n_lags,
            sparsify_method=method,
            sparsify_k=k if method in ['topk', 'gumbel_topk'] else None
        )
        
        ZS = torch.randn(4, d, d)
        A_all, _ = learner(ZS)
        
        # Check sparsity
        for A in A_all:
            if method in ['topk', 'gumbel_topk']:
                # Each column should have exactly k nonzeros
                assert all((A[0,:,j] > 0).sum() <= k for j in range(d))
            else:
                # Should produce some sparsity
                assert (A == 0).any()