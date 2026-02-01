import pytest
import torch
from src.models.invariance import IRMStructureInvariance

@pytest.fixture
def sample_data():
    """Generate synthetic data with environments."""
    B, T, d = 8, 20, 5
    X = torch.randn(B, T, d)
    M = torch.ones(B, T, d, dtype=torch.bool)
    
    # Create different environments
    e = torch.cat([torch.full((2,), i) for i in range(4)])
    
    # Create environment-specific patterns
    for i in range(4):
        env_mask = (e == i)
        # Add environment-specific temporal patterns
        t = torch.arange(T).float() / T
        env_pattern = 0.1 * (i + 1) * torch.sin(2 * torch.pi * t * (i + 1))
        X[env_mask] = X[env_mask] + env_pattern.unsqueeze(0).unsqueeze(-1)
        
    return X, M, e

def test_env_risk_computation(sample_data):
    """Test that environment-specific risks are computed correctly."""
    d = 5
    n_envs = 4
    invariance = IRMStructureInvariance(d, n_envs)
    
    # Create synthetic data
    X, M, e = sample_data
    A = torch.rand(8, d, d) # Random adjacency
    logits = torch.randn(8, d, d) # Random logits
    
    # Compute risks
    risks, grad_penalty = invariance.compute_env_risk(A, logits, X, M, e)
    
    assert len(risks) == n_envs
    assert grad_penalty.ndim == 0 # Scalar
    assert torch.all(risks >= 0)
    assert grad_penalty >= 0

def test_structure_variance(sample_data):
    """Test structure variance computation across environments."""
    d = 5
    n_envs = 4
    invariance = IRMStructureInvariance(d, n_envs)
    
    # Create different adjacency matrices per environment
    A = torch.zeros(8, d, d)
    _, _, e = sample_data
    
    # Make each environment slightly different
    for i in range(4):
        env_mask = (e == i)
        A[env_mask] = torch.eye(d) + 0.1 * i * torch.randn(d, d)
        
    var_penalty = invariance.structure_variance(A, None, e)
    
    assert var_penalty.ndim == 0 # Scalar output
    assert var_penalty > 0 # Should detect structure differences
    
    # Test with identical structures - should have zero variance
    A_same = torch.eye(d).unsqueeze(0).expand(8, -1, -1)
    var_penalty_same = invariance.structure_variance(A_same, None, e)
    assert torch.allclose(var_penalty_same, torch.tensor(0.0), atol=1e-6)

def test_gradients_flow(sample_data):
    """Test that gradients flow properly through the invariance module."""
    d = 5
    n_envs = 4
    invariance = IRMStructureInvariance(d, n_envs)
    
    X, M, e = sample_data
    
    # Create trainable adjacency
    A = torch.rand(8, d, d, requires_grad=True)
    logits = torch.randn(8, d, d, requires_grad=True)
    
    # Forward pass with temporal prediction task
    loss, metrics = invariance.forward(A, logits, X, M, e)
    
    # Check gradient flow
    loss.backward()
    
    assert A.grad is not None
    assert logits.grad is not None
    assert not torch.allclose(A.grad, torch.zeros_like(A.grad))

def test_env_consistency(sample_data):
    """Test consistency of invariance penalties across environments."""
    d = 5
    n_envs = 4
    invariance = IRMStructureInvariance(d, n_envs, gamma=1.0) # Higher gamma for testing
    
    X, M, e = sample_data
    
    # Create identical structure across environments
    A = torch.eye(d).unsqueeze(0).expand(8, -1, -1)
    logits = torch.randn(8, d, d)
    
    # Compute invariance with identical structure
    loss1, metrics1 = invariance.forward(A, logits, X, M, e)
    
    # Now make structures different across environments
    A_diff = A.clone()
    for i in range(4):
        env_mask = (e == i)
        A_diff[env_mask] = A[env_mask] + 0.1 * i * torch.randn_like(A[env_mask])
    
    # Compute invariance with different structures
    loss2, metrics2 = invariance.forward(A_diff, logits, X, M, e)
    
    # Penalty should be higher for different structures
    assert loss2 > loss1
    
    # Check metrics structure
    assert all(k in metrics1 for k in ['env_risks', 'grad_penalty', 'var_penalty', 'total_invariance'])

def test_batch_handling():
    """Test handling of batched and unbatched inputs."""
    d = 5
    n_envs = 4
    invariance = IRMStructureInvariance(d, n_envs)
    
    # Test with minimal batch
    X_min = torch.randn(2, 10, d) # Just 2 samples
    M_min = torch.ones(2, 10, d, dtype=torch.bool)
    e_min = torch.tensor([0, 1]) # 2 environments
    A_min = torch.rand(2, d, d)
    logits_min = torch.randn(2, d, d)
    
    # Should handle minimal batch without errors
    loss_min, _ = invariance.forward(A_min, logits_min, X_min, M_min, e_min)
    
    # Test with scalar environment index
    loss_scalar, _ = invariance.forward(A_min[0], logits_min[0], X_min[0], M_min[0], 0)
    
    # Both should produce valid losses
    assert not torch.isnan(loss_min)
    assert not torch.isnan(loss_scalar)