import pytest
import torch
import numpy as np
from src.models.recon import Recon
from src.models.missingness import MissingnessModel

@pytest.fixture
def sample_data():
    """Generate synthetic data with missingness patterns."""
    B, T, d = 4, 20, 5
    X = torch.randn(B, T, d)
    
    # Create missingness with temporal patterns
    t = torch.linspace(0, 1, T)
    M = torch.zeros(B, T, d, dtype=torch.bool)
    for i in range(d):
        # Different frequency per feature
        m = torch.sin(2 * np.pi * (i+1) * t) > 0
        M[..., i] = m
        
    # Add value-dependent missingness (MNAR)
    M = M | (X > 1.0)
    
    return X, M

def test_recon_shapes():
    """Test output shapes from reconstruction."""
    d = 5
    h = 32
    recon = Recon(d, h)
    
    # Test batched
    B, T = 4, 20
    S_hat = torch.randn(B, T, d)
    ZB = torch.randn(B, h)
    M_probs = torch.rand(B, T, d)
    
    X_mu, unc, components = recon(S_hat, ZB, M_probs)
    
    assert X_mu.shape == (B, T, d)
    assert unc.shape == (B, T, d)
    assert all(v.shape == (B, T, d) for k, v in components.items() 
              if v is not None)
    
    # Test unbatched
    X_mu, unc, components = recon(S_hat[0], ZB[0], M_probs[0])
    
    assert X_mu.shape == (T, d)
    assert unc.shape == (T, d)
    assert all(v.shape == (T, d) for k, v in components.items() 
              if v is not None)

def test_uncertainty_components():
    """Test that uncertainty components behave correctly."""
    d = 5
    h = 32
    recon = Recon(d, h)
    
    B, T = 4, 20
    S_hat = torch.randn(B, T, d)
    ZB = torch.randn(B, h)
    M_probs = torch.rand(B, T, d)
    
    # Base case - no missingness info
    _, unc1, comp1 = recon(S_hat, ZB)
    
    # With missingness probabilities
    _, unc2, comp2 = recon(S_hat, ZB, M_probs)
    
    # Higher missingness should increase uncertainty
    high_miss = torch.ones_like(M_probs) * 0.9
    _, unc3, comp3 = recon(S_hat, ZB, high_miss)
    
    assert torch.all(unc2 >= comp1['base_unc']) # Total unc >= base
    assert torch.all(unc3 >= unc2) # Higher missingness -> higher unc

def test_nll_loss(sample_data):
    """Test negative log likelihood loss computation."""
    d = 5
    h = 32
    recon = Recon(d, h)
    
    X, M = sample_data
    B, T, d = X.shape
    
    # Generate predictions
    S_hat = torch.randn(B, T, d)
    ZB = torch.randn(B, h)
    X_mu, unc, _ = recon(S_hat, ZB)
    
    # Compute loss with different reductions
    loss1 = recon.nll_loss(X, X_mu, unc, reduction='none')
    loss2 = recon.nll_loss(X, X_mu, unc, M=M, reduction='mean')
    loss3 = recon.nll_loss(X, X_mu, unc, reduction='sum')
    
    assert loss1.shape == (B, T, d)
    assert loss2.ndim == 0 # Scalar
    assert loss3.ndim == 0
    assert torch.all(loss1 >= 0) # NLL should be positive

def test_mnar_integration(sample_data):
    """Test integration with MNAR missingness model."""
    d = 5
    h = 32
    recon = Recon(d, h)
    miss_model = MissingnessModel(d)
    
    X, M = sample_data
    B, T, d = X.shape
    
    # Get missingness probabilities
    M_probs, _ = miss_model(X, M)
    
    # Generate reconstructions with MNAR awareness
    S_hat = torch.randn(B, T, d)
    ZB = torch.randn(B, h)
    X_mu, unc, components = recon(S_hat, ZB, M_probs)
    
    # Compute masked NLL loss
    loss = recon.nll_loss(X, X_mu, unc, M)
    
    assert not torch.isnan(loss)
    assert components['mnar_scale'] is not None
    assert torch.all(components['mnar_scale'] > 0)
