import pytest
import torch
import numpy as np
from src.models.masking_attention import MaskingAwareAttention, TransformerImputer

@pytest.fixture
def sample_data():
    """Generate synthetic time series with missing values."""
    B, T, d = 4, 50, 10
    X = torch.randn(B, T, d)
    # Create missing patterns: random, chunks, and seasonal
    M = torch.ones(B, T, d, dtype=torch.bool)
    
    # Random missingness
    random_mask = torch.rand(B, T, d) > 0.7
    M[0] = random_mask[0]
    
    # Chunked missingness (consecutive missing values)
    for j in range(d):
        start = np.random.randint(0, T-10)
        M[1, start:start+10, j] = False
        
    # Seasonal missingness (periodic pattern)
    M[2, ::5] = False
    
    # Mixed patterns
    M[3] = random_mask[3]
    M[3, ::7] = False
    
    # Zero out missing values
    X = torch.where(M, X, torch.zeros_like(X))
    
    return X, M

def test_masking_aware_attention():
    """Test attention module's handling of missing values."""
    d_model, n_heads = 64, 4
    attn = MaskingAwareAttention(d_model, n_heads)
    
    # Create input with missing values
    B, T = 2, 10
    x = torch.randn(B, T, d_model)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[:, [3,7]] = False # Make some positions missing
    
    # Test forward pass
    out = attn(x, mask)
    
    assert out.shape == (B, T, d_model)
    # Missing positions should have different patterns than observed
    missing_pos = ~mask
    obs_std = out[mask].std()
    miss_std = out[missing_pos.unsqueeze(-1).expand_as(out)].std()
    assert abs(obs_std - miss_std) > 1e-3

def test_relative_position_bias():
    """Test relative positional encoding."""
    d_model, n_heads = 64, 4
    attn = MaskingAwareAttention(d_model, n_heads)
    
    T = 20
    bias = attn.get_rel_pos_bias(T)
    
    assert bias.shape == (T, T, n_heads)
    # Check symmetry
    assert torch.allclose(bias[0,1], -bias[1,0])
    # Check scale
    assert bias.abs().mean() > 0

def test_imputer_reconstruction(sample_data):
    """Test imputer's ability to reconstruct missing values."""
    X, M = sample_data
    B, T, d = X.shape
    
    imputer = TransformerImputer(d_in=d)
    X_imp, sigma = imputer(X, M)
    
    # Check shapes
    assert X_imp.shape == (B, T, d)
    assert sigma.shape == (B, T, d)
    
    # Observed values should be preserved
    assert torch.allclose(X_imp[M], X[M], rtol=1e-5)
    
    # Missing values should be imputed (not zero)
    assert not torch.allclose(X_imp[~M], torch.zeros_like(X_imp[~M]))
    
    # Uncertainty should be higher for missing values
    obs_uncert = sigma[M].mean()
    miss_uncert = sigma[~M].mean()
    assert miss_uncert > obs_uncert

def test_imputer_temporal_patterns(sample_data):
    """Test imputer's handling of temporal patterns."""
    X, M = sample_data
    B, T, d = X.shape
    
    # Create synthetic temporal pattern
    pattern = 0.5 * torch.sin(torch.linspace(0, 4*np.pi, T)).unsqueeze(-1)
    X = X + pattern.unsqueeze(0)
    
    imputer = TransformerImputer(d_in=d)
    X_imp, _ = imputer(X, M)
    
    # Compute correlation between true and imputed patterns
    pattern_corr = torch.zeros(B, d)
    for b in range(B):
        for j in range(d):
            missing = ~M[b, :, j]
            if missing.any():
                true_pattern = pattern[missing].squeeze()
                imp_pattern = X_imp[b, missing, j]
                pattern_corr[b,j] = torch.corrcoef(
                    torch.stack([true_pattern, imp_pattern])
                )[0,1]
    
    # At least some recovered patterns should be correlated
    assert (pattern_corr > 0.5).any()

def test_imputer_uncertainty_calibration(sample_data):
    """Test if uncertainty estimates are well-calibrated."""
    X, M = sample_data
    B, T, d = X.shape
    
    imputer = TransformerImputer(d_in=d)
    X_imp, sigma = imputer(X, M)
    
    # Get missing value errors
    errors = (X_imp - X)[~M]
    uncertainties = sigma[~M]
    
    # Sort by uncertainty
    sorted_idx = torch.argsort(uncertainties)
    errors = errors[sorted_idx]
    uncertainties = uncertainties[sorted_idx]
    
    # Check calibration in bins
    n_bins = 10
    bin_size = len(errors) // n_bins
    calibration_error = 0
    
    for i in range(n_bins):
        bin_start = i * bin_size
        bin_end = (i + 1) * bin_size if i < n_bins-1 else len(errors)
        
        bin_errors = errors[bin_start:bin_end]
        bin_uncert = uncertainties[bin_start:bin_end].mean()
        
        empirical_std = bin_errors.std()
        calibration_error += torch.abs(empirical_std - bin_uncert)
    
    avg_calibration_error = calibration_error / n_bins
    assert avg_calibration_error < 0.5 # Should be reasonably calibrated

def test_imputer_batch_handling():
    """Test imputer's handling of batched and unbatched inputs."""
    d = 5
    imputer = TransformerImputer(d_in=d)
    
    # Test unbatched
    X = torch.randn(10, d)
    M = torch.ones(10, d, dtype=torch.bool)
    M[3:7] = False
    
    X_imp1, sigma1 = imputer(X, M)
    assert X_imp1.shape == (10, d)
    assert sigma1.shape == (10, d)
    
    # Test same input batched
    X_b = X.unsqueeze(0)
    M_b = M.unsqueeze(0)
    
    X_imp2, sigma2 = imputer(X_b, M_b)
    assert torch.allclose(X_imp1, X_imp2.squeeze(0))
    assert torch.allclose(sigma1, sigma2.squeeze(0))