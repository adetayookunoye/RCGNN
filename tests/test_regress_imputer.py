import torch
from src.models.masking_attention import TransformerImputer


def test_imputer_sigma_floor_and_no_nan():
    torch.manual_seed(0)
    B, T, d = 2, 6, 5
    X = torch.randn(B, T, d)
    # Create missing mask with 50% missing
    M = (torch.rand(B, T, d) > 0.5).int()

    imputer = TransformerImputer(d_in=d, d_model=64, n_heads=4, n_layers=2, n_samples=4)
    imputer.eval()
    X_imp, sigma = imputer(X, M)

    # sigma should have same shape and be finite
    assert sigma.shape == X_imp.shape
    assert torch.isfinite(sigma).all()
    # Missing entries should have sigma >= small floor
    missing_mask = (M == 0)
    if missing_mask.any():
        assert (sigma[missing_mask] >= 1e-6).all()
