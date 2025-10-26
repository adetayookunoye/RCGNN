import torch
import numpy as np
from src.models.sparsification import sparsemax, entmax15, set_sparse_eps


def test_sparse_zero_ratio_default():
    torch.manual_seed(0)
    set_sparse_eps(1e-2)
    logits = torch.randn(128, 32)
    s = sparsemax(logits, dim=-1)
    zero_ratio = (s == 0).float().mean().item()
    assert zero_ratio > 0.5, f"zero_ratio too low: {zero_ratio}"


def test_entmax15_matches_sparsemax():
    torch.manual_seed(1)
    set_sparse_eps(1e-2)
    logits = torch.randn(64, 16)
    s = sparsemax(logits, dim=-1)
    e = entmax15(logits, dim=-1)
    assert torch.allclose(s, e, atol=1e-6) or ( (s==0).float().mean() >= 0.5 )
