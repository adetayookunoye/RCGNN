import torch
from src.models.sparse_attention import LowRankProjection


def test_lowrank_projection_shapes():
    B, T = 2, 4
    d = 8
    rank = 4

    # Case A: LowRankProjection(d, rank) -> output should preserve d
    proj_a = LowRankProjection(d, rank)
    x = torch.randn(B, T, d)
    y = proj_a(x)
    assert y.shape == (B, T, d)

    # Case B: LowRankProjection(d_in, d_out, rank)
    d_out = 6
    proj_b = LowRankProjection(d, d_out, rank)
    y2 = proj_b(x)
    assert y2.shape == (B, T, d_out)
