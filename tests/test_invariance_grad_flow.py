import torch
from src.models.invariance import IRMStructureInvariance


def test_logits_grad_flow():
    d = 4
    n_envs = 2
    loss_mod = IRMStructureInvariance(n_features=d, n_envs=n_envs)

    # Create dummy adjacency and logits that require grad
    logits = torch.randn(2, d, d, requires_grad=True)
    A = torch.sigmoid(logits).detach()  # A derived from logits in real pipeline

    # Create simple X, M, e
    B, T = 2, 5
    X = torch.randn(B, T, d)
    M = torch.ones(B, T, d)
    e = torch.tensor([0, 1])

    total_loss, metrics = loss_mod(A, logits, X, M, e)
    total_loss.backward()

    # logits.grad should be populated
    assert logits.grad is not None
    assert torch.any(logits.grad != 0)
