"""Test to ensure gradient flow works correctly in training loop.

This test validates the fix for the gradient flow issue where total_loss
was incorrectly initialized as 0.0 (float) instead of None, causing
gradient backpropagation to fail.
"""

import torch
import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from torch.utils.data import DataLoader
from src.dataio.loaders import load_synth
from src.models.rcgnn import RCGNN
from src.training.optim import make_optimizer
from src.training.loop import train_epoch
import yaml
import os


def test_gradient_flow_in_training():
    """Test that gradients flow correctly through the training loop."""
    # Load minimal configs
    repo_root = Path(__file__).parents[1]
    
    with open(repo_root / "configs/data.yaml") as f:
        dc = yaml.safe_load(f)
    with open(repo_root / "configs/model.yaml") as f:
        mc = yaml.safe_load(f)
    with open(repo_root / "configs/train.yaml") as f:
        tc = yaml.safe_load(f)
    
    # Load minimal dataset
    root = os.path.join(dc["paths"]["root"], "interim", "synth_small")
    train_ds = load_synth(root, "train", seed=tc["seed"])
    train_ld = DataLoader(train_ds, batch_size=2, shuffle=False)
    
    # Create model
    d = train_ds.X.shape[-1]
    model = RCGNN(d, mc)
    device = tc["device"]
    model.to(device)
    
    # Create optimizer
    opt = make_optimizer(model.parameters(), tc)
    
    # Run one training epoch
    out = train_epoch(
        model, train_ld, opt, 
        inv_weight=mc["loss"]["invariance"]["lambda_inv"], 
        device=device
    )
    
    # Validate that training produced valid loss
    assert "loss" in out
    assert isinstance(out["loss"], (int, float))
    assert not torch.isnan(torch.tensor(out["loss"]))
    assert not torch.isinf(torch.tensor(out["loss"]))
    
    # Check that gradients were computed and are meaningful
    grad_count = 0
    zero_grad_count = 0
    nan_grad_count = 0
    nonzero_grad_count = 0
    
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_count += 1
            grad_norm = p.grad.norm().item()
            
            if torch.isnan(p.grad).any():
                nan_grad_count += 1
            elif grad_norm == 0:
                zero_grad_count += 1
            else:
                nonzero_grad_count += 1
    
    # Assert gradient health
    assert grad_count > 0, "No gradients were computed"
    assert nan_grad_count == 0, f"Found {nan_grad_count} parameters with NaN gradients"
    
    # At least 50% of parameters should have non-zero gradients
    # (Some parameters legitimately have zero gradients if not used in forward pass)
    assert nonzero_grad_count > grad_count * 0.5, \
        f"Too many zero gradients: {nonzero_grad_count}/{grad_count} have non-zero grads"
    
    print(f"\n✓ Gradient flow test passed:")
    print(f"  - {grad_count} parameters with gradients")
    print(f"  - {nonzero_grad_count} with non-zero gradients")
    print(f"  - {zero_grad_count} with zero gradients (normal)")
    print(f"  - {nan_grad_count} with NaN gradients")


def test_loss_accumulation_preserves_graph():
    """Test that loss accumulation preserves computation graph."""
    # Create simple tensors with computation graph
    x = torch.randn(3, 4, requires_grad=True)
    w = torch.randn(4, 2, requires_grad=True)
    
    # Simulate accumulation pattern from training loop
    total_loss = None
    for i in range(3):
        loss = (x[i:i+1] @ w).sum()
        if total_loss is None:
            total_loss = loss
        else:
            total_loss = total_loss + loss
    
    # Verify loss has grad_fn (computation graph)
    assert total_loss.grad_fn is not None, \
        "Loss should have grad_fn to preserve computation graph"
    assert not total_loss.is_leaf, \
        "Loss should not be a leaf tensor when accumulated from tensors"
    
    # Test backward pass
    total_loss.backward()
    
    # Verify gradients were computed
    assert x.grad is not None, "x should have gradients"
    assert w.grad is not None, "w should have gradients"
    assert not torch.isnan(x.grad).any(), "x gradients should not be NaN"
    assert not torch.isnan(w.grad).any(), "w gradients should not be NaN"
    assert x.grad.abs().sum() > 0, "x gradients should be non-zero"
    assert w.grad.abs().sum() > 0, "w gradients should be non-zero"
    
    print("\n✓ Loss accumulation test passed")


def test_loss_tensor_from_float_fails():
    """Test that creating tensor from float with requires_grad breaks gradient flow.
    
    This demonstrates the bug that was fixed - it should show that the old
    approach doesn't work.
    """
    x = torch.randn(2, 3, requires_grad=True)
    w = torch.randn(3, 1, requires_grad=True)
    
    # Simulate the OLD buggy pattern
    y = (x @ w).sum()
    loss_float = float(y)  # Convert to float
    
    # Create tensor from float with requires_grad=True (the bug!)
    loss_tensor = torch.tensor(loss_float, requires_grad=True)
    
    # This tensor is a leaf with no grad_fn
    assert loss_tensor.is_leaf, "Tensor from float is a leaf"
    assert loss_tensor.grad_fn is None, "Tensor from float has no grad_fn"
    
    # Backward pass won't propagate to original parameters
    loss_tensor.backward()
    
    # Original parameters don't get gradients
    assert x.grad is None, "x should not have gradients (disconnected)"
    assert w.grad is None, "w should not have gradients (disconnected)"
    
    print("\n✓ Float conversion bug demonstration passed")


if __name__ == "__main__":
    test_gradient_flow_in_training()
    test_loss_accumulation_preserves_graph()
    test_loss_tensor_from_float_fails()
    print("\n✅ All gradient flow tests passed!")
