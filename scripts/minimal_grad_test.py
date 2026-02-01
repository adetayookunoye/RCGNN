#!/usr/bin/env python3
"""
Ultra-minimal gradient test for bmm with A_base.
"""
import torch
import torch.nn as nn

print("=" * 70)
print("MINIMAL BMM GRADIENT TEST")
print("=" * 70)

# Minimal test
d = 13
B = 4
T = 24

# The parameter
A_base = nn.Parameter(torch.randn(d, d) * 0.1 - 3.0)
diag_mask = 1.0 - torch.eye(d)
tau = 0.25

# Some input
signal_features = torch.randn(B, T, d, requires_grad=True)

# Forward
A_logits = A_base.unsqueeze(0).expand(B, -1, -1).clone()
A_soft = torch.sigmoid(A_logits / tau) * diag_mask.unsqueeze(0)
X_msg = torch.bmm(signal_features, A_soft.transpose(1, 2))

print(f"A_base.requires_grad: {A_base.requires_grad}")
print(f"A_logits.requires_grad: {A_logits.requires_grad}")
print(f"A_soft.requires_grad: {A_soft.requires_grad}")
print(f"X_msg.requires_grad: {X_msg.requires_grad}")
print(f"A_soft.grad_fn: {A_soft.grad_fn}")
print(f"X_msg.grad_fn: {X_msg.grad_fn}")

print(f"\nA_base stats: min={A_base.min():.4f}, max={A_base.max():.4f}")
print(f"A_soft stats: min={A_soft.min():.6f}, max={A_soft.max():.6f}, mean={A_soft.mean():.6f}")

# Backward
loss = X_msg.mean()
loss.backward()

if A_base.grad is not None:
    grad_norm = A_base.grad.abs().mean().item()
    grad_max = A_base.grad.abs().max().item()
    print(f"\n||grad(A_base)||.mean(): {grad_norm:.10f}")
    print(f"||grad(A_base)||.max(): {grad_max:.10f}")
    
    if grad_norm < 1e-10:
        print("\n[WARN] Gradient is effectively zero!")
        print("Reason: A_soft values are EXTREMELY small (sigmoid(-12) ≈ 6e-6)")
        print("The gradient of sigmoid at x=-12 is sigmoid(-12)*(1-sigmoid(-12)) ≈ 6e-6")
        print("This gets multiplied through the chain, giving near-zero gradient.")
        
        print("\n--- Let's try with W_adj initialized near 0 ---")
        A_base2 = nn.Parameter(torch.randn(d, d) * 0.5) # Around 0, sigmoid(0)=0.5
        A_logits2 = A_base2.unsqueeze(0).expand(B, -1, -1).clone()
        A_soft2 = torch.sigmoid(A_logits2 / tau) * diag_mask.unsqueeze(0)
        X_msg2 = torch.bmm(signal_features.detach().clone().requires_grad_(True), A_soft2.transpose(1, 2))
        
        print(f"A_base2 stats: min={A_base2.min():.4f}, max={A_base2.max():.4f}")
        print(f"A_soft2 stats: min={A_soft2.min():.6f}, max={A_soft2.max():.6f}")
        
        loss2 = X_msg2.mean()
        loss2.backward()
        
        grad_norm2 = A_base2.grad.abs().mean().item()
        grad_max2 = A_base2.grad.abs().max().item()
        print(f"\n||grad(A_base2)||.mean(): {grad_norm2:.10f}")
        print(f"||grad(A_base2)||.max(): {grad_max2:.10f}")
        
        if grad_norm2 > 1e-6:
            print("\n[OK] With W_adj near 0, gradient exists!")
            print(" THE FIX: Initialize A_base near 0, not at -3.0")
else:
    print("A_base.grad is None!")
