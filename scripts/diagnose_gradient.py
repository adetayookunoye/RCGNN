#!/usr/bin/env python3
"""
Diagnostic script to find exactly why A_base has zero gradient.
Run this on Sapelo2 to identify the gradient bug.
"""
import sys
sys.path.insert(0, '/home/aoo29179/rcgnn')

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from src.models.rcgnn import RCGNN

def main():
    print("=" * 70)
    print("GRADIENT DIAGNOSTIC TEST")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load data
    data_path = Path('data/interim/uci_air')
    X = np.load(data_path / 'X.npy')[:32] # Just 32 samples
    M = np.load(data_path / 'M.npy')[:32]
    
    X_mean, X_std = X.mean(), X.std() + 1e-8
    X = (X - X_mean) / X_std
    
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    M_tensor = torch.tensor(M, dtype=torch.float32, device=device)
    
    d = X.shape[-1]
    print(f"d = {d}, X.shape = {X_tensor.shape}")
    
    # Create model
    model = RCGNN(d=d, latent_dim=32, hidden_dim=64, device=device)
    model.to(device)
    
    # ==========================================================================
    # TEST 1: Is A_base a proper Parameter in the optimizer?
    # ==========================================================================
    print("\n" + "=" * 70)
    print("TEST 1: Parameter Registration")
    print("=" * 70)
    
    print(f"A_base type: {type(model.structure_learner.A_base)}")
    print(f"A_base is nn.Parameter: {isinstance(model.structure_learner.A_base, nn.Parameter)}")
    print(f"A_base requires_grad: {model.structure_learner.A_base.requires_grad}")
    print(f"A_base is leaf: {model.structure_learner.A_base.is_leaf}")
    
    param_names = [n for n, _ in model.named_parameters()]
    print(f"A_base in named_parameters: {'structure_learner.A_base' in param_names}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    opt_param_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    print(f"A_base in optimizer: {id(model.structure_learner.A_base) in opt_param_ids}")
    
    # ==========================================================================
    # TEST 2: Does A_soft have gradient tracking?
    # ==========================================================================
    print("\n" + "=" * 70)
    print("TEST 2: Gradient Tracking through Forward Pass")
    print("=" * 70)
    
    model.train()
    out = model(X_tensor, M_tensor)
    
    A_soft = out['A_soft']
    print(f"A_soft requires_grad: {A_soft.requires_grad}")
    print(f"A_soft grad_fn: {A_soft.grad_fn}")
    print(f"A_soft grad_fn is None: {A_soft.grad_fn is None}")
    
    X_recon = out['X_recon']
    print(f"X_recon requires_grad: {X_recon.requires_grad}")
    print(f"X_recon grad_fn: {X_recon.grad_fn}")
    
    # ==========================================================================
    # TEST 3: Toy loss directly on A
    # ==========================================================================
    print("\n" + "=" * 70)
    print("TEST 3: Toy Loss (Direct Dependency on A)")
    print("=" * 70)
    
    optimizer.zero_grad()
    
    # This loss DIRECTLY depends on A_base via A_soft
    toy_loss = -A_soft.mean()
    print(f"toy_loss = {toy_loss.item():.6f}")
    
    toy_loss.backward()
    
    if model.structure_learner.A_base.grad is not None:
        grad_norm = model.structure_learner.A_base.grad.abs().mean().item()
        print(f"||grad(A_base)|| after toy_loss: {grad_norm:.6f}")
        if grad_norm > 0:
            print("[OK] A_base gets gradient from direct A dependency!")
        else:
            print("[X] PROBLEM: A_base grad is still 0 even with direct dependency")
    else:
        print("[X] PROBLEM: A_base.grad is None!")
    
    # ==========================================================================
    # TEST 4: Reconstruction loss gradient
    # ==========================================================================
    print("\n" + "=" * 70)
    print("TEST 4: Reconstruction Loss Gradient")
    print("=" * 70)
    
    optimizer.zero_grad()
    out = model(X_tensor, M_tensor)
    X_recon = out['X_recon']
    A_soft = out['A_soft']
    gate = out['gate']
    
    recon_loss = ((X_recon - X_tensor) ** 2 * M_tensor).sum() / (M_tensor.sum() + 1e-8)
    print(f"recon_loss = {recon_loss.item():.6f}")
    print(f"gate = {gate.item():.4f} (1-gate = {1-gate.item():.4f})")
    
    recon_loss.backward(retain_graph=True)
    
    if model.structure_learner.A_base.grad is not None:
        grad_norm = model.structure_learner.A_base.grad.abs().mean().item()
        print(f"||grad(A_base)|| after recon_loss: {grad_norm:.6f}")
        if grad_norm > 0:
            print("[OK] A_base gets gradient from recon_loss!")
        else:
            print("[X] PROBLEM: A_base grad is 0 from recon_loss")
            print(" This means the message passing path is not contributing")
    else:
        print("[X] PROBLEM: A_base.grad is None!")
    
    # ==========================================================================
    # TEST 5: Force gate=0 and check gradient
    # ==========================================================================
    print("\n" + "=" * 70)
    print("TEST 5: Force Gate=0 (100% through A path)")
    print("=" * 70)
    
    # Temporarily set gate to force all signal through A
    with torch.no_grad():
        model.gate_alpha.fill_(-10.0) # sigmoid(-10) ≈ 0.00005
    
    optimizer.zero_grad()
    out = model(X_tensor, M_tensor)
    X_recon = out['X_recon']
    gate = out['gate']
    
    recon_loss = ((X_recon - X_tensor) ** 2 * M_tensor).sum() / (M_tensor.sum() + 1e-8)
    print(f"recon_loss (gate≈0) = {recon_loss.item():.6f}")
    print(f"gate = {gate.item():.6f}")
    
    recon_loss.backward()
    
    if model.structure_learner.A_base.grad is not None:
        grad_norm = model.structure_learner.A_base.grad.abs().mean().item()
        print(f"||grad(A_base)|| with gate≈0: {grad_norm:.6f}")
        if grad_norm > 0:
            print("[OK] A_base gets gradient when gate=0!")
            print(" -> The FIX is to initialize gate_alpha much lower (e.g., -5)")
        else:
            print("[X] PROBLEM: A_base grad is STILL 0 even with gate=0")
            print(" -> There's a deeper issue in the message passing path")
    else:
        print("[X] PROBLEM: A_base.grad is None!")
    
    # ==========================================================================
    # TEST 6: Check gradient flow step by step
    # ==========================================================================
    print("\n" + "=" * 70)
    print("TEST 6: Detailed Gradient Flow Analysis")
    print("=" * 70)
    
    optimizer.zero_grad()
    
    # Manual forward pass with gradient checks
    B, T, d = X_tensor.shape
    z_s, z_n, z_b = model.tri_encoder(X_tensor)
    
    # Structure learner forward
    A_logits = model.structure_learner.A_base.unsqueeze(0).expand(B, -1, -1).clone()
    print(f"A_logits requires_grad: {A_logits.requires_grad}")
    print(f"A_logits grad_fn: {A_logits.grad_fn}")
    
    A_soft = torch.sigmoid(A_logits / model.structure_learner.temperature) * model.structure_learner.diag_mask.unsqueeze(0)
    print(f"A_soft requires_grad: {A_soft.requires_grad}")
    print(f"A_soft grad_fn: {A_soft.grad_fn}")
    
    # Signal projection
    z_s_flat = z_s.reshape(B * T, -1)
    signal_features = model.signal_projector(z_s_flat).reshape(B, T, d)
    print(f"signal_features requires_grad: {signal_features.requires_grad}")
    
    # Message passing
    X_msg = torch.bmm(signal_features, A_soft.transpose(1, 2))
    print(f"X_msg requires_grad: {X_msg.requires_grad}")
    print(f"X_msg grad_fn: {X_msg.grad_fn}")
    
    # Test gradient on X_msg
    test_loss = X_msg.mean()
    test_loss.backward()
    
    if model.structure_learner.A_base.grad is not None:
        grad_norm = model.structure_learner.A_base.grad.abs().mean().item()
        print(f"||grad(A_base)|| from X_msg.mean(): {grad_norm:.6f}")
    else:
        print("A_base.grad is None after X_msg.mean().backward()")
    
    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)

if __name__ == '__main__':
    main()
